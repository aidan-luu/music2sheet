"""Chord training data pipeline (PR-6).

Mirror of :mod:`ml.training.melody_pipeline`, but for the chord head:

    audio file -> mono mix at 44.1 kHz -> MERT features (75 Hz)
                                       -> tokenizer.encode_sequence -> tokens

Key differences from the melody pipeline
----------------------------------------
* **No Demucs.** The chord head sees the full harmonic content of the mix;
  isolating any stem would discard exactly the bass-and-comping signal the
  head needs. A future ablation could plug in the Demucs "other" stem
  (drums + bass + accompaniment minus vocals) but the v0 contract is full
  mix.
* **Chord segments instead of notes.** ``prepare`` consumes a
  ``list[Chord]`` and uses :meth:`ChordTokenizer.encode_sequence` to map
  each segment to its covering range of frames.

The output :class:`ChordTrainingExample` honours the same alignment
invariant as the melody pipeline:
``encoder_feats.shape[0] == target_tokens.shape[0]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.datasets.isophonics import IsophonicsDataset
from ml.models.chord_tokenizer import ChordTokenizer
from ml.models.mert import MERT_NATIVE_SR, MERTFeatureExtractor
from ml.types import Chord

_CACHE_SUFFIX = ".chord_train.npz"


@dataclass
class ChordTrainingExample:
    """One framewise chord example ready for the head.

    ``encoder_feats.shape[0] == target_tokens.shape[0]`` always holds.
    """

    audio_id: str
    encoder_feats: np.ndarray  # (T_enc, 1024) MERT features
    target_tokens: np.ndarray  # (T_enc,) framewise token ids
    audio_duration_s: float


def _load_mono(audio_path: Path, target_sr: int) -> np.ndarray:
    """Read ``audio_path`` and return a mono float32 waveform at ``target_sr``.

    Wraps ``librosa.load`` so callers don't need to know which loader is in
    use; centralising it also makes the slow-test mock easy to swap.
    """
    waveform, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return waveform.astype(np.float32, copy=False)


class ChordDataPipeline:
    """End-to-end audio -> (MERT features, framewise chord tokens) pipeline.

    All wrapped components are lazy: constructing the pipeline does not
    download or load any model. The wrappers themselves are kept as fields
    so callers can share a single instance across many ``prepare`` calls.

    Parameters
    ----------
    mert:
        Optional pre-constructed :class:`MERTFeatureExtractor`. Lazy.
    tokenizer:
        Optional pre-constructed :class:`ChordTokenizer` (at 75 Hz).
    """

    def __init__(
        self,
        mert: MERTFeatureExtractor | None = None,
        tokenizer: ChordTokenizer | None = None,
    ) -> None:
        self.mert: MERTFeatureExtractor = (
            mert if mert is not None else MERTFeatureExtractor()
        )
        self.tokenizer: ChordTokenizer = (
            tokenizer if tokenizer is not None else ChordTokenizer(frame_rate_hz=75.0)
        )

    def prepare(
        self,
        audio_path: str | Path,
        chords: list[Chord],
    ) -> ChordTrainingExample:
        """Build a :class:`ChordTrainingExample` for ``audio_path`` + ``chords``.

        Pipeline steps:

        1. Load the audio as mono at MERT's native sample rate.
        2. Run MERT to extract 75 Hz features on the full mix.
        3. Encode ``chords`` to a framewise token stream at the same rate.
        4. Trim / right-pad the tokens to match ``encoder_feats.shape[0]``
           exactly so the alignment invariant holds.
        """
        audio_path = Path(audio_path)

        waveform = _load_mono(audio_path, target_sr=MERT_NATIVE_SR)
        audio_duration_s = float(waveform.shape[0]) / float(MERT_NATIVE_SR)

        encoder_feats = self.mert.extract(waveform, sr=MERT_NATIVE_SR)
        if encoder_feats.ndim != 2:
            raise ValueError(
                f"MERT.extract returned ndim={encoder_feats.ndim}, expected 2"
            )
        t_enc = int(encoder_feats.shape[0])

        # Encode at the audio's true duration, then force-align to T_enc so
        # the invariant holds regardless of MERT's internal padding / chunking.
        raw_tokens = self.tokenizer.encode_sequence(
            chords, audio_duration_s=audio_duration_s
        )
        target_tokens = _align_tokens_to_length(
            raw_tokens, target_len=t_enc, pad_value=self.tokenizer.N
        )

        return ChordTrainingExample(
            audio_id=audio_path.stem,
            encoder_feats=encoder_feats.astype(np.float32, copy=False),
            target_tokens=target_tokens,
            audio_duration_s=audio_duration_s,
        )


class ChordTrainingDataset(Dataset):
    """``Dataset`` adapter that produces chord training tensors on demand.

    Each ``__getitem__`` call:

    1. Reads one annotation entry from ``underlying``.
    2. Returns the cached arrays if ``cache_dir / "<audio_id><suffix>"``
       exists.
    3. Otherwise runs ``pipeline.prepare`` and writes the cache.

    Entries whose annotated duration exceeds ``max_duration_s`` are pruned
    at construction time so ``__len__`` and ``__getitem__`` agree.
    """

    def __init__(
        self,
        underlying: IsophonicsDataset,
        pipeline: ChordDataPipeline,
        cache_dir: Path | None = None,
        max_duration_s: float = 30.0,
    ) -> None:
        self.underlying = underlying
        self.pipeline = pipeline
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self.max_duration_s: float = float(max_duration_s)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._indices: list[int] = [
            i
            for i in range(len(underlying))
            if _entry_duration(underlying[i]) <= self.max_duration_s
        ]

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not 0 <= idx < len(self._indices):
            raise IndexError(idx)
        entry = self.underlying[self._indices[idx]]

        cache_path = self._cache_path_for(entry)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as data:
                encoder_feats = data["encoder_feats"]
                target_tokens = data["target_tokens"]
        else:
            example = self.pipeline.prepare(
                audio_path=entry["audio_path"],
                chords=_coerce_chords(entry.get("chords", [])),
            )
            encoder_feats = example.encoder_feats
            target_tokens = example.target_tokens
            if cache_path is not None:
                _atomic_save_npz(
                    cache_path,
                    encoder_feats=encoder_feats,
                    target_tokens=target_tokens,
                )

        return {
            "encoder_feats": torch.from_numpy(
                np.ascontiguousarray(encoder_feats, dtype=np.float32)
            ),
            "target_tokens": torch.from_numpy(
                np.ascontiguousarray(target_tokens, dtype=np.int64)
            ),
        }

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _cache_path_for(self, entry: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        audio_id = str(entry.get("id") or Path(entry["audio_path"]).stem)
        return self.cache_dir / f"{audio_id}{_CACHE_SUFFIX}"


# ---------------------------------------------------------------------- #
# Private helpers
# ---------------------------------------------------------------------- #
def _coerce_chords(raw: Any) -> list[Chord]:
    """Accept either a list of :class:`Chord` objects or a list of dicts.

    Manifest entries store chords as plain dicts on disk; tests and in-memory
    fakes often pass pre-built :class:`Chord` instances. Both must work
    without forcing every caller through a serialization step.
    """
    out: list[Chord] = []
    for item in raw or []:
        if isinstance(item, Chord):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
                Chord(
                    label=str(item["label"]),
                    onset=float(item["onset"]),
                    duration=float(item["duration"]),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        else:
            raise TypeError(f"Unsupported chord entry: {type(item).__name__}")
    return out


def _entry_duration(entry: dict[str, Any]) -> float:
    """Return the annotated duration of an entry in seconds.

    Prefers an explicit ``audio_duration_s`` field; falls back to the latest
    chord offset (``onset + duration``) which is always available and never
    under-estimates the audio length for the purpose of the long-example
    filter.
    """
    if "audio_duration_s" in entry:
        return float(entry["audio_duration_s"])
    chords = entry.get("chords") or []
    last_offset = 0.0
    for chord in chords:
        if isinstance(chord, Chord):
            offset = chord.onset + chord.duration
        else:
            offset = float(chord["onset"]) + float(chord["duration"])
        if offset > last_offset:
            last_offset = offset
    return last_offset


def _align_tokens_to_length(
    tokens: np.ndarray, target_len: int, pad_value: int
) -> np.ndarray:
    """Trim or right-pad ``tokens`` so its length equals ``target_len``.

    Padding uses N (not PAD) because in the framewise scheme "no chord here"
    is the semantically correct label for any tail frames produced by MERT's
    chunking past the tokenizer's nominal duration. PAD is reserved for the
    collate function, where it marks frames outside a sequence's valid
    region in a batched tensor.
    """
    n = int(tokens.shape[0])
    if n == target_len:
        return tokens.astype(np.int64, copy=False)
    if n > target_len:
        return tokens[:target_len].astype(np.int64, copy=False)
    padding = np.full(target_len - n, pad_value, dtype=np.int64)
    return np.concatenate([tokens.astype(np.int64, copy=False), padding], axis=0)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write ``arrays`` to ``path`` via a sibling ``.tmp.npz`` then rename.

    ``np.savez`` auto-appends ``.npz`` if the target lacks the extension, so
    we stage to ``<name>.tmp.npz`` and rename to ``<name>``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)


__all__ = [
    "ChordDataPipeline",
    "ChordTrainingDataset",
    "ChordTrainingExample",
]
