"""Melody training data pipeline (PR-4b).

Assembles framewise training examples for the melody head by chaining the
PR-1/PR-3/PR-4a wrappers:

    audio file -> Demucs vocals stem (44.1 kHz) -> MERT features (75 Hz)
                                                -> tokenizer.encode -> tokens

The output :class:`MelodyTrainingExample` guarantees that ``encoder_feats``
and ``target_tokens`` share their leading time dimension, so a downstream
DataLoader + training loop (PR-4c) never has to align them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.models.demucs import DEMUCS_SR, DemucsWrapper
from ml.models.melody_tokenizer import MelodyTokenizer
from ml.models.mert import MERTFeatureExtractor
from ml.types import Note


@dataclass
class MelodyTrainingExample:
    """One framewise melody example ready for the decoder.

    ``encoder_feats.shape[0] == target_tokens.shape[0]`` always holds.
    """

    audio_id: str
    encoder_feats: np.ndarray  # (T_enc, 1024) MERT features
    target_tokens: np.ndarray  # (T_enc,) framewise token ids
    audio_duration_s: float


def _stem_to_mono(stem: np.ndarray) -> np.ndarray:
    """Collapse a Demucs stem to mono float32, shape ``(samples,)``.

    Demucs returns ``(channels, samples)`` at 44.1 kHz; MERT wants mono.
    """
    arr = np.asarray(stem, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported stem shape {arr.shape!r}; expected 1-D or 2-D")


class MelodyDataPipeline:
    """End-to-end audio -> (MERT features, framewise tokens) pipeline.

    All wrapped components are lazy: constructing the pipeline does not
    download or load any model. The wrappers themselves are kept as fields
    so callers can share a single instance across many ``prepare`` calls.

    Parameters
    ----------
    demucs:
        Optional pre-constructed :class:`DemucsWrapper`. A fresh one is
        instantiated lazily when omitted.
    mert:
        Optional pre-constructed :class:`MERTFeatureExtractor`. Lazy.
    tokenizer:
        Optional pre-constructed :class:`MelodyTokenizer` (at 75 Hz).
    use_vocals_stem:
        When ``True`` (default), MERT runs on the Demucs vocals stem — the
        recipe for the melody head. When ``False``, MERT runs on the full
        mix (downmixed to mono); useful as an ablation and for reuse by
        the chord head later.
    """

    def __init__(
        self,
        demucs: DemucsWrapper | None = None,
        mert: MERTFeatureExtractor | None = None,
        tokenizer: MelodyTokenizer | None = None,
        use_vocals_stem: bool = True,
    ) -> None:
        self.demucs: DemucsWrapper = demucs if demucs is not None else DemucsWrapper()
        self.mert: MERTFeatureExtractor = (
            mert if mert is not None else MERTFeatureExtractor()
        )
        self.tokenizer: MelodyTokenizer = (
            tokenizer if tokenizer is not None else MelodyTokenizer(frame_rate_hz=75.0)
        )
        self.use_vocals_stem: bool = use_vocals_stem

    def prepare(
        self,
        audio_path: str | Path,
        notes: list[Note],
    ) -> MelodyTrainingExample:
        """Build a :class:`MelodyTrainingExample` for ``audio_path`` + ``notes``.

        Pipeline steps:

        1. Run Demucs to produce stems (or skip the vocals stem and use the
           full mono mix when ``use_vocals_stem=False``).
        2. Feed the chosen mono waveform through MERT to obtain features at
           75 Hz.
        3. Encode ``notes`` to a framewise token stream at the same rate.
        4. Trim / right-pad the tokens to match ``encoder_feats.shape[0]``
           exactly so the alignment invariant holds.
        """
        audio_path = Path(audio_path)

        stems = self.demucs.separate(audio_path)
        if self.use_vocals_stem:
            if "vocals" not in stems:
                raise KeyError(
                    "Demucs output missing 'vocals' stem; cannot run melody pipeline"
                )
            waveform = _stem_to_mono(stems["vocals"])
        else:
            # Average all stems back together for the full-mix ablation. This
            # is mathematically equivalent (within float precision) to using
            # the original mix and avoids re-loading the audio file.
            mix = np.mean(
                np.stack([_stem_to_mono(s) for s in stems.values()], axis=0),
                axis=0,
            ).astype(np.float32, copy=False)
            waveform = mix

        audio_duration_s = float(waveform.shape[0]) / float(DEMUCS_SR)

        encoder_feats = self.mert.extract(waveform, sr=DEMUCS_SR)
        if encoder_feats.ndim != 2:
            raise ValueError(
                f"MERT.extract returned ndim={encoder_feats.ndim}, expected 2"
            )
        t_enc = int(encoder_feats.shape[0])

        # Encode with the audio's true duration, then force-align to T_enc
        # so the invariant ``encoder_feats.shape[0] == target_tokens.shape[0]``
        # is enforced regardless of MERT's internal padding / chunking.
        raw_tokens = self.tokenizer.encode(notes, audio_duration_s=audio_duration_s)
        target_tokens = _align_tokens_to_length(
            raw_tokens, target_len=t_enc, pad_value=self.tokenizer.REST
        )

        return MelodyTrainingExample(
            audio_id=audio_path.stem,
            encoder_feats=encoder_feats.astype(np.float32, copy=False),
            target_tokens=target_tokens,
            audio_duration_s=audio_duration_s,
        )


def _align_tokens_to_length(
    tokens: np.ndarray, target_len: int, pad_value: int
) -> np.ndarray:
    """Trim or right-pad ``tokens`` so its length equals ``target_len``.

    Padding uses REST (not PAD) because in the framewise scheme, "no melody
    here" is the semantically correct label for any tail frames produced by
    MERT chunking past the tokenizer's nominal duration. PAD is reserved for
    the collate function, where it marks frames outside a sequence's valid
    region in a batched tensor.
    """
    n = int(tokens.shape[0])
    if n == target_len:
        return tokens.astype(np.int64, copy=False)
    if n > target_len:
        return tokens[:target_len].astype(np.int64, copy=False)
    padding = np.full(target_len - n, pad_value, dtype=np.int64)
    return np.concatenate([tokens.astype(np.int64, copy=False), padding], axis=0)
