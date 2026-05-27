"""PyTorch ``Dataset`` wrappers around the melody training pipeline (PR-4b).

These wrappers consume a low-level annotation dataset (e.g.
:class:`ml.datasets.hooktheory.HookTheoryDataset`) and lazily run the
:class:`MelodyDataPipeline` per item, optionally caching the resulting
``(encoder_feats, target_tokens)`` arrays to disk so re-running the same
training job does not re-pay the Demucs + MERT cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.datasets.hooktheory import HookTheoryDataset
from ml.training.melody_pipeline import MelodyDataPipeline
from ml.types import Note

_CACHE_SUFFIX = ".melody_train.npz"


class MelodyTrainingDataset(Dataset):
    """``Dataset`` adapter that produces melody training tensors on demand.

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
        underlying: HookTheoryDataset,
        pipeline: MelodyDataPipeline,
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
                notes=_coerce_notes(entry.get("notes", [])),
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
def _coerce_notes(raw: Any) -> list[Note]:
    """Accept either a list of :class:`Note` objects or a list of dicts.

    The HookTheory manifest stores notes as plain dicts on disk; tests and
    in-memory fakes often pass pre-built :class:`Note` instances. Both must
    work without forcing every caller through a serialization step.
    """
    out: list[Note] = []
    for item in raw or []:
        if isinstance(item, Note):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
                Note(
                    pitch=int(item["pitch"]),
                    onset=float(item["onset"]),
                    duration=float(item["duration"]),
                    velocity=int(item.get("velocity", 80)),
                )
            )
        else:
            raise TypeError(f"Unsupported note entry: {type(item).__name__}")
    return out


def _entry_duration(entry: dict[str, Any]) -> float:
    """Return the annotated duration of an entry in seconds.

    The manifest may carry an explicit ``audio_duration_s`` field; otherwise
    we fall back to the latest note offset (``onset + duration``) which is
    always available and never under-estimates the audio length for the
    purpose of the long-example filter.
    """
    if "audio_duration_s" in entry:
        return float(entry["audio_duration_s"])
    notes = entry.get("notes") or []
    last_offset = 0.0
    for note in notes:
        if isinstance(note, Note):
            offset = note.onset + note.duration
        else:
            offset = float(note["onset"]) + float(note["duration"])
        if offset > last_offset:
            last_offset = offset
    return last_offset


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write ``arrays`` to ``path`` via a sibling ``.tmp.npz`` then rename.

    ``np.savez`` auto-appends ``.npz`` if the target path lacks that
    extension, so we stage to ``<name>.tmp.npz`` and rename to ``<name>``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)


__all__ = ["MelodyTrainingDataset"]
