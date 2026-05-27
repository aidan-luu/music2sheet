"""Common dataset interface.

All ML datasets used by training jobs in PRs 4+ must subclass
:class:`MusicDataset` and yield items conforming to the schema below. This
guarantees that the training loops (and Agent D's eval harness) can iterate
HookTheory, POP909, Isophonics, Billboard, and RWC-Pop interchangeably.

Common item schema (yielded by ``__getitem__``)::

    {
        "audio_path": str,                       # path to a local audio file
        "beats":  list[ml.types.Beat],           # may be empty if dataset lacks beats
        "notes":  list[ml.types.Note],           # melody / multi-track notes
        "chords": list[ml.types.Chord],          # Harte-labelled segments
        "key":    ml.types.Key,                  # global key (or first key segment)
    }

Datasets without a particular annotation modality (e.g. POP909 has notes but
no Harte chord labels) yield an empty list for that field, never ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MusicDataset(ABC):
    """Abstract base class for music annotation datasets.

    Subclasses are typically constructed with a root directory pointing at the
    extracted, license-compliant copy of the dataset on disk. Subclasses MAY
    accept additional kwargs (e.g. split = "train") but MUST honour the common
    item schema documented at module level.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of items in this split."""

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one item following the common annotation schema."""
