"""HookTheory TheoryTab dataset loader.

License
-------
HookTheory's TheoryTab database is distributed under HookTheory's own terms
of use; data is community-contributed. Users must obtain a copy via the
official API/scrape pipeline and accept their TOS. We do NOT redistribute.

Citation
--------
Donahue, C. et al. "Melody Transcription via Generative Pre-training," ISMIR
2022, references the HookTheory corpus and details the JSON schema.

Obtaining the data
------------------
See ``scripts/data/download_hooktheory.py`` (lands in PR-3 alongside the MERT
wrapper that consumes it). The script writes per-song JSON + audio links to
``data/hooktheory/`` and never re-uploads raw HookTheory bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class HookTheoryDataset(MusicDataset):
    """HookTheory TheoryTab loader.

    Yields chord + melody annotations aligned to beat positions; audio is
    fetched lazily from YouTube links in the source JSON.
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        raise NotImplementedError("Implemented in PR-4 (melody decoder port).")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in PR-4 (melody decoder port).")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("Implemented in PR-4 (melody decoder port).")
