"""HookTheory TheoryTab dataset loader.

License
-------
HookTheory's TheoryTab database is distributed under CC BY-NC-SA 3.0. Data is
community-contributed; users must respect the upstream TOS. We do NOT
redistribute raw HookTheory bytes — the fetch script at
``ml.datasets.scripts.fetch_hooktheory`` downloads SheetSage's released
JSON dump on demand and writes a normalised manifest.

Citation
--------
Donahue, C. et al. "Melody Transcription via Generative Pre-training," ISMIR
2022, references the HookTheory corpus and details the JSON schema.

Obtaining the data
------------------
Run::

    python -m ml.datasets.scripts.fetch_hooktheory --out ~/sheet-sage-data

then construct ``HookTheoryDataset("~/sheet-sage-data/manifests/hooktheory.jsonl")``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class HookTheoryDataset(MusicDataset):
    """HookTheory TheoryTab loader.

    Reads a JSON Lines manifest produced by
    ``ml.datasets.scripts.fetch_hooktheory`` and yields chord + melody
    annotations. Audio is not loaded here; ``audio_path`` is included in the
    returned dict for downstream code (``ml.audio_io``) to resolve.
    """

    def __init__(self, manifest_path: str | Path, split: str = "train") -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self._entries: list[dict[str, Any]] = _load_manifest(self.manifest_path, split)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._entries[index]


def _load_manifest(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"HookTheory manifest not found at {manifest_path}. "
            "Run `python -m ml.datasets.scripts.fetch_hooktheory --out <dir>` first."
        )
    out: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("split") == split:
                out.append(entry)
    return out
