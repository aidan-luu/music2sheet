"""POP909 dataset loader (piano arrangements of Chinese pop songs).

License
-------
POP909 is released under CC BY-NC 4.0 by Wang et al. Non-commercial research
use only; redistribution must preserve attribution.

Citation
--------
Wang, Z. et al. "POP909: A Pop-song Dataset for Music Arrangement
Generation," ISMIR 2020.

Obtaining the data
------------------
Run::

    python -m ml.datasets.scripts.fetch_pop909 --out ~/sheet-sage-data

then construct ``POP909Dataset("~/sheet-sage-data/manifests/pop909.jsonl")``.
Audio is NOT included; the manifest's ``audio_path`` field references the
bundled aligned MIDI files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class POP909Dataset(MusicDataset):
    """Piano arrangement loader. Primary use: voicing model pretraining (PR-9)."""

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
            f"POP909 manifest not found at {manifest_path}. "
            "Run `python -m ml.datasets.scripts.fetch_pop909 --out <dir>` first."
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
