"""McGill Billboard chord-annotation loader.

License
-------
The McGill Billboard Project annotations are released for academic use only;
see https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/
for the research-use agreement. Audio is NOT distributed.

Citation
--------
Burgoyne, J. A., Wild, J., and Fujinaga, I. "An Expert Ground Truth Set for
Audio Chord Recognition and Music Analysis," ISMIR 2011.

Obtaining the data
------------------
Run::

    python -m ml.datasets.scripts.fetch_billboard --out ~/sheet-sage-data --agree

then construct ``BillboardDataset("~/sheet-sage-data/manifests/billboard.jsonl")``.
Audio is absent, so ``audio_path`` is ``None`` in every yielded item; this is
fine for chord-head training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class BillboardDataset(MusicDataset):
    """SALAMI-format chord annotations from McGill Billboard. PR-6/PR-7 input."""

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
            f"Billboard manifest not found at {manifest_path}. "
            "Run `python -m ml.datasets.scripts.fetch_billboard --out <dir> --agree` first."
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
