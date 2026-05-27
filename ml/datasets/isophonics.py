"""Isophonics reference annotations loader (Beatles, Carole King, Queen, Zweieck, Robbie Williams).

License
-------
Annotations are released under CC BY-NC-SA 4.0 by the Centre for Digital
Music at Queen Mary, University of London. Audio is NOT distributed; users
must source their own legally-acquired audio if they need it.

Citation
--------
Mauch, M., Cannam, C., Davies, M., Dixon, S., Harte, C., Kolozali, S.,
Tidhar, D., and Sandler, M. "OMRAS2 metadata project 2009," ISMIR 2009
(late-breaking demo).

Obtaining the data
------------------
Run::

    python -m ml.datasets.scripts.fetch_isophonics --out ~/sheet-sage-data

then construct ``IsophonicsDataset("~/sheet-sage-data/manifests/isophonics.jsonl")``.
Audio is absent, so ``audio_path`` is ``None`` in every yielded item — this
loader is intended for chord-head training (PR-6) where MERT features are
precomputed offline against user-supplied audio (or, for purely chord-label
training, no audio is needed at all).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class IsophonicsDataset(MusicDataset):
    """Harte-format chord annotations. Primary use: PR-6/PR-8 chord head training."""

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
            f"Isophonics manifest not found at {manifest_path}. "
            "Run `python -m ml.datasets.scripts.fetch_isophonics --out <dir>` first."
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
