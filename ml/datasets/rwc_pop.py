"""RWC Popular Music Database loader.

License
-------
The RWC Music Database is distributed by AIST under a non-commercial
research license; audio must be obtained directly from the RWC project
(https://staff.aist.go.jp/m.goto/RWC-MDB/) under a signed agreement.
Free chord/key annotations are distributed via tmc323/Chord-Annotations on
GitHub under that repo's research-use terms.

Citation
--------
Goto, M., Hashiguchi, H., Nishimura, T., and Oka, R. "RWC Music Database:
Popular, Classical and Jazz Music Databases," ISMIR 2002.

Obtaining the data
------------------
Run::

    python -m ml.datasets.scripts.fetch_rwc_pop --out ~/sheet-sage-data \\
        --audio-root /path/to/legally-acquired/rwc/audio

then construct ``RWCPopDataset("~/sheet-sage-data/manifests/rwc_pop.jsonl")``.

HELD-OUT EVAL SET
-----------------
RWC-Pop is the project's pass-criterion held-out test set. Every manifest
entry produced by the fetch script has ``split == "test"``. ``__init__``
defaults ``split="test"`` accordingly; constructing with any other split
returns an empty dataset (an explicit safety net against accidental leakage
into a training loop).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class RWCPopDataset(MusicDataset):
    """RWC Popular Music loader. HELD-OUT eval set (PR-5/6/8 pass-criterion)."""

    def __init__(self, manifest_path: str | Path, split: str = "test") -> None:
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
            f"RWC-Pop manifest not found at {manifest_path}. "
            "Run `python -m ml.datasets.scripts.fetch_rwc_pop --out <dir>` first."
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
