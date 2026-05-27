"""RWC Popular Music Database loader.

License
-------
The RWC Music Database is distributed by AIST under a non-commercial
research license; audio must be obtained directly from the RWC project
(https://staff.aist.go.jp/m.goto/RWC-MDB/) under a signed agreement.
Annotations (chord/key/beat) are distributed separately under the same
research-use terms.

Citation
--------
Goto, M., Hashiguchi, H., Nishimura, T., and Oka, R. "RWC Music Database:
Popular, Classical and Jazz Music Databases," ISMIR 2002.

Obtaining the data
------------------
1. Sign and return the RWC license to AIST to obtain audio CDs.
2. Download chord/key/beat annotations from
   https://github.com/tmc323/Chord-Annotations and place beside the audio.
3. Configure the loader via a manifest mapping ``RM-Pxxx`` ids to local
   audio paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class RWCPopDataset(MusicDataset):
    """RWC Popular Music loader. Multi-annotation: chord + key + beat. PR-6+."""

    def __init__(self, root: Path, audio_manifest: Path, split: str = "train") -> None:
        raise NotImplementedError("Implemented in PR-6 (chord head).")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in PR-6 (chord head).")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("Implemented in PR-6 (chord head).")
