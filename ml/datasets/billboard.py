"""McGill Billboard chord-annotation loader.

License
-------
The McGill Billboard Project annotations are released for academic use only
under the terms posted at https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/
Audio is NOT distributed.

Citation
--------
Burgoyne, J. A., Wild, J., and Fujinaga, I. "An Expert Ground Truth Set for
Audio Chord Recognition and Music Analysis," ISMIR 2011.

Obtaining the data
------------------
Request access via the McGill DDMaL website. Annotations expand to one
folder per Billboard chart slot containing ``salami_chords.txt`` and a
metadata stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class BillboardDataset(MusicDataset):
    """SALAMI-format chord annotations from McGill Billboard. PR-6/PR-7 input."""

    def __init__(self, root: Path, audio_manifest: Path, split: str = "train") -> None:
        raise NotImplementedError("Implemented in PR-6 (chord head).")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in PR-6 (chord head).")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("Implemented in PR-6 (chord head).")
