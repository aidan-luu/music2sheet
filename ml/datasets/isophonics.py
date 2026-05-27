"""Isophonics reference annotations loader (Beatles, Carole King, Queen, Zweieck).

License
-------
Annotations are released under CC BY-NC-SA 4.0 by the Centre for Digital
Music at Queen Mary, University of London. Audio is NOT distributed; users
must source their own legally-acquired audio.

Citation
--------
Mauch, M., Cannam, C., Davies, M., Dixon, S., Harte, C., Kolozali, S.,
Tidhar, D., and Sandler, M. "OMRAS2 metadata project 2009," ISMIR 2009
(late-breaking demo).

Obtaining the data
------------------
Download annotation archives from http://isophonics.net/datasets and place
the unpacked ``*.lab`` files under ``data/isophonics/<artist>/``. Audio paths
are configured separately in a user-provided manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class IsophonicsDataset(MusicDataset):
    """Harte-format chord annotations + key labels. Primary use: PR-6/PR-8."""

    def __init__(self, root: Path, audio_manifest: Path, split: str = "train") -> None:
        raise NotImplementedError("Implemented in PR-6 (chord head).")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in PR-6 (chord head).")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("Implemented in PR-6 (chord head).")
