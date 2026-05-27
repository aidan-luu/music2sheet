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
Clone https://github.com/music-x-lab/POP909-Dataset and point the loader at
the extracted ``POP909/`` directory. Audio derivatives can be synthesised
from the bundled MIDI via FluidSynth (see ``scripts/data/render_pop909.py``,
lands in PR-9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.datasets.base import MusicDataset


class POP909Dataset(MusicDataset):
    """Piano arrangement loader. Primary use: voicing model pretraining (PR-9)."""

    def __init__(self, root: Path, split: str = "train") -> None:
        raise NotImplementedError("Implemented in PR-9 (voicing model).")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in PR-9 (voicing model).")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("Implemented in PR-9 (voicing model).")
