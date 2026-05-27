"""HT-Demucs v4 source-separation wrapper.

Real implementation lands in PR-1. The stub here pins the public surface
so downstream pipeline code (PR-2+) can import the type without circular
deps during the skeleton phase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DemucsWrapper:
    """Thin wrapper over the official ``demucs`` package's HT-Demucs v4 model.

    The wrapped model separates an input mix into ``vocals``, ``drums``,
    ``bass``, and ``other`` stems at 44.1 kHz. The downstream chord head
    consumes the ``no-drums`` mix (sum of vocals + bass + other) and the
    melody head consumes the ``vocals`` stem.
    """

    def __init__(self, device: str = "cuda", model_name: str = "htdemucs") -> None:
        raise NotImplementedError("Implemented in PR-1 (HT-Demucs wrapper).")

    def infer(self, waveform: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """Run separation. Returns ``{stem_name: waveform_float32}``."""
        raise NotImplementedError("Implemented in PR-1 (HT-Demucs wrapper).")

    def infer_file(self, audio_path: Path) -> dict[str, np.ndarray]:
        """Convenience wrapper: load + infer."""
        raise NotImplementedError("Implemented in PR-1 (HT-Demucs wrapper).")
