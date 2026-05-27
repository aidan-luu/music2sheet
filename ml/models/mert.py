"""MERT-v1-330M feature-extractor wrapper.

Real implementation lands in PR-3. We expose only the typed surface here
so the melody/chord heads (PR-4..PR-7) can pin their inputs against a
stable API while the team works in parallel.
"""

from __future__ import annotations

import numpy as np


class MERTFeatureExtractor:
    """Hugging Face ``m-a-p/MERT-v1-330M`` wrapper.

    MERT operates at 24 kHz; the wrapper handles resampling internally.
    The extractor returns layer-wise hidden states; downstream heads
    learn per-layer weights (Chen et al., 2023).
    """

    def __init__(
        self,
        device: str = "cuda",
        target_sr: int = 24000,
        layer_selection: tuple[int, ...] | None = None,
    ) -> None:
        raise NotImplementedError("Implemented in PR-3 (MERT wrapper).")

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return hidden states with shape ``(num_layers, frames, hidden_dim)``.

        Frame rate is fixed at 75 Hz (the MERT default).
        """
        raise NotImplementedError("Implemented in PR-3 (MERT wrapper).")
