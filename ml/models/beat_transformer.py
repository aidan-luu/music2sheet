"""Beat Transformer wrapper (beats + downbeats).

Real implementation lands in PR-2. The stub fixes the public class name
and the output type (``list[Beat]``) so the quantization engine (PR-11)
can be implemented against a typed contract.
"""

from __future__ import annotations

import numpy as np

from ml.types import Beat


class BeatTransformerWrapper:
    """Wrapper for the Beat Transformer model (Zhao et al., 2022).

    Consumes a mono float32 waveform and emits a list of :class:`Beat`
    events with the ``downbeat`` flag set on bar boundaries.
    """

    def __init__(self, device: str = "cuda", checkpoint: str | None = None) -> None:
        raise NotImplementedError("Implemented in PR-2 (Beat Transformer wrapper).")

    def infer(self, waveform: np.ndarray, sample_rate: int) -> list[Beat]:
        """Run beat + downbeat tracking; return the merged beat list."""
        raise NotImplementedError("Implemented in PR-2 (Beat Transformer wrapper).")
