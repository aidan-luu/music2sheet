"""Tests for :class:`ml.models.mert.MERTFeatureExtractor`.

The heavyweight forward-pass test is gated on ``RUN_SLOW=1`` so CI does
not have to pull the ~1.3 GB MERT-v1-330M checkpoint on every run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ml.models.mert import MERTFeatureExtractor


def test_init_is_lazy() -> None:
    """Constructor must NOT touch the network / GPU."""
    extractor = MERTFeatureExtractor()
    assert extractor._model is None
    assert extractor._processor is None


def test_device_autodetect() -> None:
    """Without an explicit device, pick a valid torch device string."""
    torch = pytest.importorskip("torch")

    extractor = MERTFeatureExtractor()
    assert extractor.device in {"cuda", "mps", "cpu"}
    # Confirm torch can actually parse the chosen device string.
    torch.device(extractor.device)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Slow test: downloads MERT-v1-330M (~1.3 GB). Set RUN_SLOW=1 to enable.",
)
def test_extract_smoke() -> None:
    """End-to-end smoke test on 3 s of synthetic audio."""
    sr = 44100
    duration_s = 3
    # Simple 440 Hz sine wave so the model sees a benign input distribution.
    t = np.arange(sr * duration_s, dtype=np.float32) / sr
    audio = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    extractor = MERTFeatureExtractor()
    features = extractor.extract(audio, sr=sr)

    assert isinstance(features, np.ndarray)
    assert features.ndim == 2

    expected_T = 75 * duration_s
    assert abs(features.shape[0] - expected_T) <= 20, (
        f"time dim {features.shape[0]} too far from expected {expected_T}"
    )

    # MERT-v1-330M hidden size.
    assert features.shape[1] == 1024
