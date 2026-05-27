"""Unit tests for the HT-Demucs v4 wrapper and audio_io helpers.

Agent C writes these as part of PR-1 to lock the public surface; Agent D
owns the broader test suite long-term and will fold these into the regular
QA runs.

The ``test_separate_smoke`` test is marked ``slow`` because it downloads
the htdemucs_ft checkpoint (~80 MB) on first run. It is skipped unless
``RUN_SLOW=1`` is set in the environment. NOTE: this file uses a custom
"slow" marker. If pytest emits an "unknown marker" warning, Agent D should
register the marker in pyproject.toml's [tool.pytest.ini_options.markers]
list. A ticket to that effect lives at
.orchestrator/tickets/pr-1-register-slow-marker.json.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ml.audio_io import audio_hash
from ml.models.demucs import DEMUCS_SR, STEM_NAMES, DemucsWrapper


def test_wrapper_init_is_lazy() -> None:
    """Constructing a wrapper must not load the model (CI would download ~80 MB)."""
    wrapper = DemucsWrapper()
    # The model field is the lazy-load sentinel; it must stay None until
    # separate() (or _ensure_model_loaded) is called.
    assert wrapper._model is None
    # Construction also exposes the requested model name + a resolved device.
    assert wrapper.model_name == "htdemucs_ft"
    assert wrapper.device in {"cuda", "mps", "cpu"}


def test_wrapper_respects_explicit_device(tmp_path: Path) -> None:
    """Passing device=... overrides the autodetect path."""
    wrapper = DemucsWrapper(device="cpu", cache_dir=tmp_path)
    assert wrapper.device == "cpu"
    assert wrapper.cache_dir == tmp_path


def test_audio_hash_stable(tmp_path: Path) -> None:
    """Hashing the same byte content twice yields identical digests."""
    payload = b"sheet-sage-2 deterministic payload \x00\x01\x02"
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(payload)
    b.write_bytes(payload)

    h_a = audio_hash(a)
    h_b = audio_hash(b)

    assert h_a == h_b
    assert len(h_a) == 16
    assert all(c in "0123456789abcdef" for c in h_a)


def test_audio_hash_changes_with_content(tmp_path: Path) -> None:
    """A single byte difference must produce a different hash."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"sheet-sage-2-A")
    b.write_bytes(b"sheet-sage-2-B")

    assert audio_hash(a) != audio_hash(b)


def _write_synthetic_wav(path: Path, duration_s: float = 2.0, sr: int = DEMUCS_SR) -> int:
    """Write a deterministic synthetic stereo WAV. Returns the per-channel sample count."""
    import soundfile as sf

    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(seed=0xC0FFEE)
    # Left channel: a 440 Hz sine; right channel: a 660 Hz sine plus a sliver
    # of noise. Amplitude well below clipping.
    left = 0.3 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    right = (
        0.3 * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
        + 0.02 * rng.standard_normal(n).astype(np.float32)
    )
    stereo = np.stack([left, right], axis=1)  # soundfile wants (samples, channels)
    sf.write(str(path), stereo, sr, subtype="FLOAT")
    return n


# Custom 'slow' marker — see module docstring for the marker-registration ticket.
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Slow test: downloads htdemucs_ft (~80 MB). Set RUN_SLOW=1 to enable.",
)
def test_separate_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test of HT-Demucs separation on synthetic audio."""
    wav_path = tmp_path / "synthetic.wav"
    n_samples = _write_synthetic_wav(wav_path, duration_s=2.0, sr=DEMUCS_SR)

    wrapper = DemucsWrapper(cache_dir=tmp_path / "cache")
    stems = wrapper.separate(wav_path)

    # Standard 4-stem layout.
    assert set(stems.keys()) == set(STEM_NAMES)

    # Each stem must be (channels, samples) float32. demucs pads + chunks,
    # so allow a small tolerance on sample count.
    for name, arr in stems.items():
        assert arr.dtype == np.float32, f"{name} dtype: {arr.dtype}"
        assert arr.ndim == 2, f"{name} ndim: {arr.ndim}"
        assert arr.shape[0] in (1, 2), f"{name} channels: {arr.shape[0]}"
        assert abs(arr.shape[1] - n_samples) <= 1024, (
            f"{name} sample count {arr.shape[1]} not within 1024 of {n_samples}"
        )
