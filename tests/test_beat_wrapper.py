"""Tests for ``ml.models.beat_transformer.BeatTransformerWrapper``.

The slow test under ``RUN_SLOW=1`` exercises the real madmom pipeline on a
synthetic click track. The other three tests are pure unit checks of the
backend dispatch and do not require madmom to be importable.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from ml.models.beat_transformer import BeatTransformerWrapper
from ml.types import Beat


def test_init_default_backend_is_madmom() -> None:
    wrapper = BeatTransformerWrapper()
    assert wrapper.backend == "madmom"


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        BeatTransformerWrapper(backend="garbage")


def test_beat_transformer_backend_raises_not_implemented(tmp_path: Path) -> None:
    wrapper = BeatTransformerWrapper(backend="beat_transformer")
    fake_audio = tmp_path / "x.wav"
    fake_audio.touch()
    with pytest.raises(NotImplementedError):
        wrapper.detect(fake_audio)


def _write_click_track(path: Path, bpm: float = 120.0, duration_s: float = 5.0) -> None:
    """Write a 5 s, mono 44.1 kHz click track at the requested BPM."""
    sr = 44100
    n_samples = int(duration_s * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    click_len = int(0.010 * sr)  # 10 ms decaying-noise burst
    rng = np.random.default_rng(0)
    envelope = np.exp(-np.linspace(0.0, 6.0, click_len, dtype=np.float32))
    click = (rng.standard_normal(click_len).astype(np.float32) * envelope) * 0.8

    period_samples = int(sr * 60.0 / bpm)
    for i in range(0, n_samples, period_samples):
        end = min(i + click_len, n_samples)
        audio[i:end] += click[: end - i]

    # Clamp to int16 range and write a standard PCM WAV that madmom can decode.
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Slow test; set RUN_SLOW=1 to enable.",
)
def test_detect_madmom_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test on a 120 BPM click track."""
    wav_path = tmp_path / "click_120.wav"
    _write_click_track(wav_path, bpm=120.0, duration_s=5.0)

    wrapper = BeatTransformerWrapper()
    beats = wrapper.detect(wav_path)

    assert isinstance(beats, list)
    assert len(beats) > 0
    assert all(isinstance(b, Beat) for b in beats)

    times = [b.time for b in beats]
    assert times == sorted(times), "beats must be time-sorted"

    # 5 s @ 120 BPM = 10 beats; allow noise from the click-track synthesis.
    assert len(beats) >= 8, f"expected >=8 beats, got {len(beats)}"

    assert any(b.downbeat for b in beats), "expected at least one downbeat"
