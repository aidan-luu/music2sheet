"""Audio I/O helpers.

PR-0 ships signatures only; bodies land alongside the first real pipeline
stage (PR-1, HT-Demucs wrapper) so we can choose between ``librosa`` and
``torchaudio`` after benchmarking decode latency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path_or_url: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load a local audio file or remote URL into a mono float32 waveform.

    Parameters
    ----------
    path_or_url:
        A local filesystem path OR a remote URL (http/https/youtube). YouTube
        URLs are routed through :func:`download_youtube`.
    target_sr:
        Resample target. 44100 matches the HT-Demucs v4 expected SR.

    Returns
    -------
    (waveform, sample_rate)
        ``waveform`` is a 1-D float32 array in [-1.0, 1.0]; ``sample_rate``
        equals ``target_sr`` after resampling.
    """
    raise NotImplementedError("Implemented in PR-1 (HT-Demucs wrapper).")


def download_youtube(url: str, dest_dir: Path) -> Path:
    """Download a YouTube video's audio track to ``dest_dir``.

    Uses ``yt-dlp`` to extract the best available audio stream and writes
    it as ``<video_id>.<ext>`` inside ``dest_dir``. Returns the final path.
    """
    raise NotImplementedError("Implemented in PR-1 (HT-Demucs wrapper).")


def audio_hash(audio_path: Path) -> str:
    """Return the lowercase hex SHA-256 of the file at ``audio_path``.

    Used by the model registry and the inference cache to key results on
    exact input bytes (a single bit-flip yields a fresh hash, intentionally).
    """
    raise NotImplementedError("Implemented in PR-1 (HT-Demucs wrapper).")
