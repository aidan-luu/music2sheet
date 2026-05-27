"""Audio I/O helpers.

Implementations land here in PR-1 alongside the HT-Demucs wrapper. The
``load_audio`` helper is intentionally permissive: it accepts local paths
or URLs and falls back from ``soundfile`` (fast C-backed WAV/FLAC) to
``librosa`` (handles MP3/M4A and odd containers, slower).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Public hash length. Truncating sha256 to 16 hex chars is plenty of entropy
# for a per-file cache key (~64 bits) and keeps cache dir names readable.
_HASH_LEN = 16

# Buffer size for streaming file hashing. 1 MiB strikes a balance between
# syscall overhead and memory footprint for large audio (a 5-minute WAV at
# 44.1 kHz stereo is ~50 MiB).
_HASH_CHUNK = 1024 * 1024

_YOUTUBE_RE = re.compile(
    r"^(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/|youtu\.be/)",
    re.IGNORECASE,
)


def _is_youtube_url(s: str) -> bool:
    return bool(_YOUTUBE_RE.match(s))


def _is_http_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def load_audio(
    path_or_url: str | Path,
    target_sr: int = 44100,
) -> tuple[np.ndarray, int]:
    """Load a local audio file or remote URL into a float32 waveform.

    Parameters
    ----------
    path_or_url:
        A local filesystem path OR a remote URL (http/https/youtube). YouTube
        URLs are routed through :func:`download_youtube`; the resulting local
        file is then decoded.
    target_sr:
        Resample target. 44100 matches the HT-Demucs v4 expected SR.

    Returns
    -------
    (waveform, sample_rate)
        ``waveform`` is a float32 array shaped either ``(samples,)`` (mono
        source) or ``(channels, samples)`` (multi-channel source).
        ``sample_rate`` equals ``target_sr`` after resampling.

    Notes
    -----
    The function preserves the source channel count rather than forcing mono
    because HT-Demucs operates on stereo input. Callers that need mono can
    average across axis 0.
    """
    src = str(path_or_url)

    if _is_youtube_url(src):
        # Route YouTube URLs through yt-dlp; download into a temp area chosen
        # by the caller via a wrapper, but for a bare load_audio call we use
        # a sibling 'downloads' dir alongside the cwd. The function is mostly
        # used in tests with local files, so this branch is best-effort.
        dest = Path.cwd() / "downloads"
        dest.mkdir(parents=True, exist_ok=True)
        local = download_youtube(src, dest)
        return _decode_file(local, target_sr=target_sr)

    if _is_http_url(src):
        # Generic http(s) — defer to librosa which knows how to stream via
        # audioread. Treat as a file path-like by handing it to soundfile
        # first; if that fails, librosa will retry with its own backend.
        return _decode_file(src, target_sr=target_sr)

    return _decode_file(Path(src), target_sr=target_sr)


def _decode_file(
    path: Path | str,
    target_sr: int,
) -> tuple[np.ndarray, int]:
    """Decode an audio file. Tries soundfile first, falls back to librosa."""
    # Imports are local so module import stays cheap and CI environments
    # without librosa/soundfile installed can still import ml.audio_io
    # (e.g. for audio_hash() in unit tests).
    try:
        import soundfile as sf

        # soundfile returns (samples,) for mono, (samples, channels) for multi.
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            # Convert to (channels, samples) — the convention demucs uses.
            data = data.T
    except Exception:
        # Fall back to librosa for codecs soundfile can't handle (mp3, m4a, ...).
        import librosa

        data, sr = librosa.load(str(path), sr=None, mono=False)
        # librosa returns (samples,) for mono, (channels, samples) for multi
        # when mono=False, which already matches our convention.
        data = np.asarray(data, dtype=np.float32)

    if sr != target_sr:
        import librosa

        if data.ndim == 1:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        else:
            # Resample each channel; librosa.resample handles 2-D arrays in
            # newer versions, but loop for safety across versions.
            resampled = [
                librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in data
            ]
            data = np.stack(resampled, axis=0)
        sr = target_sr

    return np.ascontiguousarray(data, dtype=np.float32), sr


def download_youtube(url: str, dest_dir: Path) -> Path:
    """Download a YouTube video's audio track to ``dest_dir`` via ``yt-dlp``.

    Uses ``yt-dlp`` (invoked as a subprocess) to extract the best available
    audio stream. The output filename is templated as ``<video_id>.<ext>``
    so repeated calls overwrite a single artifact rather than accumulating
    timestamped duplicates.

    Returns the path to the downloaded audio file.

    Raises
    ------
    RuntimeError
        If ``yt-dlp`` is not on PATH, or the subprocess exits non-zero.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("yt-dlp") is None:
        raise RuntimeError(
            "yt-dlp is not installed or not on PATH. Install it via "
            "`pip install yt-dlp` to enable YouTube ingest."
        )

    out_template = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--print",
        "after_move:filepath",
        "-o",
        out_template,
        url,
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("yt-dlp is not installed or not on PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"yt-dlp failed (exit {e.returncode}): {e.stderr.strip()}"
        ) from e

    # yt-dlp's `--print after_move:filepath` prints the final on-disk path
    # for the post-processed audio file (one per video). If multiple lines
    # are produced (shouldn't happen with --no-playlist), take the last.
    printed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if printed:
        candidate = Path(printed[-1])
        if candidate.exists():
            return candidate

    # Fallback: scan dest_dir for the newest audio file matching our pattern.
    audio_exts = {".mp3", ".m4a", ".webm", ".opus", ".wav", ".flac"}
    candidates = [
        p for p in dest_dir.iterdir() if p.is_file() and p.suffix.lower() in audio_exts
    ]
    if not candidates:
        raise RuntimeError(
            f"yt-dlp completed but no audio file was found in {dest_dir}."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def audio_hash(audio_path: str | Path) -> str:
    """Return a stable 16-character hex hash of the file's bytes.

    Uses SHA-256 internally and truncates the hex digest. This is the cache
    key for :meth:`ml.models.demucs.DemucsWrapper.separate_cached` and the
    model registry's per-input result lookup. A single bit-flip in the input
    yields a fresh hash; the hash does not depend on file mtime or path.
    """
    p = Path(audio_path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()[:_HASH_LEN]
