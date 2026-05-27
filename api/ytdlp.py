"""Thin wrapper around yt-dlp for the `audio_url` intake path.

The API accepts a YouTube (or direct-MP3/WAV) URL on transcribe submission;
when the worker picks the job up it calls :func:`download_audio` to actually
fetch the bytes and normalize them to a 44.1 kHz WAV that downstream stages
(Demucs, MERT, …) can ingest.

Errors that originate inside yt-dlp (`DownloadError` from private videos,
geo-blocks, network failures, missing format, etc.) are translated into
:class:`YtdlpError` so callers don't have to import yt-dlp internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# yt-dlp is intentionally imported lazily inside the function so that:
#   1. tests which monkey-patch `api.workers.download_audio` don't require
#      yt-dlp to be installed in the test environment;
#   2. import time of the FastAPI app isn't hit by yt-dlp's heavy module init.


# 30s matches the task spec: enough headroom for the initial metadata fetch
# on a slow link, short enough that a hung/timed-out probe doesn't pin a
# worker slot indefinitely.
_SOCKET_TIMEOUT_S = 30


class YtdlpError(Exception):
    """Raised when yt-dlp cannot produce a usable audio file.

    The wrapped message is safe to surface to API clients as the `error`
    field on a failed job (yt-dlp's messages are user-readable: "Private
    video", "Video unavailable", "HTTP Error 429: Too Many Requests", …).
    """


def download_audio(url: str, dest_dir: Path) -> Path:
    """Download best-audio from ``url`` and convert to 44.1 kHz WAV.

    Parameters
    ----------
    url:
        HTTP(S) URL accepted by yt-dlp (YouTube watch URL, youtu.be short
        link, or direct audio file URL).
    dest_dir:
        Directory the resulting WAV is written into. Created if missing.
        The file is named ``input.wav`` to match the convention used by
        :class:`api.storage.JobBlobStore` for locally-uploaded audio.

    Returns
    -------
    Path
        Absolute path to the produced WAV file.

    Raises
    ------
    YtdlpError
        If yt-dlp's extractor or downloader fails for any reason, or if
        the postprocessor finishes but the expected WAV is missing on disk.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "input.wav"

    # yt-dlp's outtmpl receives the *pre-postprocessor* extension, so we
    # template the stem and let FFmpegExtractAudio swap the suffix to .wav.
    outtmpl = str(dest_dir / "input.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": _SOCKET_TIMEOUT_S,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # 44.1 kHz mono-or-stereo WAV is what Demucs/MERT expect. We let
        # FFmpegExtractAudio default to the source sample rate; downstream
        # resampling is handled by Agent C's preprocessor. Keeping this
        # wrapper agnostic of model sample rates avoids a second decode if
        # the source happens to already be 44.1 kHz.
    }

    try:
        # Imported lazily — see module docstring rationale.
        import yt_dlp  # noqa: PLC0415
        from yt_dlp.utils import DownloadError  # noqa: PLC0415

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except DownloadError as exc:
        raise YtdlpError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # Unexpected (e.g. ffmpeg missing, permissions): surface to the
        # caller as a YtdlpError too so the job-failure path stays uniform.
        raise YtdlpError(f"yt-dlp failed: {exc}") from exc

    if not dest_path.exists():
        # Postprocessor reported success but file isn't where we expect —
        # treat as a download failure rather than crashing the worker.
        raise YtdlpError(
            "yt-dlp completed but no WAV was produced "
            "(ffmpeg postprocessor may have failed silently)"
        )

    return dest_path
