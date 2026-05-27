"""Blob storage abstraction for job inputs and result artifacts.

Local filesystem implementation today; the same `JobBlobStore` interface
swaps cleanly to an S3 or Modal-volume backend in a future PR.

Layout under `$MUSIC2SHEET_BLOB_DIR` (default `/tmp/music2sheet-blobs`):

    <root>/<job_id>/input.<ext>
    <root>/<job_id>/result.musicxml
    <root>/<job_id>/result.mid
    <root>/<job_id>/result.pdf
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

ResultKind = Literal["musicxml", "midi", "pdf"]

_RESULT_FILENAME: dict[str, str] = {
    "musicxml": "result.musicxml",
    "midi": "result.mid",
    "pdf": "result.pdf",
}


def _default_root() -> Path:
    return Path(os.environ.get("MUSIC2SHEET_BLOB_DIR", "/tmp/music2sheet-blobs"))


class JobBlobStore:
    """Per-job blob store rooted at a single directory."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else _default_root()

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def put_audio(self, job_id: str, audio_bytes: bytes, filename: str) -> Path:
        """Persist raw audio bytes for `job_id`. Returns the written path.

        `filename` provides the extension; the on-disk name is normalized to
        `input.<ext>` so the worker can find it without consulting the DB.
        """
        ext = Path(filename).suffix.lstrip(".").lower() or "bin"
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / f"input.{ext}"
        path.write_bytes(audio_bytes)
        return path

    def put_audio_url(self, job_id: str, url: str) -> Path:
        """Persist a deferred-download URL marker for `job_id`.

        The real download happens in a future PR (yt-dlp + range GET); for now
        we just record the URL so the worker has a breadcrumb.
        """
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / "input.url"
        path.write_text(url, encoding="utf-8")
        return path

    def get_result_path(self, job_id: str, kind: ResultKind) -> Path:
        filename = _RESULT_FILENAME[kind]
        return self._job_dir(job_id) / filename

    def job_exists(self, job_id: str) -> bool:
        return self._job_dir(job_id).is_dir()
