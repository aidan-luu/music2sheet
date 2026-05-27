"""FastAPI app for the music2sheet transcription service.

Endpoints:
- GET /healthz
- POST /transcribe
- GET /jobs/{job_id}
- GET /jobs/{job_id}/results/{kind}

Jobs are tracked in an in-memory dict and advanced by `api.workers.process_job`
scheduled via :func:`asyncio.create_task`. Real model inference lands in a
later PR. FastAPI ``BackgroundTasks`` is intentionally avoided because httpx's
ASGITransport awaits BackgroundTasks before returning the response, which would
prevent tests from observing intermediate ``running`` state via polling.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from api.schemas.jobs import JobStatus, JobStatusResponse
from api.schemas.transcribe import TranscribeRequest, TranscribeResponse
from api.storage import JobBlobStore
from api.workers import process_job

app = FastAPI(
    title="music2sheet API",
    version="0.1.0",
    description=(
        "Audio-to-lead-sheet transcription service. Inference is a stub "
        "(fixture MusicXML) until Agent C's models land."
    ),
)

_JOBS: dict[str, JobStatusResponse] = {}
_BLOB_STORE = JobBlobStore()
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()

_RESULT_MEDIA_TYPE: dict[str, str] = {
    "musicxml": "application/vnd.recordare.musicxml+xml",
    "midi": "audio/midi",
    "pdf": "application/pdf",
}


def _detect_audio_extension(data: bytes) -> str | None:
    """Return 'wav' or 'mp3' based on magic bytes, or None if unknown."""
    if len(data) < 4:
        return None
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3":
        return "mp3"
    if data[0] == 0xFF and data[1] in (0xFB, 0xF3, 0xF2):
        return "mp3"
    return None


def _validate_audio_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="audio_url is not a well-formed URL")
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    is_youtube = host.endswith("youtube.com") or host.endswith("youtu.be")
    is_direct_audio = path.endswith(".mp3") or path.endswith(".wav")
    if not (is_youtube or is_direct_audio):
        raise HTTPException(
            status_code=400,
            detail="audio_url must be a YouTube link or end in .mp3/.wav",
        )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_transcription_job(
    request: TranscribeRequest,
) -> TranscribeResponse:
    job_id = uuid4()
    job_id_str = str(job_id)

    if request.audio_file_b64 is not None:
        try:
            audio_bytes = base64.b64decode(request.audio_file_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="audio_file_b64 is not valid base64"
            ) from exc
        ext = _detect_audio_extension(audio_bytes)
        if ext is None:
            raise HTTPException(
                status_code=400,
                detail="audio_file_b64 magic bytes did not match MP3 or WAV",
            )
        _BLOB_STORE.put_audio(job_id_str, audio_bytes, f"input.{ext}")
    else:
        assert request.audio_url is not None  # enforced by schema validator
        _validate_audio_url(request.audio_url)
        _BLOB_STORE.put_audio_url(job_id_str, request.audio_url)

    created_at = datetime.now(timezone.utc)
    _JOBS[job_id_str] = JobStatusResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        result_urls=None,
        error=None,
    )
    task = asyncio.create_task(process_job(job_id_str, _BLOB_STORE, _JOBS))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return TranscribeResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=created_at,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: UUID) -> JobStatusResponse:
    job = _JOBS.get(str(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/results/{kind}")
def get_job_result(
    job_id: UUID, kind: Literal["musicxml", "midi", "pdf"]
) -> FileResponse:
    job_id_str = str(job_id)
    if job_id_str not in _JOBS:
        raise HTTPException(status_code=404, detail="job not found")
    path = _BLOB_STORE.get_result_path(job_id_str, kind)
    if not path.exists():
        raise HTTPException(status_code=404, detail="result not ready")
    return FileResponse(
        path,
        media_type=_RESULT_MEDIA_TYPE[kind],
        filename=path.name,
    )
