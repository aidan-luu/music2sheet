"""FastAPI app for the music2sheet transcription service.

Endpoints:
- GET /healthz
- POST /transcribe                (JSON; deprecated — kept for back-compat)
- POST /transcribe/upload         (multipart/form-data; preferred for files)
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
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


def _persist_audio_bytes(job_id: str, audio_bytes: bytes, *, source_label: str) -> None:
    """Validate audio magic bytes and persist to the blob store.

    `source_label` is woven into the 400 detail so the client gets a useful
    hint about which intake path failed.
    """
    ext = _detect_audio_extension(audio_bytes)
    if ext is None:
        raise HTTPException(
            status_code=400,
            detail=f"{source_label} magic bytes did not match MP3 or WAV",
        )
    _BLOB_STORE.put_audio(job_id, audio_bytes, f"input.{ext}")


def _intake_and_schedule(
    *,
    audio_bytes: bytes | None,
    audio_url: str | None,
    audio_source_label: str,
) -> TranscribeResponse:
    """Shared body for both intake endpoints.

    Exactly one of `audio_bytes` / `audio_url` must be non-None — the caller
    enforces the XOR with the validation rules appropriate to its content type
    (Pydantic model_validator for JSON, manual 422 for multipart).

    Persists the audio, records the job, schedules the background worker, and
    returns the response shape Agent A is coding against.
    """
    job_id = uuid4()
    job_id_str = str(job_id)

    if audio_bytes is not None:
        _persist_audio_bytes(job_id_str, audio_bytes, source_label=audio_source_label)
    else:
        assert audio_url is not None  # enforced by callers
        _validate_audio_url(audio_url)
        _BLOB_STORE.put_audio_url(job_id_str, audio_url)

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


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_201_CREATED,
    deprecated=True,
    description=(
        "Deprecated: prefer POST /transcribe/upload (multipart/form-data) for "
        "file uploads. This JSON endpoint requires base64-encoding the audio "
        "payload, which is bandwidth-wasteful and forces the browser through a "
        "chunked btoa workaround for files >~5MB. Kept indefinitely for "
        "back-compat with existing clients."
    ),
)
async def create_transcription_job(
    request: TranscribeRequest,
) -> TranscribeResponse:
    audio_bytes: bytes | None = None
    if request.audio_file_b64 is not None:
        try:
            audio_bytes = base64.b64decode(request.audio_file_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="audio_file_b64 is not valid base64"
            ) from exc

    return _intake_and_schedule(
        audio_bytes=audio_bytes,
        audio_url=request.audio_url,
        audio_source_label="audio_file_b64",
    )


@app.post(
    "/transcribe/upload",
    response_model=TranscribeResponse,
    status_code=status.HTTP_201_CREATED,
    description=(
        "Create a transcription job from a multipart/form-data upload. "
        "Provide exactly one of `audio_file` (binary MP3/WAV) or `audio_url` "
        "(remote fetch). Returns the same TranscribeResponse shape as the "
        "JSON endpoint."
    ),
)
async def create_transcription_job_upload(
    audio_file: UploadFile | None = File(
        default=None,
        description="Binary audio payload (MP3 or WAV). Mutually exclusive with audio_url.",
    ),
    audio_url: str | None = Form(
        default=None,
        description="HTTP(S) URL the backend will fetch the audio from.",
    ),
) -> TranscribeResponse:
    # FastAPI/Starlette gives us an UploadFile sentinel even when the client
    # didn't include the field; treat empty filename as "not provided".
    has_file = audio_file is not None and bool(audio_file.filename)
    has_url = audio_url is not None and audio_url != ""

    if has_file == has_url:
        raise HTTPException(
            status_code=422,
            detail="Exactly one of 'audio_file' or 'audio_url' must be provided.",
        )

    audio_bytes: bytes | None = None
    if has_file:
        assert audio_file is not None
        audio_bytes = await audio_file.read()

    return _intake_and_schedule(
        audio_bytes=audio_bytes,
        audio_url=audio_url if has_url else None,
        audio_source_label="audio_file",
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
