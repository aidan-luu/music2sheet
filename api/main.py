"""FastAPI app for the music2sheet transcription service.

Endpoints:
- GET /healthz
- POST /transcribe
- GET /jobs/{job_id}

Inference is stubbed: jobs are stored in an in-memory dict and stay in
status="pending" indefinitely. Real workers land in PR-14.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status

from api.schemas.jobs import JobStatus, JobStatusResponse
from api.schemas.transcribe import TranscribeRequest, TranscribeResponse

app = FastAPI(
    title="music2sheet API",
    version="0.0.1",
    description=(
        "Audio-to-lead-sheet transcription service. Endpoints are stubbed "
        "until ML inference (Agent C) and the job queue (PR-14) land."
    ),
)

_JOBS: dict[UUID, JobStatusResponse] = {}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_transcription_job(request: TranscribeRequest) -> TranscribeResponse:
    job_id = uuid4()
    created_at = datetime.now(timezone.utc)
    _JOBS[job_id] = JobStatusResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        result_urls=None,
        error=None,
    )
    return TranscribeResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=created_at,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: UUID) -> JobStatusResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
