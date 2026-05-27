"""Pydantic schemas for job status."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobResultUrls(BaseModel):
    """Artifact URLs produced by a successful transcription job."""

    model_config = ConfigDict(extra="forbid")

    musicxml: str
    midi: str
    pdf: str


class JobStatusResponse(BaseModel):
    """Response body for GET /jobs/{job_id}."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    result_urls: JobResultUrls | None = None
    error: str | None = None
