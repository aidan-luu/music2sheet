"""Pydantic schemas for the POST /transcribe endpoint."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.schemas.jobs import JobStatus


class TranscribeRequest(BaseModel):
    """Request body for POST /transcribe.

    Exactly one of `audio_url` or `audio_file_b64` must be provided.
    """

    model_config = ConfigDict(extra="forbid")

    audio_url: str | None = Field(
        default=None,
        description="HTTP(S) URL the backend will fetch the audio from.",
    )
    audio_file_b64: str | None = Field(
        default=None,
        description="Base64-encoded audio payload (mp3/wav).",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> TranscribeRequest:
        provided = [v is not None for v in (self.audio_url, self.audio_file_b64)]
        if sum(provided) != 1:
            raise ValueError(
                "Exactly one of 'audio_url' or 'audio_file_b64' must be provided."
            )
        return self


class TranscribeResponse(BaseModel):
    """Response body for POST /transcribe."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    status: JobStatus = JobStatus.PENDING
    created_at: datetime
