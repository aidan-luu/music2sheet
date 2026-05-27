"""Background worker stub for transcription jobs.

Real inference (HT-Demucs + MERT + decoder, owned by Agent C) lands in a
later PR. For PR-B1 the worker simulates progress and writes a fixture
MusicXML so the frontend can drive an end-to-end demo.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from api.schemas.jobs import JobResultUrls, JobStatus, JobStatusResponse
from api.storage import JobBlobStore

FIXTURE_MUSICXML = Path(__file__).parent / "fixtures" / "sample.musicxml"


def _stage_delay_seconds() -> float:
    """Per-stage simulated wait. Overridden in tests via env var."""
    raw = os.environ.get("MUSIC2SHEET_WORKER_DELAY_S", "2")
    try:
        return float(raw)
    except ValueError:
        return 2.0


def _result_urls(job_id: str) -> JobResultUrls:
    return JobResultUrls(
        musicxml=f"/jobs/{job_id}/results/musicxml",
        midi=f"/jobs/{job_id}/results/midi",
        pdf=f"/jobs/{job_id}/results/pdf",
    )


async def process_job(
    job_id: str,
    blob_store: JobBlobStore,
    jobs: dict[str, JobStatusResponse],
) -> None:
    """Advance `job_id` from pending → running → done.

    Stages mirror the eventual real pipeline so progress increments feel
    natural in the UI: load (0.1) → forward (0.5) → assemble (0.9) → done.
    """
    delay = _stage_delay_seconds()

    job = jobs.get(job_id)
    if job is None:
        return

    job.status = JobStatus.RUNNING
    job.progress = 0.1

    await asyncio.sleep(delay)
    job.progress = 0.5

    await asyncio.sleep(delay)
    job.progress = 0.9

    result_path = blob_store.get_result_path(job_id, "musicxml")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_MUSICXML, result_path)

    job.status = JobStatus.DONE
    job.progress = 1.0
    job.result_urls = _result_urls(job_id)
