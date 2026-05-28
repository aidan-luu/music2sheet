"""Background worker stub for transcription jobs.

Real inference (HT-Demucs + MERT + decoder, owned by Agent C) lands in a
later PR. For PR-B1 the worker simulates progress and writes a fixture
MusicXML so the frontend can drive an end-to-end demo.

PR-B3 adds a real yt-dlp download step in front of the simulated pipeline:
when the intake stored a deferred-download URL marker (``input.url``), the
worker calls :func:`api.ytdlp.download_audio` to fetch and convert the
audio. Failures are translated into a ``failed`` job status with the
error message populated.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from api.schemas.jobs import JobResultUrls, JobStatus, JobStatusResponse
from api.storage import JobBlobStore
from api.ytdlp import YtdlpError, download_audio

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


def _mark_failed(job: JobStatusResponse, error: str) -> None:
    job.status = JobStatus.FAILED
    job.error = error
    # progress intentionally left where it was — surfaces *how far* the
    # pipeline got before bailing, useful for triage.


async def _ensure_audio_downloaded(
    job_dir: Path,
) -> None:
    """If a URL marker is present, download via yt-dlp and remove the marker.

    No-op if there's no ``input.url`` (i.e. the audio was uploaded directly).
    Runs the blocking yt-dlp call in a worker thread so the event loop keeps
    serving status polls during the download.
    """
    url_marker = job_dir / "input.url"
    if not url_marker.exists():
        return

    url = url_marker.read_text(encoding="utf-8").strip()
    # `download_audio` is imported at module scope so tests can monkey-patch
    # `api.workers.download_audio` directly.
    await asyncio.to_thread(download_audio, url, job_dir)

    # Clean up the marker so re-runs (and the result-served endpoint) don't
    # confuse the URL stub with a real input.
    url_marker.unlink(missing_ok=True)


async def process_job(
    job_id: str,
    blob_store: JobBlobStore,
    jobs: dict[str, JobStatusResponse],
) -> None:
    """Advance `job_id` from pending → running → done.

    Stages mirror the eventual real pipeline so progress increments feel
    natural in the UI: download (if URL) → load (0.1) → forward (0.5) →
    assemble (0.9) → done. yt-dlp failures short-circuit to ``failed``.
    """
    delay = _stage_delay_seconds()

    job = jobs.get(job_id)
    if job is None:
        return

    job.status = JobStatus.RUNNING
    job.progress = 0.1

    job_dir = blob_store.get_job_dir(job_id)
    try:
        await _ensure_audio_downloaded(job_dir)
    except YtdlpError as exc:
        _mark_failed(job, str(exc))
        return

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
