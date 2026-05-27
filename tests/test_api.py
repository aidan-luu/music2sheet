"""Integration tests for the FastAPI service (Agent B owns scope; file lives
under tests/ per the documented exception in skills.md).
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from httpx import ASGITransport

from api import main as api_main
from api.main import app
from api.storage import JobBlobStore


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def isolated_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point the app's blob store at a tmp dir and shorten worker sleeps.

    Resets in-memory job state between tests so polling assertions stay
    deterministic.
    """
    monkeypatch.setenv("MUSIC2SHEET_BLOB_DIR", str(tmp_path))
    monkeypatch.setenv("MUSIC2SHEET_WORKER_DELAY_S", "0.05")
    api_main._BLOB_STORE = JobBlobStore(tmp_path)
    api_main._JOBS.clear()
    return tmp_path


_WAV_HEADER_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="


async def test_healthz(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_create_job_returns_uuid(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    r = await client.post(
        "/transcribe",
        json={"audio_url": "https://example.com/song.mp3"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "pending"
    # Raises ValueError if not a valid UUID.
    UUID(body["job_id"])
    assert "created_at" in body


async def test_get_unknown_job_404(client: httpx.AsyncClient) -> None:
    r = await client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


async def test_create_then_get_job(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    create = await client.post(
        "/transcribe",
        json={"audio_url": "https://example.com/song.mp3"},
    )
    assert create.status_code == 201
    job_id = create.json()["job_id"]

    got = await client.get(f"/jobs/{job_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["job_id"] == job_id
    # Worker is now active (PR-B1); status will be pending/running/done depending
    # on how much of the event loop has run between POST and GET. The contract
    # the existing test was checking is that the job exists and has the right
    # shape — not that progress is frozen at 0.
    assert body["status"] in ("pending", "running", "done")
    assert 0.0 <= body["progress"] <= 1.0


async def test_one_of_audio_required(client: httpx.AsyncClient) -> None:
    r = await client.post("/transcribe", json={})
    assert r.status_code == 422


async def test_upload_b64_wav_round_trip(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    r = await client.post(
        "/transcribe",
        json={"audio_file_b64": _WAV_HEADER_B64},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    job_dir = isolated_blobs / job_id
    assert job_dir.is_dir()
    inputs = list(job_dir.glob("input.wav"))
    assert len(inputs) == 1
    written = inputs[0].read_bytes()
    assert written == base64.b64decode(_WAV_HEADER_B64)


async def test_invalid_audio_format_400(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    junk = base64.b64encode(b"not-real-audio-bytes-just-noise").decode("ascii")
    r = await client.post("/transcribe", json={"audio_file_b64": junk})
    assert r.status_code == 400, r.text


async def test_job_progresses_to_done(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    create = await client.post(
        "/transcribe", json={"audio_file_b64": _WAV_HEADER_B64}
    )
    assert create.status_code == 201
    job_id = create.json()["job_id"]

    saw_running = False
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        got = await client.get(f"/jobs/{job_id}")
        assert got.status_code == 200
        body = got.json()
        if body["status"] == "running":
            saw_running = True
        if body["status"] == "done":
            assert body["progress"] == 1.0
            assert body["result_urls"]["musicxml"] == f"/jobs/{job_id}/results/musicxml"
            assert saw_running, "expected to observe running state before done"
            return
        await asyncio.sleep(0.02)
    pytest.fail("job did not reach done within timeout")


async def test_result_musicxml_served(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    create = await client.post(
        "/transcribe", json={"audio_file_b64": _WAV_HEADER_B64}
    )
    job_id = create.json()["job_id"]

    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        got = await client.get(f"/jobs/{job_id}")
        if got.json()["status"] == "done":
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not reach done within timeout")

    r = await client.get(f"/jobs/{job_id}/results/musicxml")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/vnd.recordare.musicxml+xml")
    body = r.text
    assert body.lstrip().startswith("<?xml")
    assert "<score-partwise" in body


async def test_result_404_before_done(
    client: httpx.AsyncClient, isolated_blobs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Slow the worker down so the result is reliably absent at request time.
    monkeypatch.setenv("MUSIC2SHEET_WORKER_DELAY_S", "5")
    create = await client.post(
        "/transcribe", json={"audio_file_b64": _WAV_HEADER_B64}
    )
    job_id = create.json()["job_id"]
    r = await client.get(f"/jobs/{job_id}/results/musicxml")
    assert r.status_code == 404


async def test_result_unknown_kind(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    create = await client.post(
        "/transcribe", json={"audio_file_b64": _WAV_HEADER_B64}
    )
    job_id = create.json()["job_id"]
    r = await client.get(f"/jobs/{job_id}/results/garbage")
    assert r.status_code == 422


# --------------------------------------------------------------------------
# PR-B2: multipart upload endpoint (POST /transcribe/upload)
# --------------------------------------------------------------------------

# Minimal MP3 frame: ID3v2 header (10 bytes, no payload) followed by a single
# MPEG-1 Layer 3 frame sync (0xFF 0xFB ...). The magic-bytes sniffer keys off
# the leading "ID3" tag, so the payload length doesn't matter for validation.
_MP3_BYTES = (
    b"ID3\x04\x00\x00\x00\x00\x00\x00"  # ID3v2.4 header, zero-size tag
    b"\xff\xfb\x90\x00" + b"\x00" * 16  # one frame header + padding
)
_WAV_BYTES = base64.b64decode(_WAV_HEADER_B64)


async def test_upload_multipart_wav_ok(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    r = await client.post(
        "/transcribe/upload",
        files={"audio_file": ("song.wav", _WAV_BYTES, "audio/wav")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    UUID(body["job_id"])
    assert body["status"] == "pending"
    job_dir = isolated_blobs / body["job_id"]
    assert job_dir.is_dir()
    written = (job_dir / "input.wav").read_bytes()
    assert written == _WAV_BYTES


async def test_upload_multipart_mp3_ok(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    r = await client.post(
        "/transcribe/upload",
        files={"audio_file": ("song.mp3", _MP3_BYTES, "audio/mpeg")},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    written = (isolated_blobs / job_id / "input.mp3").read_bytes()
    assert written == _MP3_BYTES


async def test_upload_multipart_bad_audio_400(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    r = await client.post(
        "/transcribe/upload",
        files={"audio_file": ("noise.bin", b"not-real-audio-bytes-just-noise", "audio/wav")},
    )
    assert r.status_code == 400, r.text


async def test_upload_multipart_missing_input_422(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    # No audio_file, no audio_url — multipart with empty body.
    r = await client.post(
        "/transcribe/upload",
        files={},
        data={},
    )
    assert r.status_code == 422, r.text


async def test_upload_multipart_both_inputs_422(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    r = await client.post(
        "/transcribe/upload",
        files={"audio_file": ("song.wav", _WAV_BYTES, "audio/wav")},
        data={"audio_url": "https://example.com/other.mp3"},
    )
    assert r.status_code == 422, r.text


async def test_legacy_json_endpoint_still_works(
    client: httpx.AsyncClient, isolated_blobs: Path
) -> None:
    """Back-compat: PR-B2 deprecates the JSON endpoint but must not remove it."""
    r = await client.post(
        "/transcribe",
        json={"audio_file_b64": _WAV_HEADER_B64},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    assert (isolated_blobs / job_id / "input.wav").read_bytes() == _WAV_BYTES


# --------------------------------------------------------------------------
# PR-B3: real yt-dlp download on the audio_url intake path.
#
# These tests must NEVER hit YouTube — we monkey-patch the import the worker
# actually calls (`api.workers.download_audio`). Hitting the network would be
# flaky in CI, slow in dev, and would change the meaning of "test failure"
# from "regression" to "internet was bad today".
# --------------------------------------------------------------------------

from api import workers as api_workers  # noqa: E402
from api.ytdlp import YtdlpError  # noqa: E402


async def _poll_until_status(
    client: httpx.AsyncClient, job_id: str, *, timeout_s: float = 5.0
) -> dict:
    """Poll GET /jobs/{job_id} until status is terminal (done/failed)."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        got = await client.get(f"/jobs/{job_id}")
        assert got.status_code == 200
        body = got.json()
        if body["status"] in ("done", "failed"):
            return body
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} did not reach terminal status within {timeout_s}s")


async def test_yt_dlp_failure_marks_job_failed(
    client: httpx.AsyncClient,
    isolated_blobs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(url: str, dest_dir: Path) -> Path:
        raise YtdlpError("private video")

    monkeypatch.setattr(api_workers, "download_audio", _boom)

    r = await client.post(
        "/transcribe",
        json={"audio_url": "https://youtube.com/watch?v=privateXYZ"},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]

    body = await _poll_until_status(client, job_id)
    assert body["status"] == "failed"
    assert body["error"] is not None
    assert "private video" in body["error"]
    # No result was produced — the results endpoint should 404.
    rr = await client.get(f"/jobs/{job_id}/results/musicxml")
    assert rr.status_code == 404


async def test_yt_dlp_success_persists_audio(
    client: httpx.AsyncClient,
    isolated_blobs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_download(url: str, dest_dir: Path) -> Path:
        # Mirror the real wrapper's contract: write a 44.1kHz-ish tiny WAV
        # to <dest_dir>/input.wav and return the path. Bytes match the same
        # header used elsewhere in this file so the worker sees a real WAV.
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / "input.wav"
        out.write_bytes(_WAV_BYTES)
        return out

    monkeypatch.setattr(api_workers, "download_audio", _fake_download)

    r = await client.post(
        "/transcribe",
        json={"audio_url": "https://youtube.com/watch?v=okayVID"},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]

    body = await _poll_until_status(client, job_id)
    assert body["status"] == "done", body
    assert body["result_urls"]["musicxml"] == f"/jobs/{job_id}/results/musicxml"

    # The fake download wrote input.wav into the job dir.
    assert (isolated_blobs / job_id / "input.wav").read_bytes() == _WAV_BYTES
    # URL marker should have been cleaned up post-download.
    assert not (isolated_blobs / job_id / "input.url").exists()
    # And the fixture MusicXML is served as the result.
    rr = await client.get(f"/jobs/{job_id}/results/musicxml")
    assert rr.status_code == 200
    assert rr.text.lstrip().startswith("<?xml")
    assert "<score-partwise" in rr.text


async def test_yt_dlp_not_called_for_b64_path(
    client: httpx.AsyncClient,
    isolated_blobs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """b64/multipart intake must NOT invoke yt-dlp — no URL marker is present,
    so the download branch should be skipped entirely."""

    calls: list[tuple[str, Path]] = []

    def _spy(url: str, dest_dir: Path) -> Path:
        calls.append((url, dest_dir))
        raise AssertionError(
            "download_audio must not be called for b64/multipart intake"
        )

    monkeypatch.setattr(api_workers, "download_audio", _spy)

    r = await client.post(
        "/transcribe",
        json={"audio_file_b64": _WAV_HEADER_B64},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]

    body = await _poll_until_status(client, job_id)
    assert body["status"] == "done", body
    assert calls == []
