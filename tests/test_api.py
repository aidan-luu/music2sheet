"""Integration tests for the FastAPI service (Agent B owns scope; file lives
under tests/ per the documented exception in skills.md).
"""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest
from httpx import ASGITransport

from api.main import app


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_healthz(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_create_job_returns_uuid(client: httpx.AsyncClient) -> None:
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


async def test_create_then_get_job(client: httpx.AsyncClient) -> None:
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
    assert body["status"] == "pending"
    assert body["progress"] == 0.0


async def test_one_of_audio_required(client: httpx.AsyncClient) -> None:
    r = await client.post("/transcribe", json={})
    assert r.status_code == 422
