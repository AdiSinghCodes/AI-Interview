"""Session + status API with SilenceTTS (no Kokoro load)."""

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.api import deps
from engine.api.main import app


class _FastTTS:
    sample_rate = 24_000

    def warmup(self):
        pass

    async def synthesize(self, text):
        yield np.zeros(2400, dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(deps, "get_tts", lambda: _FastTTS())
    return TestClient(app)


def test_create_session_then_prefetch_completes(client):
    r = client.post("/v1/sessions", json={"role": "python", "num_questions": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "python"
    assert body["num_questions"] == 6
    assert body["persona_name"]
    sid = body["session_id"]

    # prefetch runs as a background task on the app loop — poll status
    for _ in range(50):
        st = client.get(f"/v1/sessions/{sid}/status").json()
        if st["ready"]:
            break
    # poll until fully warmed (ready fires early, background finishes the rest)
    for _ in range(80):
        st = client.get(f"/v1/sessions/{sid}/status").json()
        if st["done"] == st["total"]:
            break
    assert st["ready"] is True
    assert st["done"] == st["total"] == 6 + 2 + 4


def test_unknown_role_is_accepted_and_falls_back(client):
    body = client.post("/v1/sessions", json={"role": "wizard"}).json()
    assert body["role"] == "technical"


def test_status_404_for_unknown_session(client):
    assert client.get("/v1/sessions/nope/status").status_code == 404
