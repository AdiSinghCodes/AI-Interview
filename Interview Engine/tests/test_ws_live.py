"""End-to-end WS path with fake VAD / STT / provider + SilenceTTS — audio
in, transcript + streamed spoken reply out. No models loaded."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import engine.transport.ws_live as ws_live
from engine.api import deps
from engine.api.main import app
from engine.inference.provider import GenerationResult, InferenceProvider
from engine.speech.tts_kokoro import SilenceTTS


class _FakeProvider(InferenceProvider):
    async def generate(self, spec, messages) -> GenerationResult:
        return GenerationResult(text='{"answer": "ok"}')

    async def stream(self, spec, messages):
        for part in ('{"answer": "', "That makes sense, ", "so how do you paginate it?", '"}'):
            yield part

    async def embed(self, spec, text):
        return [0.0] * 768

    async def cancel(self, request_id):
        return None


class _FakeVAD:
    def is_speech(self, frame):
        return bool(np.abs(frame).max() > 0.1)

    def reset(self):
        pass


class _FakeSTT:
    def warmup(self):
        pass

    def begin_utterance(self):
        pass

    def _text(self, audio):
        loud = audio.size and float(np.abs(audio).max()) > 0.1
        return "my answer about backend services" if loud else ""

    transcribe = _text
    feed_partial = _text
    finalize = _text


def _pcm(samples) -> bytes:
    return np.asarray(samples, dtype="<f4").tobytes()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ws_live, "_ECHO_TAIL_MS", 0.0)  # no wall-clock wait in tests
    monkeypatch.setattr(deps, "get_vad", lambda: _FakeVAD())
    monkeypatch.setattr(deps, "get_stt", lambda: _FakeSTT())
    monkeypatch.setattr(deps, "get_tts", lambda: SilenceTTS())
    monkeypatch.setattr(deps, "get_inference_provider", lambda: _FakeProvider())
    return TestClient(app)


def _drain_until(ws, wanted_type: str):
    """Collect frames (JSON + binary) until a JSON frame of `wanted_type`."""
    seen_json: list[dict] = []
    audio_frames = 0
    while True:
        msg = ws.receive()
        if "text" in msg and msg["text"] is not None:
            payload = json.loads(msg["text"])
            seen_json.append(payload)
            if payload.get("type") == wanted_type:
                return seen_json, audio_frames
        elif "bytes" in msg and msg["bytes"] is not None:
            audio_frames += 1


def test_audio_in_yields_transcript_then_streamed_spoken_reply(client):
    with client.websocket_connect("/v1/rounds/t1/live") as ws:
        ws.send_bytes(_pcm(np.zeros(512, dtype=np.float32)))  # wake the round
        assert ws.receive_json()["type"] == "opening"
        _drain_until(ws, "speaking_end")                      # opening line plays out

        for _ in range(6):
            ws.send_bytes(_pcm(np.ones(512, dtype=np.float32)))   # speech
        for _ in range(40):
            ws.send_bytes(_pcm(np.zeros(512, dtype=np.float32)))  # silence -> endpoint

        events, audio_frames = _drain_until(ws, "interviewer")
        types = [e["type"] for e in events]

        assert types[0] == "transcript" and "backend" in events[0]["text"]
        assert "speaking_start" in types and "speaking_end" in types
        assert types.index("speaking_start") < types.index("speaking_end")
        assert audio_frames >= 1                       # at least one clause of audio

        reply = events[-1]
        assert reply["type"] == "interviewer"
        assert reply["text"] == "That makes sense, so how do you paginate it?"
        assert reply["state"] == "listening"
        for stage in ("stt_final", "ttft", "first_tts_chunk", "playback_start"):
            assert stage in reply["trace"]

        ws.send_text(json.dumps({"type": "end_round"}))


def test_silence_only_never_produces_a_turn(client):
    with client.websocket_connect("/v1/rounds/t2/live") as ws:
        ws.send_bytes(_pcm(np.zeros(512, dtype=np.float32)))
        assert ws.receive_json()["type"] == "opening"
        for _ in range(40):
            ws.send_bytes(_pcm(np.zeros(512, dtype=np.float32)))
        ws.send_text(json.dumps({"type": "end_round"}))
