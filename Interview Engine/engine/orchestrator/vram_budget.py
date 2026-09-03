"""
VRAM budget manager (plan p5, module #10). Enforce at boot and between
phases: exactly ONE Ollama model resident during a live session.

  - planner (qwen3:8b, ~5.2 GB): explicitly unloaded before the call starts
  - interviewer (qwen3:4b): keep_alive -1, stays resident the whole session
  - STT / TTS / VAD: CPU by design, no card contention

STUB — build-order pass 3/7. Interface fixed now.
"""

from __future__ import annotations

import httpx

from engine.settings import get_settings


async def unload(model_name: str) -> None:
    """Ask Ollama to evict a model now (keep_alive: 0)."""
    host = get_settings().ollama_host.rstrip("/")
    async with httpx.AsyncClient(base_url=host, timeout=30.0) as client:
        await client.post(
            "/api/generate", json={"model": model_name, "keep_alive": 0, "prompt": ""}
        )


async def resident_models() -> list[str]:
    host = get_settings().ollama_host.rstrip("/")
    async with httpx.AsyncClient(base_url=host, timeout=10.0) as client:
        resp = await client.get("/api/ps")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]


async def assert_single_resident() -> None:
    """Raise if more than one model is on the card during a live session."""
    raise NotImplementedError("VRAM enforcement lands in build-order pass 3")
