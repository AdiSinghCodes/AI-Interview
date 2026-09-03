"""
Ollama backend. Local models: interviewer + fallback chain, planner,
embedder, and the local fallback for parser/scorer.

Contract this file must keep (Phase 0.2 test 11):
  - send `think` ONLY when the spec sets it. phi4-mini / llama3.2:3b have
    no thinking mode and 400 on an unexpected `think` key — which once
    made the entire interviewer fallback chain dead.
  - a JSON schema in the spec becomes Ollama's `format` object, which
    structurally blocks qwen3:4b's prose-leak (a monologue can't validate
    against {"answer": "..."}).
  - `keep_alive: -1` keeps the model resident — a cold reload is 15-20s.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from engine.settings import LLMSpec, get_settings
from .provider import ChatMessage, GenerationResult, InferenceError, InferenceProvider


class OllamaProvider(InferenceProvider):
    def __init__(self, host: str | None = None):
        self._host = (host or get_settings().ollama_host).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._host, timeout=120.0)

    # -- payload -----------------------------------------------------------
    def _chat_payload(
        self, spec: LLMSpec, messages: list[ChatMessage], *, stream: bool
    ) -> dict:
        payload: dict = {
            "model": spec.name,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
        }
        if spec.think is not None:            # never send it otherwise
            payload["think"] = spec.think
        if spec.keep_alive is not None:
            payload["keep_alive"] = spec.keep_alive

        if spec.json_schema_def is not None:
            payload["format"] = spec.json_schema_def
        elif spec.format == "json":
            payload["format"] = "json"

        options = {
            "num_predict": spec.max_tokens,
            "num_ctx": spec.num_ctx,
            "temperature": spec.temperature,
        }
        options = {k: v for k, v in options.items() if v is not None}
        if options:
            payload["options"] = options
        return payload

    # -- calls -----------------------------------------------------------
    async def generate(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> GenerationResult:
        payload = self._chat_payload(spec, messages, stream=False)
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise InferenceError(f"ollama generate {spec.name!r}: {e}") from e
        data = resp.json()
        elapsed = (time.perf_counter() - t0) * 1000.0
        return GenerationResult(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_ms=elapsed,
            model_used=spec.name,
        )

    async def stream(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        payload = self._chat_payload(spec, messages, stream=True)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        return
        except httpx.HTTPError as e:
            raise InferenceError(f"ollama stream {spec.name!r}: {e}") from e

    async def embed(self, spec: LLMSpec, text: str) -> list[float]:
        try:
            resp = await self._client.post(
                "/api/embeddings", json={"model": spec.name, "prompt": text}
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise InferenceError(f"ollama embed {spec.name!r}: {e}") from e
        vector = resp.json().get("embedding", [])
        if spec.dim is not None and len(vector) != spec.dim:
            raise InferenceError(
                f"embedder {spec.name!r} returned dim {len(vector)}, "
                f"config expects {spec.dim} — pgvector column mismatch"
            )
        return vector

    async def cancel(self, request_id: str) -> None:
        # Ollama has no cancel endpoint; aborting is done by closing the
        # stream context. Real per-request cancellation lands with barge-in
        # (build-order pass 4).
        return None

    async def aclose(self) -> None:
        await self._client.aclose()
