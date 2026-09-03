"""
Groq backend. Off the live path: parser (pre-interview) and scorer
(post-interview) only. Constructed only when ENGINE_GROQ_API_KEY is set;
otherwise those roles fall straight through to their local qwen3:8b
fallback via RoutingProvider.

A 429 (free-tier ceiling — Phase 0.2 test 5 probes it) is raised as
InferenceError so the fallback engages: rate limits degrade quality, they
do not break the session.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from engine.settings import LLMSpec, get_settings
from .provider import ChatMessage, GenerationResult, InferenceError, InferenceProvider


class GroqProvider(InferenceProvider):
    def __init__(self, api_key: str | None = None):
        key = api_key or get_settings().groq_api_key
        if not key:
            raise InferenceError("GroqProvider constructed without an API key")
        try:
            from groq import AsyncGroq
        except ImportError as e:  # pragma: no cover - optional dep
            raise InferenceError("groq SDK not installed: pip install 'interview-engine[groq]'") from e
        self._client = AsyncGroq(api_key=key)

    def _kwargs(self, spec: LLMSpec, messages: list[ChatMessage], *, stream: bool) -> dict:
        kw: dict = {
            "model": spec.name,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
        }
        if spec.temperature is not None:
            kw["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            kw["max_tokens"] = spec.max_tokens
        if spec.response_format:
            kw["response_format"] = {"type": spec.response_format}
        return kw

    async def generate(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> GenerationResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                **self._kwargs(spec, messages, stream=False)
            )
        except Exception as e:  # noqa: BLE001 - SDK raises many types; all mean "try fallback"
            raise InferenceError(f"groq generate {spec.name!r}: {e}") from e
        elapsed = (time.perf_counter() - t0) * 1000.0
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_ms=elapsed,
            model_used=spec.name,
        )

    async def stream(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                **self._kwargs(spec, messages, stream=True)
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"groq stream {spec.name!r}: {e}") from e

    async def embed(self, spec: LLMSpec, text: str) -> list[float]:
        raise InferenceError("Groq has no embeddings API — embedder must stay on Ollama")

    async def cancel(self, request_id: str) -> None:
        return None
