"""
RoutingProvider — dispatches each call to the backend named on its own
`LLMSpec.provider`, and walks `spec.fallback` when a call fails.

Callers hold one of these and never branch on backend. "scorer -> Groq,
interviewer -> Ollama, Groq 429 -> local qwen3:8b" is entirely config +
this file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from engine.settings import LLMSpec, Provider, get_settings
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .provider import (
    ChatMessage,
    GenerationResult,
    InferenceError,
    InferenceProvider,
)


class RoutingProvider(InferenceProvider):
    def __init__(self, ollama: OllamaProvider, groq: GroqProvider | None = None):
        self._ollama = ollama
        self._groq = groq

    def _backend(self, spec: LLMSpec) -> InferenceProvider:
        if spec.provider == Provider.GROQ:
            if self._groq is None:
                raise InferenceError(
                    f"{spec.name!r} is configured for Groq but no API key is set"
                )
            return self._groq
        return self._ollama

    def _chain(self, spec: LLMSpec) -> list[LLMSpec]:
        return [spec, *spec.fallback]

    async def generate(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> GenerationResult:
        last: Exception | None = None
        for candidate in self._chain(spec):
            try:
                return await self._backend(candidate).generate(candidate, messages)
            except InferenceError as e:
                last = e
        raise last or InferenceError("empty spec chain")

    async def stream(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        last: Exception | None = None
        for candidate in self._chain(spec):
            started = False
            try:
                async for chunk in self._backend(candidate).stream(candidate, messages):
                    started = True
                    yield chunk
                return
            except InferenceError as e:
                # Once text is out, switching models would splice two voices
                # into one answer — only fall back on a clean pre-first-token
                # failure.
                if started:
                    raise
                last = e
        raise last or InferenceError("empty spec chain")

    async def embed(self, spec: LLMSpec, text: str) -> list[float]:
        last: Exception | None = None
        for candidate in self._chain(spec):
            try:
                return await self._backend(candidate).embed(candidate, text)
            except InferenceError as e:
                last = e
        raise last or InferenceError("empty spec chain")

    async def cancel(self, request_id: str) -> None:
        await self._ollama.cancel(request_id)
        if self._groq is not None:
            await self._groq.cancel(request_id)

    async def aclose(self) -> None:
        await self._ollama.aclose()


def build_router() -> RoutingProvider:
    """Construct the standard provider from settings — Ollama always, Groq
    only if a key is present."""
    settings = get_settings()
    ollama = OllamaProvider(settings.ollama_host)
    groq: GroqProvider | None = None
    if settings.groq_api_key:
        try:
            groq = GroqProvider(settings.groq_api_key)
        except InferenceError:
            groq = None
    return RoutingProvider(ollama, groq)
