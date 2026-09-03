"""
InferenceProvider — the interface every backend implements.

Swapping the backend for a role = editing config/models.yaml, not this
file or any caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel

from engine.settings import LLMSpec


class ChatMessage(BaseModel):
    role: str          # "system" | "user" | "assistant"
    content: str


class GenerationResult(BaseModel):
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None       # time to first token — the number that matters live
    total_ms: float | None = None
    model_used: str | None = None      # which spec.name actually answered (post-fallback)


class InferenceError(RuntimeError):
    """Backend call failed. RoutingProvider catches this to try the next spec."""


class InferenceProvider(ABC):
    """One instance per backend (Ollama, Groq, ...)."""

    @abstractmethod
    async def generate(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> GenerationResult:
        """Non-streaming call. Used for planner, parser, scorer."""

    @abstractmethod
    def stream(
        self, spec: LLMSpec, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        """Streaming call for the live interviewer turn. Yields raw text deltas —
        the clause chunker, not this method, decides when a chunk is TTS-ready."""

    @abstractmethod
    async def embed(self, spec: LLMSpec, text: str) -> list[float]:
        ...

    @abstractmethod
    async def cancel(self, request_id: str) -> None:
        """Abort an in-flight generation — required for barge-in. No-op on an
        id the backend doesn't recognise."""
