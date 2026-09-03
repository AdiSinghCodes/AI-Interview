"""
Resume / JD -> 768-dim vectors (nomic-embed-text). Must stay on Ollama:
Groq has no embeddings API. The dim guard in `OllamaProvider.embed` fails
loudly on a mismatch rather than silently breaking retrieval.

STUB — build-order pass 7.
"""

from __future__ import annotations

from engine.inference.provider import InferenceProvider
from engine.settings import Role, get_models


async def embed_chunks(provider: InferenceProvider, chunks: list[str]) -> list[list[float]]:
    spec = get_models().roles[Role.EMBEDDER]
    return [await provider.embed(spec, chunk) for chunk in chunks]
