"""
Per-turn retrieval: embed the candidate's last answer, pull the top-k
resume/JD chunks from pgvector, hand them to the prompt builder. Budget
is 20-50 ms (plan p7) — cap k at 3-4, cosine, HNSW index.

First rung of the degrade ladder drops this entirely.

STUB — build-order pass 7.
"""

from __future__ import annotations

from engine.inference.provider import InferenceProvider


class ContextRetriever:
    def __init__(self, provider: InferenceProvider, round_id: str):
        self._provider = provider
        self._round_id = round_id

    async def for_answer(self, answer_text: str, k: int = 3) -> str | None:
        raise NotImplementedError("retrieval lands in build-order pass 7")
