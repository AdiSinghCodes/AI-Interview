"""
Transcript + frozen Rubric -> Report. Runs post-session, off the live
path, when VRAM is free again. Groq-first (gpt-oss-20b), local qwen3:8b
fallback. Every DimensionScore must cite the turn indices it rests on.

STUB — build-order pass 10.
"""

from __future__ import annotations

from engine.inference.provider import InferenceProvider
from engine.schemas.rubric import Rubric
from engine.schemas.score import Report
from engine.schemas.transcript import Transcript


async def score_round(
    provider: InferenceProvider, transcript: Transcript, rubric: Rubric
) -> Report:
    raise NotImplementedError("scorer lands in build-order pass 10")
