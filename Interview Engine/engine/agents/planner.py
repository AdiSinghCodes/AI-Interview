"""
Resume + InterviewConfig -> InterviewPlan. Runs pre-session, latency-free.
Owns the two-pass generate -> critique -> regenerate flow, plus a
deterministic trim when the plan overruns its time budget.

The SCHEMA is fixed; every value inside is model-generated (requirements
§7). This file holds no question text.

STUB — build-order pass 7. `run_turn_loop_text.py` uses a hand-written
plan until then.
"""

from __future__ import annotations

from engine.inference.provider import InferenceProvider
from engine.schemas.interview_config import InterviewConfig
from engine.schemas.interview_plan import InterviewPlan


class PlanGenerationError(RuntimeError):
    """Model could not produce a schema-valid plan even after repair passes."""


async def generate_plan(
    provider: InferenceProvider,
    config: InterviewConfig,
    resume_json: dict,
    gap_analysis: dict | None = None,
) -> InterviewPlan:
    raise NotImplementedError("planner lands in build-order pass 7")
