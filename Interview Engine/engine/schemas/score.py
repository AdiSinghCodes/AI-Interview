"""
DimensionScore / Report — the scorer's output. Scores are relational rows
(not JSONB): the progress dashboard queries them. Every score must cite
the specific turn(s) it is based on (requirements §4).

STUB — landed in build-order pass 10.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    dimension: str
    value: float                    # 0-100
    evidence_turn_indices: list[int] = Field(default_factory=list)
    rationale: str = ""


class Report(BaseModel):
    round_id: str
    dimension_scores: list[DimensionScore]
    overall: float = 0.0
    proctor_integrity_score: float | None = None   # folded in from the Phase 1 log
    summary: str = ""
