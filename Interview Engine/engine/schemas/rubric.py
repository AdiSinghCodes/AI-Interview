"""
Rubric — AI-generated once per session, then FROZEN and persisted
(requirements §7). Scoring runs against the stored copy so the same
answer scores the same across re-runs.

Dimensions stay stable across sessions (comparable scores); criteria and
weights within a dimension are generated per role.

STUB — landed in build-order pass 10 (scorer + report).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RubricDimension(BaseModel):
    name: str                      # e.g. "technical correctness"
    weight: float
    criteria: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    round_type: str
    dimensions: list[RubricDimension]
    frozen: bool = False
