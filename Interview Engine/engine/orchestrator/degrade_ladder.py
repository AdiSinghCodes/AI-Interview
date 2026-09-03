"""
Degrade ladder (plan p9, module #12). When TTFT p95 crosses the danger
line the interview must get faster, not hang. Each rung trades a bit of
quality for latency, in this order:

    1  drop retrieval          (skip pgvector context)
    2  drop the JSON schema    (looser prompt, fewer output tokens)
    3  force the fallback model (llama3.2:3b — smallest, fastest cold load)
    4  static-image avatar      (drop the mesh render path)

Rungs only ratchet up. Recovery (stepping back down after conditions
improve) is deliberately out of scope for now — a flapping ladder is
worse than a sticky one.
"""

from __future__ import annotations

from enum import IntEnum

from engine.settings import LLMSpec
from .latency_tracer import LatencyTracer


class DegradeLevel(IntEnum):
    NONE = 0
    NO_RETRIEVAL = 1
    NO_SCHEMA = 2
    FALLBACK_MODEL = 3
    STATIC_AVATAR = 4


class DegradeLadder:
    def __init__(self, tracer: LatencyTracer):
        self._tracer = tracer
        self.level = DegradeLevel.NONE

    def reassess(self) -> DegradeLevel:
        if self._tracer.over_danger_line() and self.level < DegradeLevel.STATIC_AVATAR:
            self.level = DegradeLevel(self.level + 1)
        return self.level

    @property
    def retrieval_enabled(self) -> bool:
        return self.level < DegradeLevel.NO_RETRIEVAL

    @property
    def json_schema_enabled(self) -> bool:
        return self.level < DegradeLevel.NO_SCHEMA

    @property
    def force_fallback_model(self) -> bool:
        return self.level >= DegradeLevel.FALLBACK_MODEL

    @property
    def static_avatar(self) -> bool:
        return self.level >= DegradeLevel.STATIC_AVATAR

    def apply_to(self, spec: LLMSpec) -> LLMSpec:
        """Return the effective interviewer spec for the current rung."""
        effective = spec
        if self.force_fallback_model and spec.fallback:
            effective = spec.fallback[-1].model_copy(deep=True)
        else:
            effective = spec.model_copy(deep=True)
        if not self.json_schema_enabled:
            effective.json_schema = None
            effective.json_schema_def = None
        return effective

    def describe(self) -> str:
        return {
            DegradeLevel.NONE: "full quality",
            DegradeLevel.NO_RETRIEVAL: "retrieval dropped",
            DegradeLevel.NO_SCHEMA: "retrieval + JSON schema dropped",
            DegradeLevel.FALLBACK_MODEL: "on fallback model, no schema, no retrieval",
            DegradeLevel.STATIC_AVATAR: "minimum: fallback model + static avatar",
        }[self.level]
