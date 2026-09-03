"""
Nudge injector (plan p5, module #7). Proctoring should SPEAK quality
corrections, not just log them. This is the queue between the shipped
Phase 1 proctor and the orchestrator:

  - debounce: max 1 spoken nudge per 30 s overall
  - priority queue: AUDIO_DEAD jumps ahead
  - pre-synthesised lines where possible
  - never interrupt a candidate mid-answer — wait for a natural pause
    (a turn boundary), except AUDIO_DEAD

STUB — build-order pass 8. Consumes `engine.policy.nudge_policy` decisions.
"""

from __future__ import annotations

from engine.schemas.nudge_event import NudgeEvent
from engine.schemas.proctor_signal import ProctorSignal


class NudgeInjector:
    def __init__(self) -> None:
        self._queue: list[NudgeEvent] = []

    def offer(self, signal: ProctorSignal) -> None:
        raise NotImplementedError("nudge injector lands in build-order pass 8")

    def take_for_turn_boundary(self) -> NudgeEvent | None:
        return None
