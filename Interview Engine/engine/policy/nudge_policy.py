"""
Nudge policy (requirements §9) — deterministic. Decides WHETHER a quality
signal becomes a spoken nudge; the interviewer model only decides the
words.

    cooldown            60 s per category
    max_per_round       2 per category
    defer_to_boundary   yes, except AUDIO_DEAD (spoken immediately)
    escalation          gentle -> direct on the 2nd fire

Timestamps are float seconds (time.monotonic in production, plain numbers
in tests).
"""

from __future__ import annotations

from engine.schemas.nudge_event import NudgeCategory, NudgeEvent, Severity

_COOLDOWN_S = 60.0
_MAX_PER_ROUND = 2
_IMMEDIATE = {NudgeCategory.AUDIO_DEAD}


class NudgePolicy:
    def __init__(self, cooldown_s: float = _COOLDOWN_S, max_per_round: int = _MAX_PER_ROUND):
        self._cooldown_s = cooldown_s
        self._max_per_round = max_per_round
        self._fired_at: dict[NudgeCategory, list[float]] = {}

    def _history(self, category: NudgeCategory) -> list[float]:
        return self._fired_at.setdefault(category, [])

    def should_fire(self, category: NudgeCategory, now: float) -> bool:
        history = self._history(category)
        if len(history) >= self._max_per_round:
            return False
        if history and now - history[-1] < self._cooldown_s:
            return False
        return True

    def build_event(self, category: NudgeCategory, now: float) -> NudgeEvent | None:
        if not self.should_fire(category, now):
            return None
        history = self._history(category)
        severity = Severity.DIRECT if history else Severity.GENTLE
        history.append(now)
        return NudgeEvent(
            category=category,
            severity=severity,
            deferred=category not in _IMMEDIATE,
        )

    def reset(self) -> None:
        """Call at round boundaries — counters are per-round."""
        self._fired_at.clear()
