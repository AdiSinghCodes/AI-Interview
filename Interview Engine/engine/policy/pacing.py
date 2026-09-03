"""
Pacing — deterministic decisions about how the round flows:

  - when to probe with a live follow-up instead of moving to the next
    planned question (requirements §8: follow up on what was actually said)
  - when the round is over

Follow-ups cost a live LLM + TTS turn, so they're kept occasional: only
after a substantive answer, never twice in a row, capped per round.
"""

from __future__ import annotations

_SUBSTANTIVE_WORDS = 22
_MAX_FOLLOWUPS = 3


class Pacing:
    def __init__(self, num_planned: int, *, max_followups: int = _MAX_FOLLOWUPS):
        self._num_planned = num_planned
        self._max_followups = max_followups
        self._followups_used = 0
        self._last_was_followup = False

    def should_follow_up(self, last_answer: str, *, planned_cursor: int) -> bool:
        if self._last_was_followup or self._followups_used >= self._max_followups:
            return False
        if planned_cursor >= self._num_planned:      # no planned questions left to defer
            return False
        if planned_cursor == 0:                       # let the first real answer land plainly
            return False
        words = len(last_answer.split())
        # substantive answer, and roughly every other eligible turn
        return words >= _SUBSTANTIVE_WORDS and (self._followups_used + planned_cursor) % 2 == 0

    def record_turn(self, *, was_followup: bool) -> None:
        self._last_was_followup = was_followup
        if was_followup:
            self._followups_used += 1

    def is_complete(self, *, planned_cursor: int, total_turns: int) -> bool:
        return planned_cursor >= self._num_planned or total_turns >= self._num_planned * 2 + 4
