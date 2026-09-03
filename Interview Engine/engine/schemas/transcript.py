"""
Transcript — the running record of one round. Written in a
trainable-friendly shape from day one (requirements §20): each Turn keeps
the context the model saw when it produced it, so logged turns can later
become fine-tuning pairs without reconstruction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TurnRole(str, Enum):
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"


class TurnKind(str, Enum):
    OPENING = "opening"       # the persona's opening line
    QUESTION = "question"     # a planned question
    FOLLOWUP = "followup"     # generated from what the candidate just said
    NUDGE = "nudge"           # a quality nudge folded into the utterance
    CLOSING = "closing"       # the wrap-up line at the end of the round
    ANSWER = "answer"         # candidate speech
    INTERRUPTED = "interrupted"  # turn cut short by barge-in


class Turn(BaseModel):
    index: int
    role: TurnRole
    kind: TurnKind
    text: str
    plan_question_id: str | None = None
    started_at: datetime = Field(default_factory=_now)
    context_snapshot: list[dict] = Field(default_factory=list)  # messages the model saw
    latency_ms: dict[str, float] = Field(default_factory=dict)  # from the TurnTrace


class Transcript(BaseModel):
    round_id: str
    turns: list[Turn] = Field(default_factory=list)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)

    @property
    def next_index(self) -> int:
        return len(self.turns)

    def recent(self, n: int) -> list[Turn]:
        return self.turns[-n:]
