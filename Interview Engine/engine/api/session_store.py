"""
In-memory session + round store. One process, one candidate at a time —
DB persistence and multi-tenancy are a later pass (requirements §12-13).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from engine.orchestrator.prefetch_cache import PrefetchCache
from engine.schemas.interview_plan import InterviewPlan


@dataclass
class Round:
    round_id: str
    plan: InterviewPlan
    prefetch: PrefetchCache
    transcript: list[dict] = field(default_factory=list)   # written when the WS closes
    summary: str | None = None
    started: bool = False


@dataclass
class Session:
    session_id: str
    role: str
    difficulty: str
    round: Round


_sessions: dict[str, Session] = {}
_rounds: dict[str, Round] = {}


def create_session(
    role: str, difficulty: str, plan: InterviewPlan, prefetch: PrefetchCache
) -> Session:
    rnd = Round(round_id=uuid.uuid4().hex[:12], plan=plan, prefetch=prefetch)
    sess = Session(
        session_id=uuid.uuid4().hex[:12], role=role, difficulty=difficulty, round=rnd
    )
    _sessions[sess.session_id] = sess
    _rounds[rnd.round_id] = rnd
    return sess


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def get_round(round_id: str) -> Round | None:
    return _rounds.get(round_id)
