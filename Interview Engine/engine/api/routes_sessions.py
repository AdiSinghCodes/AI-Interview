"""
POST /v1/sessions          — create a session, pick the curated plan, start prefetch
GET  /v1/sessions/{id}      — session details
GET  /v1/sessions/{id}/status — prefetch progress (the Preflight screen polls this)
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.api import deps, session_store
from engine.orchestrator.prefetch_cache import PrefetchCache
from engine.plans import DEFAULT_ROLE, get_plan, role_ids

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


class NewSession(BaseModel):
    role: str = DEFAULT_ROLE
    difficulty: str = "medium"
    num_questions: int = Field(default=10, ge=3, le=15)


class SessionInfo(BaseModel):
    session_id: str
    round_id: str
    role: str
    persona_name: str
    num_questions: int


class PrefetchStatus(BaseModel):
    done: int
    total: int
    ready: bool


async def _run_prefetch(prefetch: PrefetchCache, plan) -> None:
    tts = deps.get_tts()
    await asyncio.to_thread(tts.warmup)
    await prefetch.warm(plan, tts)


@router.post("", response_model=SessionInfo)
async def create_session(body: NewSession) -> SessionInfo:
    role = body.role if body.role in role_ids() else DEFAULT_ROLE
    plan = get_plan(role, num_questions=body.num_questions)

    prefetch = PrefetchCache()
    sess = session_store.create_session(role, body.difficulty, plan, prefetch)
    asyncio.create_task(_run_prefetch(prefetch, plan))

    return SessionInfo(
        session_id=sess.session_id,
        round_id=sess.round.round_id,
        role=role,
        persona_name=plan.persona.name,
        num_questions=len(plan.questions),
    )


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    sess = session_store.get_session(session_id)
    if sess is None:
        raise HTTPException(404, "session not found")
    return SessionInfo(
        session_id=sess.session_id,
        round_id=sess.round.round_id,
        role=sess.role,
        persona_name=sess.round.plan.persona.name,
        num_questions=len(sess.round.plan.questions),
    )


@router.get("/{session_id}/status", response_model=PrefetchStatus)
async def session_status(session_id: str) -> PrefetchStatus:
    sess = session_store.get_session(session_id)
    if sess is None:
        raise HTTPException(404, "session not found")
    done, total = sess.round.prefetch.progress
    return PrefetchStatus(done=done, total=total, ready=sess.round.prefetch.ready)
