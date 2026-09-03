"""
GET  /v1/rounds/{id}/transcript — the conversation so far / at the end
POST /v1/rounds/{id}/summary    — a short spoken-style wrap-up (not rubric scoring)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import json

from engine.api import deps, session_store
from engine.inference.provider import ChatMessage, InferenceError
from engine.settings import Role, get_models

router = APIRouter(prefix="/v1/rounds", tags=["rounds"])

_SUMMARY_SYSTEM = (
    "You are an interviewer writing a brief internal note right after the "
    "interview. Respond with a JSON object with one field, 'summary', whose "
    "value is four or five plain sentences on how the candidate did: what they "
    "were strong on, where they were shakier, and an overall read. No bullet "
    "points, no score, and never include your reasoning about how you wrote it."
)
_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


@router.get("/{round_id}/transcript")
async def get_transcript(round_id: str) -> dict:
    rnd = session_store.get_round(round_id)
    if rnd is None:
        raise HTTPException(404, "round not found")
    return {"round_id": round_id, "turns": rnd.transcript}


@router.post("/{round_id}/summary")
async def make_summary(round_id: str) -> dict:
    rnd = session_store.get_round(round_id)
    if rnd is None:
        raise HTTPException(404, "round not found")
    if rnd.summary:
        return {"summary": rnd.summary}
    if not rnd.transcript:
        raise HTTPException(409, "no transcript yet")

    convo = "\n".join(
        f"{t['role']}: {t['text']}" for t in rnd.transcript if t.get("text")
    )
    spec = get_models().roles[Role.INTERVIEWER].model_copy(deep=True)
    spec.json_schema_def = _SUMMARY_SCHEMA
    spec.max_tokens = 400
    try:
        result = await deps.get_inference_provider().generate(
            spec,
            [
                ChatMessage(role="system", content=_SUMMARY_SYSTEM),
                ChatMessage(role="user", content=convo),
            ],
        )
    except InferenceError as e:
        raise HTTPException(502, f"summary generation failed: {e}") from e

    try:
        rnd.summary = str(json.loads(result.text)["summary"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        rnd.summary = result.text.strip()
    return {"summary": rnd.summary}
