"""
POST /v1/preflight — the 15-second device check (requirements §9: "Build
it first"). The browser measures mic RMS, checks the camera opens, and
detects headphones; it POSTs the result here and the engine decides
pass/fail. This eliminates most quality nudges before they can interrupt
anything.

Headphones are required: laptop-speaker AEC is poor, so without them the
interviewer's own audio leaks into the mic and false-triggers the VAD
endpoint (see engine.orchestrator.speech_gate).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["preflight"])

# RMS of normalised (-1..1) mic audio over a spoken test sentence.
_MIN_SPEECH_RMS = 0.02
_MAX_SPEECH_RMS = 0.5


class PreflightResult(BaseModel):
    mic_rms: float
    headphones: bool
    camera_ok: bool


class PreflightVerdict(BaseModel):
    ok: bool
    reasons: list[str]


@router.post("/preflight", response_model=PreflightVerdict)
async def check_preflight(result: PreflightResult) -> PreflightVerdict:
    reasons: list[str] = []
    if not result.headphones:
        reasons.append("Headphones are required so the interviewer's voice doesn't leak into your mic.")
    if result.mic_rms < _MIN_SPEECH_RMS:
        reasons.append("Your microphone is too quiet — move closer or raise the input level.")
    elif result.mic_rms > _MAX_SPEECH_RMS:
        reasons.append("Your microphone is clipping — lower the input level.")
    if not result.camera_ok:
        reasons.append("The camera could not be opened.")
    return PreflightVerdict(ok=not reasons, reasons=reasons)
