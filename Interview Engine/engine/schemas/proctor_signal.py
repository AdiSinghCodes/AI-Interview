"""
ProctorSignal — one already-computed observation arriving from the
shipped Phase 1 proctoring module on a SEPARATE channel (never the audio
socket — a YOLO spike must not stutter the interviewer's voice).

The engine reads these; it does not run vision itself. Quality signals
may become a NudgeEvent; integrity signals are logged only.

STUB — landed in build-order pass 8.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class SignalType(str, Enum):
    QUALITY = "quality"       # something is broken — speak it
    INTEGRITY = "integrity"   # something is suspicious — log it


class ProctorSignal(BaseModel):
    type: SignalType
    name: str                 # e.g. "phone_detected", "audio_low"
    confidence: float
    started_at: datetime
    ended_at: datetime | None = None
