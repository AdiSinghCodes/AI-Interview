"""
SpeechGate — the server half of the acoustic-echo fix (requirements §11,
plan p3). While the interviewer is speaking, the candidate's mic is
hearing the avatar; VAD run on that audio fires a false endpoint and the
avatar interrupts itself in a loop. So: ignore VAD entirely during
SPEAKING and for a short tail after playback ends.

The other two parts of the fix live in the browser: `getUserMedia`
echoCancellation/noiseSuppression flags, and requiring headphones in the
pre-flight check.
"""

from __future__ import annotations

import time


class SpeechGate:
    def __init__(self, tail_ms: float = 150.0):
        self._tail_s = tail_ms / 1000.0
        self._speaking = False
        self._stopped_at: float | None = None

    def enter_speaking(self) -> None:
        self._speaking = True
        self._stopped_at = None

    def leave_speaking(self, now: float | None = None) -> None:
        self._speaking = False
        self._stopped_at = time.monotonic() if now is None else now

    def accepting_audio(self, now: float | None = None) -> bool:
        if self._speaking:
            return False
        if self._stopped_at is None:
            return True
        now = time.monotonic() if now is None else now
        return (now - self._stopped_at) >= self._tail_s
