"""
Prefetch cache (plan p5, module #5) — the big latency win. Planned
questions are the bulk of an interview; synthesise them (and the opening,
the closing, and a handful of generic acknowledgements) up front, while
the candidate is on the device-check screen. A planned-question turn then
plays instantly; only genuine follow-ups pay the live LLM + TTS cost.

Audio is 24 kHz mono float32 LE PCM bytes — the same wire format
`ws_live` streams.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import numpy as np

from engine.schemas.interview_plan import InterviewPlan

CLOSING_LINE = (
    "That's everything I wanted to cover. Thanks a lot for taking the time "
    "today, it was good talking with you. The team will be in touch about "
    "the next steps."
)

_ACKS = [
    "Okay, thank you.",
    "Got it, thanks.",
    "That's helpful.",
    "Alright.",
]

# The cache reports `ready` once this much is synthesised; the rest of the
# questions keep warming in the background and any that a turn reaches before
# they're done just fall back to a live turn.
_READY_AFTER_QUESTIONS = 3


@dataclass
class PrefetchedTurn:
    text: str
    pcm: bytes


async def _synth_pcm(tts, text: str) -> bytes:
    chunks = [c async for c in tts.synthesize(text)]
    if not chunks:
        return b""
    return np.concatenate(chunks).astype("<f4").tobytes()


@dataclass
class PrefetchCache:
    opening: PrefetchedTurn | None = None
    closing: PrefetchedTurn | None = None
    _questions: dict[str, PrefetchedTurn] = field(default_factory=dict)
    _acks: list[PrefetchedTurn] = field(default_factory=list)
    _total: int = 0
    _done: int = 0
    _ready: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def progress(self) -> tuple[int, int]:
        return (self._done, self._total)

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    async def warm(self, plan: InterviewPlan, tts) -> None:
        # opening + acks first (so "ready" comes fast), then questions, then closing
        jobs: list[tuple[str, str]] = [("__opening__", plan.persona.opening_line)]
        jobs += [(f"__ack{i}__", a) for i, a in enumerate(_ACKS)]
        jobs += [(q.id, q.prompt) for q in plan.questions]
        jobs.append(("__closing__", CLOSING_LINE))
        self._total = len(jobs)
        ready_at = 1 + len(_ACKS) + min(_READY_AFTER_QUESTIONS, len(plan.questions))

        for key, text in jobs:
            turn = PrefetchedTurn(text=text, pcm=await _synth_pcm(tts, text))
            if key == "__opening__":
                self.opening = turn
            elif key == "__closing__":
                self.closing = turn
            elif key.startswith("__ack"):
                self._acks.append(turn)
            else:
                self._questions[key] = turn
            self._done += 1
            if self._done >= ready_at:
                self._ready.set()
        self._ready.set()

    def get_question(self, question_id: str) -> PrefetchedTurn | None:
        return self._questions.get(question_id)

    def random_ack(self) -> PrefetchedTurn:
        return random.choice(self._acks) if self._acks else PrefetchedTurn("Okay.", b"")
