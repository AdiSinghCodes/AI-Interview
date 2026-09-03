"""
TurnTrace — per-turn stage instrumentation. The plan is blunt about this
(p9, module #11): "ship this in week 1, not week 8. You cannot optimise
what you have not measured."

One TurnTrace per candidate utterance. `mark()` stamps a stage the moment
it completes; everything is milliseconds since the turn clock started
(ANSWER_RECEIVED). The audio stages exist now so the shape doesn't change
when STT/TTS land — in text mode only ANSWER_RECEIVED, TTFT and
RESPONSE_READY get marked.

Aggregation across turns (p50/p95, the degrade-ladder trigger) lives in
`engine.orchestrator.latency_tracer`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    ANSWER_RECEIVED = "answer_received"   # text in hand, or VAD endpoint fired
    STT_FINAL = "stt_final"               # transcription finalised
    CONTEXT_READY = "context_ready"       # retrieval / prompt assembly done
    TTFT = "ttft"                         # first LLM token out
    FIRST_TTS_CHUNK = "first_tts_chunk"   # first synthesised audio chunk
    PLAYBACK_START = "playback_start"     # audio reaches the client
    RESPONSE_READY = "response_ready"     # full interviewer utterance assembled

    @classmethod
    def order(cls) -> list[str]:
        return [s.value for s in cls]


@dataclass
class TurnTrace:
    turn_index: int
    cache_hit: bool = False
    _origin: float = field(default_factory=time.perf_counter, repr=False)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: Stage | str) -> None:
        key = stage.value if isinstance(stage, Stage) else stage
        self.marks[key] = (time.perf_counter() - self._origin) * 1000.0

    def ms(self, stage: Stage | str) -> float | None:
        key = stage.value if isinstance(stage, Stage) else stage
        return self.marks.get(key)

    def total_ms(self) -> float:
        return max(self.marks.values(), default=0.0)

    def ordered(self) -> list[tuple[str, float]]:
        return [(s, self.marks[s]) for s in Stage.order() if s in self.marks]

    def deltas(self) -> dict[str, float]:
        """Time spent IN each stage (gap from the previous marked stage)."""
        out: dict[str, float] = {}
        prev = 0.0
        for name, val in self.ordered():
            out[name] = val - prev
            prev = val
        return out
