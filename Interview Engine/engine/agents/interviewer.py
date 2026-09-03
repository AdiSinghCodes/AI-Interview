"""
Live turn generation. Called once per candidate utterance.

Streams from the routing provider. When `on_clause` is given, each
clause is handed off (to TTS) the moment the chunker completes it —
audio starts before the model finishes the sentence (plan p5). The
model is schema-constrained to `{"answer": "..."}`, so `_AnswerStreamer`
pulls that string field out of the token stream incrementally rather
than waiting for the whole object.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from engine.inference.prompt_builder import build_interviewer_messages
from engine.inference.provider import ChatMessage, InferenceProvider
from engine.schemas.interview_plan import InterviewPlan, PlannedQuestion
from engine.schemas.latency_trace import Stage, TurnTrace
from engine.schemas.nudge_event import NudgeEvent
from engine.schemas.transcript import Transcript, TurnKind
from engine.settings import LLMSpec
from engine.speech.clause_chunker import ClauseChunker

OnClause = Callable[[str], Awaitable[None]]

_UNESCAPE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


@dataclass
class InterviewerReply:
    text: str
    kind: TurnKind
    plan_question_id: str | None = None
    model_used: str | None = None
    ttft_ms: float | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    prefetched_audio: bytes | None = None   # set on a cache hit; no live synth happened


class _AnswerStreamer:
    """Incrementally decodes the value of the top-level "answer" string
    field from a streamed JSON object."""

    _OPENER = re.compile(r'"answer"\s*:\s*"')

    def __init__(self) -> None:
        self._pre = ""
        self._in_value = False
        self._escape = False
        self.done = False

    def feed(self, delta: str) -> str:
        if self.done:
            return ""
        out: list[str] = []
        for ch in delta:
            if not self._in_value:
                self._pre += ch
                if self._OPENER.search(self._pre):
                    self._in_value = True
                    self._pre = ""
                continue
            if self._escape:
                out.append(_UNESCAPE.get(ch, ch))
                self._escape = False
            elif ch == "\\":
                self._escape = True
            elif ch == '"':
                self.done = True
                break
            else:
                out.append(ch)
        return "".join(out)


def strip_markdown(text: str) -> str:
    """TTS safety net (from Phase 0.2 interviewer/test_01)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    text = re.sub(r"`+", "", text)
    return re.sub(r"\n{2,}", " ", text).strip()


def _extract_answer(raw: str) -> str:
    try:
        return str(json.loads(raw)["answer"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return raw.strip()


def _rechunk(text: str) -> list[str]:
    c = ClauseChunker()
    return c.feed(text) + c.flush()


async def next_turn(
    provider: InferenceProvider,
    plan: InterviewPlan,
    transcript: Transcript,
    *,
    spec: LLMSpec,
    next_planned: PlannedQuestion | None,
    pending_nudge: NudgeEvent | None = None,
    retrieved_context: str | None = None,
    trace: TurnTrace | None = None,
    on_clause: OnClause | None = None,
) -> InterviewerReply:
    messages = build_interviewer_messages(
        plan,
        transcript,
        next_planned=next_planned,
        pending_nudge=pending_nudge,
        retrieved_context=retrieved_context,
    )

    chunker = ClauseChunker()
    streamer = _AnswerStreamer() if spec.json_schema_def is not None else None
    raw = ""
    answer_parts: list[str] = []
    emitted_any = False
    first_token = False

    async def _emit(clause: str) -> None:
        nonlocal emitted_any
        emitted_any = True
        if on_clause is not None:
            await on_clause(clause)

    async for delta in provider.stream(spec, messages):
        if not first_token:
            first_token = True
            if trace is not None:
                trace.mark(Stage.TTFT)
        raw += delta
        piece = streamer.feed(delta) if streamer is not None else delta
        if not piece:
            continue
        answer_parts.append(piece)
        for clause in chunker.feed(piece):
            await _emit(clause)

    for clause in chunker.flush():
        await _emit(clause)

    answer_text = "".join(answer_parts).strip() or _extract_answer(raw)
    if not emitted_any and answer_text:
        for clause in _rechunk(answer_text):
            await _emit(clause)

    if trace is not None:
        trace.mark(Stage.RESPONSE_READY)

    if pending_nudge is not None:
        kind = TurnKind.NUDGE
    elif next_planned is not None:
        kind = TurnKind.QUESTION
    else:
        kind = TurnKind.FOLLOWUP

    return InterviewerReply(
        text=strip_markdown(answer_text),
        kind=kind,
        plan_question_id=next_planned.id if next_planned else None,
        model_used=spec.name,
        ttft_ms=trace.ms(Stage.TTFT) if trace else None,
        messages=messages,
    )
