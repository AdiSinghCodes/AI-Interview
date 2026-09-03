"""Orchestrator flow with a fake provider — no Ollama needed."""

import json

import pytest

from engine.inference.provider import ChatMessage, GenerationResult, InferenceProvider
from engine.orchestrator import SessionOrchestrator, TurnState
from engine.orchestrator.turn_state import IllegalTransition
from engine.schemas.interview_plan import InterviewPlan, Persona, PlannedQuestion
from engine.schemas.transcript import TurnKind, TurnRole


class FakeProvider(InferenceProvider):
    """Streams a canned {"answer": ...} for each turn, in small deltas."""

    def __init__(self, replies: list[str]):
        self._replies = replies
        self.calls = 0

    async def generate(self, spec, messages) -> GenerationResult:
        return GenerationResult(text='{"answer": "ok"}')

    async def stream(self, spec, messages):
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        payload = json.dumps({"answer": reply})
        for i in range(0, len(payload), 7):
            yield payload[i : i + 7]

    async def embed(self, spec, text) -> list[float]:
        return [0.0] * 768

    async def cancel(self, request_id) -> None:
        return None


def _plan(n: int) -> InterviewPlan:
    return InterviewPlan(
        round_type="technical",
        persona=Persona(opening_line="Hello, let's begin."),
        questions=[
            PlannedQuestion(id=f"q{i}", topic="t", prompt=f"question {i}?")
            for i in range(n)
        ],
        duration_min=30,
    )


async def test_start_speaks_opening_then_listens():
    orch = SessionOrchestrator("r", _plan(3), FakeProvider(["a"]))
    opening = await orch.start()
    assert opening == "Hello, let's begin."
    assert orch.state == TurnState.LISTENING
    assert orch.transcript.turns[0].kind == TurnKind.OPENING


async def test_one_turn_advances_and_returns_to_listening():
    orch = SessionOrchestrator("r", _plan(3), FakeProvider(["First follow-up.", "Second."]))
    await orch.start()
    res = await orch.submit_answer("my answer about services")

    assert res.interviewer_text == "First follow-up."
    assert res.interview_over is False
    assert orch.state == TurnState.LISTENING
    assert [t.role for t in orch.transcript.turns[-2:]] == [
        TurnRole.CANDIDATE,
        TurnRole.INTERVIEWER,
    ]
    assert orch.transcript.turns[-1].latency_ms  # trace was captured onto the turn


async def test_interview_ends_with_a_closing_turn():
    orch = SessionOrchestrator("r", _plan(2), FakeProvider(["r1", "r2", "r3"]))
    await orch.start()
    await orch.submit_answer("a1")               # asks q0
    r2 = await orch.submit_answer("a2")          # asks q1 (last planned)
    assert r2.interview_over is False
    r3 = await orch.submit_answer("a3")          # closing line
    assert r3.kind == TurnKind.CLOSING
    assert r3.interview_over is True
    assert orch.state == TurnState.ENDED


async def test_submit_answer_rejected_outside_listening():
    orch = SessionOrchestrator("r", _plan(1), FakeProvider(["r1"]))
    with pytest.raises(IllegalTransition):
        await orch.submit_answer("too early")  # state is still IDLE


async def test_tracer_captures_ttft_and_response_ready():
    orch = SessionOrchestrator("r", _plan(2), FakeProvider(["hello there candidate"]))
    await orch.start()
    res = await orch.submit_answer("a1")
    assert res.ttft_ms is not None and res.ttft_ms >= 0
    assert "response_ready" in res.trace_marks
    assert orch.tracer.summary()["turns"] == 1


async def test_nudge_is_folded_into_the_next_turn():
    from engine.schemas.nudge_event import NudgeCategory, NudgeEvent

    orch = SessionOrchestrator("r", _plan(3), FakeProvider(["noted, and here is a question"]))
    await orch.start()
    orch.inject_nudge(NudgeEvent(category=NudgeCategory.AUDIO_LOW))
    res = await orch.submit_answer("a1")
    assert res.kind == TurnKind.NUDGE
    # nudge is one-shot — the next turn is a normal question
    res2 = await orch.submit_answer("a2")
    assert res2.kind == TurnKind.QUESTION
