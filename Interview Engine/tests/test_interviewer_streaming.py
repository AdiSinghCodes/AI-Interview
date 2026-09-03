"""interviewer.next_turn streams clauses to `on_clause` as the LLM
generates, and still returns the full assembled reply."""

import json

from engine.agents.interviewer import _AnswerStreamer, next_turn
from engine.plans import get_plan


def demo_plan():
    return get_plan("technical")
from engine.inference.provider import GenerationResult, InferenceProvider
from engine.schemas.transcript import Transcript, Turn, TurnKind, TurnRole
from engine.settings import Role, get_models


class _StreamProvider(InferenceProvider):
    """Streams one JSON answer, in small deltas that split mid-token."""

    def __init__(self, answer: str):
        self._answer = answer

    async def generate(self, spec, messages) -> GenerationResult:
        return GenerationResult(text="{}")

    async def stream(self, spec, messages):
        payload = json.dumps({"answer": self._answer})
        for i in range(0, len(payload), 5):
            yield payload[i : i + 5]

    async def embed(self, spec, text):
        return [0.0] * 768

    async def cancel(self, request_id):
        return None


def _transcript_with_answer() -> Transcript:
    t = Transcript(round_id="r")
    t.append(Turn(index=0, role=TurnRole.CANDIDATE, kind=TurnKind.ANSWER, text="I used FastAPI."))
    return t


async def test_clauses_stream_in_order_and_text_is_assembled():
    plan = demo_plan()
    answer = (
        "That makes sense, and it is a common choice. "
        "How would you handle pagination over a very large result set?"
    )
    clauses: list[str] = []

    async def sink(clause: str) -> None:
        clauses.append(clause)

    reply = await next_turn(
        _StreamProvider(answer),
        plan,
        _transcript_with_answer(),
        spec=get_models().roles[Role.INTERVIEWER],
        next_planned=plan.questions[0],
        on_clause=sink,
    )

    assert clauses, "expected the chunker to emit at least one clause"
    assert " ".join(clauses).split() == reply.text.split()   # same words, in order
    assert clauses[0].startswith("That makes sense")


async def test_answer_streamer_decodes_incrementally():
    s = _AnswerStreamer()
    out = "".join(s.feed(ch) for ch in '{"answer": "Hello, world."}')
    assert out == "Hello, world."
    assert s.done


async def test_answer_streamer_handles_escaped_quote():
    s = _AnswerStreamer()
    raw = '{"answer": "she said \\"hi\\" then left"}'
    out = "".join(s.feed(ch) for ch in raw)
    assert out == 'she said "hi" then left'
