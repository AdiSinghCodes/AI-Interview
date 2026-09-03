"""
SessionOrchestrator — the spine of the engine (plan p5). Owns the turn
state machine, the transcript, the plan cursor, pacing, the degrade
ladder, and per-turn latency tracing. Transport and speech are adapters
around it; they never own turn logic.

Per turn it decides, in order:
  1. round complete?           -> speak the closing line, end
  2. probe with a follow-up?   -> live LLM + TTS turn (pacing says when)
  3. planned question cached?  -> play the prefetched audio, instantly
  4. otherwise                 -> generate the question live

Text callers (`run_turn_loop_text.py`) pass no prefetch cache and get the
live path for everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.agents import interviewer
from engine.inference.provider import InferenceProvider
from engine.orchestrator.prefetch_cache import CLOSING_LINE, PrefetchCache
from engine.policy.pacing import Pacing
from engine.schemas.interview_plan import InterviewPlan
from engine.schemas.latency_trace import Stage
from engine.schemas.nudge_event import NudgeEvent
from engine.schemas.transcript import Transcript, Turn, TurnKind, TurnRole
from engine.settings import Role, get_models

from .degrade_ladder import DegradeLadder
from .latency_tracer import LatencyTracer
from .turn_state import IllegalTransition, TurnState, advance


@dataclass
class TurnResult:
    interviewer_text: str
    kind: TurnKind
    state: TurnState
    turn_index: int
    ttft_ms: float | None
    trace_marks: dict[str, float] = field(default_factory=dict)
    degrade_level: str = "full quality"
    cache_hit: bool = False
    prefetched_audio: bytes | None = None
    interview_over: bool = False


class SessionOrchestrator:
    def __init__(
        self,
        round_id: str,
        plan: InterviewPlan,
        provider: InferenceProvider,
        *,
        tracer: LatencyTracer | None = None,
        prefetch: PrefetchCache | None = None,
    ):
        self.round_id = round_id
        self.plan = plan
        self.state = TurnState.IDLE
        self.transcript = Transcript(round_id=round_id)
        self.tracer = tracer or LatencyTracer()

        self._provider = provider
        self._prefetch = prefetch
        self._ladder = DegradeLadder(self.tracer)
        self._pacing = Pacing(len(plan.questions))
        self._cursor = 0
        self._turns_taken = 0
        self._pending_nudge: NudgeEvent | None = None

    @property
    def opening_audio(self) -> bytes | None:
        return self._prefetch.opening.pcm if self._prefetch and self._prefetch.opening else None

    # -- lifecycle ----------------------------------------------------
    async def start(self) -> str:
        self._transition(TurnState.SPEAKING)
        opening = self.plan.persona.opening_line
        self.transcript.append(
            Turn(index=self.transcript.next_index, role=TurnRole.INTERVIEWER,
                 kind=TurnKind.OPENING, text=opening)
        )
        self._transition(TurnState.LISTENING)
        return opening

    async def submit_answer(self, text: str, *, trace=None, on_clause=None) -> TurnResult:
        if self.state != TurnState.LISTENING:
            raise IllegalTransition(f"submit_answer called while {self.state.value}")

        if trace is None:
            trace = self.tracer.new_turn(self.transcript.next_index + 1)
            trace.mark(Stage.ANSWER_RECEIVED)
            trace.mark(Stage.STT_FINAL)
        else:
            self.tracer.adopt(trace)

        self.transcript.append(
            Turn(index=self.transcript.next_index, role=TurnRole.CANDIDATE,
                 kind=TurnKind.ANSWER, text=text)
        )

        self._transition(TurnState.THINKING)
        reply = await self._run_thinking(text, trace, on_clause)
        if self.state == TurnState.THINKING:
            self._transition(TurnState.SPEAKING)

        self.transcript.append(
            Turn(
                index=self.transcript.next_index,
                role=TurnRole.INTERVIEWER,
                kind=reply.kind,
                text=reply.text,
                plan_question_id=reply.plan_question_id,
                context_snapshot=[m.model_dump() for m in reply.messages],
                latency_ms=dict(trace.marks),
            )
        )

        self._turns_taken += 1
        if reply.kind in (TurnKind.QUESTION, TurnKind.NUDGE):
            self._cursor += 1  # a nudge turn folds the next planned question in with it
        self._pacing.record_turn(was_followup=reply.kind == TurnKind.FOLLOWUP)
        self._ladder.reassess()

        over = reply.kind == TurnKind.CLOSING
        self._transition(TurnState.ENDED if over else TurnState.LISTENING)

        return TurnResult(
            interviewer_text=reply.text,
            kind=reply.kind,
            state=self.state,
            turn_index=self.transcript.turns[-1].index,
            ttft_ms=reply.ttft_ms,
            trace_marks=dict(trace.marks),
            degrade_level=self._ladder.describe(),
            cache_hit=reply.prefetched_audio is not None,
            prefetched_audio=reply.prefetched_audio,
            interview_over=over,
        )

    def end(self) -> None:
        if self.state != TurnState.ENDED:
            self._transition(TurnState.ENDED)

    def inject_nudge(self, nudge: NudgeEvent) -> None:
        self._pending_nudge = nudge

    # -- internals -------------------------------------------------
    async def _run_thinking(self, last_answer, trace, on_clause) -> interviewer.InterviewerReply:
        trace.mark(Stage.CONTEXT_READY)

        if self._pacing.is_complete(planned_cursor=self._cursor, total_turns=self._turns_taken):
            return self._closing_reply(trace)

        follow_up = self._pacing.should_follow_up(last_answer, planned_cursor=self._cursor)
        next_planned = None if follow_up else self.plan.question_at(self._cursor)

        if next_planned is not None and self._pending_nudge is None and self._prefetch is not None:
            cached = self._prefetch.get_question(next_planned.id)
            if cached is not None:
                return self._cached_reply(next_planned.id, cached, trace)

        return await self._live_reply(next_planned, trace, on_clause)

    def _cached_reply(self, qid, cached, trace) -> interviewer.InterviewerReply:
        self._transition(TurnState.SPEAKING)
        ack = self._prefetch.random_ack()
        trace.mark(Stage.FIRST_TTS_CHUNK)
        trace.mark(Stage.PLAYBACK_START)
        trace.mark(Stage.RESPONSE_READY)
        return interviewer.InterviewerReply(
            text=f"{ack.text} {cached.text}".strip(),
            kind=TurnKind.QUESTION,
            plan_question_id=qid,
            prefetched_audio=ack.pcm + cached.pcm,
        )

    def _closing_reply(self, trace) -> interviewer.InterviewerReply:
        self._transition(TurnState.SPEAKING)
        audio = self._prefetch.closing.pcm if self._prefetch and self._prefetch.closing else None
        if audio is not None:
            trace.mark(Stage.FIRST_TTS_CHUNK)
            trace.mark(Stage.PLAYBACK_START)
        trace.mark(Stage.RESPONSE_READY)
        return interviewer.InterviewerReply(
            text=CLOSING_LINE, kind=TurnKind.CLOSING, prefetched_audio=audio
        )

    async def _live_reply(self, next_planned, trace, on_clause) -> interviewer.InterviewerReply:
        spec = self._ladder.apply_to(get_models().roles[Role.INTERVIEWER])

        async def clause_sink(clause: str) -> None:
            if self.state == TurnState.THINKING:
                self._transition(TurnState.SPEAKING)
            if on_clause is not None:
                await on_clause(clause)

        reply = await interviewer.next_turn(
            self._provider,
            self.plan,
            self.transcript,
            spec=spec,
            next_planned=next_planned,
            pending_nudge=self._pending_nudge,
            retrieved_context=None,
            trace=trace,
            on_clause=clause_sink,
        )
        self._pending_nudge = None
        return reply

    def _transition(self, target: TurnState) -> None:
        self.state = advance(self.state, target)
