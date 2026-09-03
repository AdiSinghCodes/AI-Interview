"""
Assembles the message list for each interviewer turn. This is the file
most likely to drift into a question bank — it must not. Question text
comes from the InterviewPlan; persona comes from the plan's Persona;
this file only arranges them.
"""

from __future__ import annotations

from engine.schemas.interview_plan import InterviewPlan, PlannedQuestion
from engine.schemas.nudge_event import NudgeEvent
from engine.schemas.transcript import Transcript, TurnRole
from .provider import ChatMessage

# Adapted from Phase 0.2 interviewer/test_01. The JSON-object constraint is
# the structural fix for small-model prose-leak: a reasoning monologue
# cannot validate against {"answer": "..."}.
_BASE_SYSTEM = (
    "You are an interviewer speaking aloud to a candidate. Respond with a "
    "JSON object containing one field, 'answer', whose value is exactly what "
    "you would say out loud. That value is converted directly to speech: "
    "plain conversational sentences, no markdown, no lists, no formatting. "
    "Keep it to two or three sentences. Acknowledge what the candidate just "
    "said in a few words, then ask one question. Never respond to a question "
    "with another question about the same thing, and never just restate what "
    "was asked. If the candidate's answer opens an interesting thread, follow "
    "up on it instead of moving to the next planned topic. Put only the spoken "
    "words in 'answer' — never your reasoning about what to ask."
)

_RECENT_TURNS = 8


def _persona_system(plan: InterviewPlan) -> str:
    p = plan.persona
    extra = f" {p.system_prompt_extra}" if p.system_prompt_extra else ""
    return (
        f"{_BASE_SYSTEM}\n\n"
        f"You are {p.name}, a {p.role_title}. Your manner is {p.demeanour}. "
        f"This is a {plan.round_type} round.{extra}"
    )


def _plan_framing(plan: InterviewPlan, next_planned: PlannedQuestion | None) -> str:
    topics = ", ".join(t.name for t in plan.topics) or "(open)"
    lines = [
        f"Round focus: {topics}.",
        f"What you are assessing: {', '.join(plan.criteria) or 'general competence'}.",
    ]
    if next_planned is not None:
        pts = "; ".join(next_planned.expected_points)
        lines.append(
            f"The next planned question covers '{next_planned.topic}': "
            f"\"{next_planned.prompt}\""
            + (f" (a strong answer touches on: {pts})" if pts else "")
        )
        lines.append(
            "Ask this next unless the candidate's last answer clearly warrants a "
            "follow-up first."
        )
    else:
        lines.append("There are no more planned questions — wind the round down.")
    return "\n".join(lines)


def build_interviewer_messages(
    plan: InterviewPlan,
    transcript: Transcript,
    *,
    next_planned: PlannedQuestion | None = None,
    pending_nudge: NudgeEvent | None = None,
    retrieved_context: str | None = None,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=_persona_system(plan)),
        ChatMessage(role="system", content=_plan_framing(plan, next_planned)),
    ]

    if retrieved_context:
        messages.append(
            ChatMessage(
                role="system",
                content=f"Relevant background on this candidate:\n{retrieved_context}",
            )
        )

    for turn in transcript.recent(_RECENT_TURNS):
        role = "assistant" if turn.role == TurnRole.INTERVIEWER else "user"
        messages.append(ChatMessage(role=role, content=turn.text))

    if pending_nudge is not None:
        messages.append(
            ChatMessage(
                role="system",
                content=(
                    f"[SIGNAL] {pending_nudge.category.value}, "
                    f"severity={pending_nudge.severity.value}, "
                    f"deferred={str(pending_nudge.deferred).lower()} — fold a brief, "
                    f"warm mention of this into your next utterance, as one sentence "
                    f"before your question. Do not make it a separate turn."
                ),
            )
        )

    return messages
