"""Tiny helper so each plan module stays declarative."""

from __future__ import annotations

from engine.schemas.interview_plan import (
    InterviewPlan,
    Persona,
    PlannedQuestion,
    Topic,
)


def q(id_: str, topic: str, prompt: str, points: list[str], difficulty: int = 3, minutes: float = 3.0) -> PlannedQuestion:
    return PlannedQuestion(
        id=id_, topic=topic, prompt=prompt, expected_points=points,
        difficulty=difficulty, est_minutes=minutes,
    )


def plan(
    *,
    round_type: str,
    persona: Persona,
    topics: list[str],
    questions: list[PlannedQuestion],
    criteria: list[str],
) -> InterviewPlan:
    return InterviewPlan(
        round_type=round_type,
        persona=persona,
        topics=[Topic(name=t) for t in topics],
        questions=questions,
        criteria=criteria,
        duration_min=30,
    )
