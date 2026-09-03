"""
InterviewPlan — the Planner's output. The SCHEMA is fixed; every value
inside it is model-generated (requirements §7). It is a parsing contract,
not a prompt template — no question text is ever authored here.

The live interviewer receives the plan and is explicitly allowed to
deviate from it (generate follow-ups on what the candidate just said).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Persona(BaseModel):
    """Who the interviewer is for this round and how they behave."""

    name: str = "Interviewer"
    role_title: str = "Engineer"
    demeanour: str = "probing but warm"
    opening_line: str
    system_prompt_extra: str = ""  # persona-specific guidance folded into the system prompt


class Topic(BaseModel):
    name: str
    weight: float = 1.0  # relative emphasis within the round


class PlannedQuestion(BaseModel):
    id: str
    topic: str
    prompt: str
    expected_points: list[str] = Field(default_factory=list)
    difficulty: int = 3               # 1-5
    follow_ups: list[str] = Field(default_factory=list)
    est_minutes: float = 3.0


class InterviewPlan(BaseModel):
    round_type: str
    persona: Persona
    topics: list[Topic] = Field(default_factory=list)
    questions: list[PlannedQuestion]
    criteria: list[str] = Field(default_factory=list)
    duration_min: int = 30
    critic_pass: bool = False

    def fits_budget(self) -> bool:
        """Total estimated question time must not exceed the round length."""
        return sum(q.est_minutes for q in self.questions) <= self.duration_min

    def question_at(self, cursor: int) -> PlannedQuestion | None:
        return self.questions[cursor] if 0 <= cursor < len(self.questions) else None
