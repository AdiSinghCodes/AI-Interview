"""
Curated interview plans, one per role. Hand-written and panel-safe —
these stand in for the generative planner (build-order pass 7) so the
product has reliable, calibrated content now.

    from engine.plans import get_plan
    plan = get_plan("python", num_questions=10)

Role ids match the frontend Setup screen. `get_plan` returns a deep copy
trimmed to `num_questions`, so callers can mutate freely.
"""

from __future__ import annotations

from engine.schemas.interview_plan import InterviewPlan

from .dsa import PLAN as _DSA
from .hr import PLAN as _HR
from .python_backend import PLAN as _PYTHON
from .react_frontend import PLAN as _REACT
from .technical import PLAN as _TECHNICAL

_REGISTRY: dict[str, InterviewPlan] = {
    "technical": _TECHNICAL,
    "python": _PYTHON,
    "react": _REACT,
    "hr": _HR,
    "dsa": _DSA,
}

DEFAULT_ROLE = "technical"


def role_ids() -> list[str]:
    return list(_REGISTRY)


def get_plan(role: str, *, num_questions: int | None = None, duration_min: int = 30) -> InterviewPlan:
    base = _REGISTRY.get(role, _REGISTRY[DEFAULT_ROLE])
    plan = base.model_copy(deep=True)
    plan.duration_min = duration_min
    if num_questions is not None:
        plan.questions = plan.questions[: max(1, num_questions)]
    return plan
