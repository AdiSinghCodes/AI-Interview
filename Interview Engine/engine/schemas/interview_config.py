"""
InterviewConfig — collected on the setup screen, before anything loads.
Everything downstream (planner persona, rubric, pacing, proctor profile)
reads from this. See the NTRVSTA v1 requirements §6.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Stage(str, Enum):
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"


class RoundType(str, Enum):
    SCREENING = "screening"
    TECHNICAL = "technical"
    DSA = "dsa"
    SYSTEM_DESIGN = "system_design"
    MANAGERIAL = "managerial"
    HR = "hr"


class CompanyType(str, Enum):
    SERVICE = "service"
    PRODUCT = "product"
    STARTUP = "startup"
    MNC = "mnc"


class InterviewConfig(BaseModel):
    role: str
    stage: Stage
    round_type: RoundType
    company_type: CompanyType
    company_name: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    job_description: str | None = None
    duration_min: int = 30
    difficulty_override: int | None = None  # 1-5; None = derive from stage

    @property
    def effective_stage(self) -> Stage:
        return self.stage
