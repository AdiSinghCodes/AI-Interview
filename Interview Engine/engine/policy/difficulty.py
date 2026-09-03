"""
Difficulty calibration. Small models judge RELATIVE difficulty well and
ABSOLUTE difficulty poorly (requirements §7) — so the planner gets 2-3
example questions per stage as calibration anchors, and this module owns
that mapping plus any in-round "they're cruising, go harder" nudge.

STUB — build-order pass 7.
"""

from __future__ import annotations

from engine.schemas.interview_config import RoundType, Stage


def stage_anchors_for(stage: Stage, round_type: RoundType) -> list[str]:
    raise NotImplementedError("difficulty anchors land in build-order pass 7")
