"""
The turn loop's spine. `SessionOrchestrator` owns turn state, the
transcript, the plan cursor and the degrade ladder. Everything else —
transport, speech, avatar — plugs into it without changing its shape.
"""

from .session_orchestrator import SessionOrchestrator, TurnResult
from .turn_state import IllegalTransition, TurnState

__all__ = ["SessionOrchestrator", "TurnResult", "TurnState", "IllegalTransition"]
