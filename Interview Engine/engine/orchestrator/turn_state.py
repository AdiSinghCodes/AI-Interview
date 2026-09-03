"""
The turn state machine, as a value type. `SessionOrchestrator` holds one
`TurnState` and moves it only through `advance()`, which rejects illegal
transitions loudly — a turn loop that silently ends up in the wrong state
is the hardest class of bug to find later.

    IDLE ──start──> SPEAKING (opening line)
    SPEAKING ──playback done──> LISTENING
    LISTENING ──candidate endpoint──> THINKING
    THINKING ──first audio/text ready──> SPEAKING
    SPEAKING ──VAD fires mid-playback──> LISTENING     (barge-in, pass 4)
    LISTENING/THINKING/SPEAKING ──end round──> ENDED
"""

from __future__ import annotations

from enum import Enum


class TurnState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ENDED = "ended"


class IllegalTransition(RuntimeError):
    pass


_ALLOWED: dict[TurnState, set[TurnState]] = {
    TurnState.IDLE: {TurnState.SPEAKING, TurnState.ENDED},
    TurnState.SPEAKING: {TurnState.LISTENING, TurnState.ENDED},
    TurnState.LISTENING: {TurnState.THINKING, TurnState.ENDED},
    TurnState.THINKING: {TurnState.SPEAKING, TurnState.ENDED},
    TurnState.ENDED: set(),
}


def advance(current: TurnState, target: TurnState) -> TurnState:
    if target not in _ALLOWED[current]:
        raise IllegalTransition(f"{current.value} -> {target.value} is not allowed")
    return target


def can_advance(current: TurnState, target: TurnState) -> bool:
    return target in _ALLOWED[current]
