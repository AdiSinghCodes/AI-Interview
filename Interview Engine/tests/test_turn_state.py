import pytest

from engine.orchestrator.turn_state import (
    IllegalTransition,
    TurnState,
    advance,
    can_advance,
)


def test_full_legal_turn_cycle():
    s = TurnState.IDLE
    for target in (
        TurnState.SPEAKING,   # opening line
        TurnState.LISTENING,  # wait for candidate
        TurnState.THINKING,   # endpoint fired
        TurnState.SPEAKING,   # reply ready
        TurnState.LISTENING,  # playback done
    ):
        s = advance(s, target)
    assert s == TurnState.LISTENING


def test_barge_in_transition_is_legal():
    # VAD fires while SPEAKING -> back to LISTENING
    assert can_advance(TurnState.SPEAKING, TurnState.LISTENING)


@pytest.mark.parametrize(
    "current,target",
    [
        (TurnState.IDLE, TurnState.THINKING),
        (TurnState.IDLE, TurnState.LISTENING),
        (TurnState.LISTENING, TurnState.SPEAKING),
        (TurnState.THINKING, TurnState.LISTENING),
        (TurnState.ENDED, TurnState.LISTENING),
    ],
)
def test_illegal_transitions_raise(current, target):
    with pytest.raises(IllegalTransition):
        advance(current, target)


def test_any_state_can_end():
    for s in (TurnState.IDLE, TurnState.LISTENING, TurnState.THINKING, TurnState.SPEAKING):
        assert can_advance(s, TurnState.ENDED)
