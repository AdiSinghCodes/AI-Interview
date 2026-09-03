from engine.policy.pacing import Pacing

LONG = "word " * 30
SHORT = "yes it does"


def test_no_followup_on_the_first_answer():
    p = Pacing(num_planned=10)
    assert p.should_follow_up(LONG, planned_cursor=0) is False


def test_followup_only_after_a_substantive_answer():
    p = Pacing(num_planned=10)
    assert p.should_follow_up(SHORT, planned_cursor=2) is False
    assert p.should_follow_up(LONG, planned_cursor=2) is True


def test_never_two_followups_in_a_row():
    p = Pacing(num_planned=10)
    assert p.should_follow_up(LONG, planned_cursor=2) is True
    p.record_turn(was_followup=True)
    assert p.should_follow_up(LONG, planned_cursor=2) is False


def test_followups_are_capped():
    p = Pacing(num_planned=20, max_followups=2)
    used = 0
    for cursor in range(2, 20):
        if p.should_follow_up(LONG, planned_cursor=cursor):
            used += 1
            p.record_turn(was_followup=True)
        else:
            p.record_turn(was_followup=False)
    assert used == 2


def test_is_complete_when_all_planned_asked():
    p = Pacing(num_planned=5)
    assert p.is_complete(planned_cursor=4, total_turns=4) is False
    assert p.is_complete(planned_cursor=5, total_turns=6) is True


def test_is_complete_hard_cap_on_runaway_turns():
    p = Pacing(num_planned=5)
    assert p.is_complete(planned_cursor=2, total_turns=14) is True
