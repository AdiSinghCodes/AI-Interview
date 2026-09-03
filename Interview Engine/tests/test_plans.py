import pytest

from engine.plans import DEFAULT_ROLE, get_plan, role_ids


def test_all_roles_have_a_plan():
    assert set(role_ids()) == {"technical", "python", "react", "hr", "dsa"}


@pytest.mark.parametrize("role", role_ids())
def test_each_plan_is_well_formed(role):
    plan = get_plan(role)
    assert plan.persona.opening_line.strip()
    assert len(plan.questions) >= 8
    assert all(q.id and q.prompt for q in plan.questions)
    assert plan.criteria
    assert plan.fits_budget()


def test_get_plan_trims_to_num_questions_and_deep_copies():
    a = get_plan("python", num_questions=4)
    assert len(a.questions) == 4
    a.questions[0].prompt = "MUTATED"
    b = get_plan("python", num_questions=4)
    assert b.questions[0].prompt != "MUTATED"


def test_unknown_role_falls_back_to_default():
    assert get_plan("astronaut").round_type == get_plan(DEFAULT_ROLE).round_type
