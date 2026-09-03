"""config/models.yaml is the single source of truth — these assert it
loads into the shape the rest of the engine assumes."""

from engine.settings import Provider, Role, get_models


def test_all_roles_present():
    roles = get_models().roles
    assert set(roles) == {
        Role.INTERVIEWER,
        Role.PLANNER,
        Role.PARSER,
        Role.SCORER,
        Role.EMBEDDER,
    }


def test_interviewer_primary_and_fallback_chain():
    spec = get_models().roles[Role.INTERVIEWER]
    assert spec.name == "qwen3:4b"
    assert spec.provider == Provider.OLLAMA
    assert spec.think is False          # explicitly set — suppresses the <think> block
    assert spec.keep_alive == -1
    assert spec.json_schema_def is not None
    assert spec.json_schema_def["required"] == ["answer"]
    assert [f.name for f in spec.fallback] == ["phi4-mini", "llama3.2:3b"]


def test_fallback_models_never_carry_a_think_flag():
    # phi4-mini / llama3.2:3b 400 on an unexpected `think` key (Phase 0.2 test 11).
    for fb in get_models().roles[Role.INTERVIEWER].fallback:
        assert fb.think is None


def test_parser_and_scorer_are_groq_first_local_fallback():
    for role in (Role.PARSER, Role.SCORER):
        spec = get_models().roles[role]
        assert spec.provider == Provider.GROQ
        assert spec.fallback, f"{role} needs a local fallback"
        assert spec.fallback[0].provider == Provider.OLLAMA
        assert spec.fallback[0].name == "qwen3:8b"


def test_embedder_dim_guard():
    assert get_models().roles[Role.EMBEDDER].dim == 768


def test_latency_danger_line_is_1500ms():
    assert get_models().latency_budget_ms.danger_line == 1500


def test_tts_config():
    tts = get_models().tts
    assert tts.engine == "kokoro"
    assert tts.voice == "af_heart"
    assert tts.enabled is True
