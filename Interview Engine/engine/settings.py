"""
Loads config/models.yaml into typed specs, and reads infrastructure
settings (hosts, keys, DB url) from the environment.

RULE: no model name, size, device, or fallback chain appears anywhere
else in the codebase. A module that needs the interviewer model calls
`get_models().roles[Role.INTERVIEWER]` and gets an `LLMSpec` — it never
writes "qwen3:4b".

The YAML is validated here at boot: a malformed role entry raises now,
not mid-interview.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _ROOT / ".env"
_DEFAULT_MODELS_YAML = _ROOT / "config" / "models.yaml"


class Provider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"


class Role(str, Enum):
    """What a model is used FOR. Roles are stable; the models behind them are not."""

    INTERVIEWER = "interviewer"  # live turn generation — the latency-critical role
    PLANNER = "planner"          # config + resume -> InterviewPlan (pre-session)
    PARSER = "parser"            # resume text -> structured JSON (pre-session)
    SCORER = "scorer"            # transcript + rubric -> Report (post-session)
    EMBEDDER = "embedder"        # resume / JD -> vectors for retrieval


class LLMSpec(BaseModel):
    """One model binding — everything a provider needs to make the call.

    Optional fields left as None mean "provider default". `think` in
    particular must stay None unless the YAML sets it: sending `think`
    to a model with no thinking mode (phi4-mini) 400s — the bug Phase
    0.2 test 11 caught.
    """

    name: str
    provider: Provider = Provider.OLLAMA

    num_ctx: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    think: bool | None = None
    keep_alive: int | str | None = None

    # JSON-output controls — at most one applies per provider.
    json_schema: str | None = None          # name into the yaml `schemas:` block
    json_schema_def: dict[str, Any] | None = None  # resolved from json_schema by the loader
    format: str | None = None               # Ollama free-form json mode ("json")
    response_format: str | None = None      # Groq ("json_object")

    dim: int | None = None                  # embedder output width; guards the pgvector column

    # VRAM / lifecycle hints (enforced by orchestrator.vram_budget, later pass).
    max_vram_mb: int | None = None
    unload_before_session: bool = False

    fallback: list["LLMSpec"] = Field(default_factory=list)
    # tried in order after `name` fails; RoutingProvider stops at the first success.


LLMSpec.model_rebuild()


class STTSpec(BaseModel):
    engine: str
    model: str
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "int8"


class TTSSpec(BaseModel):
    engine: str
    voice: str
    lang_code: str = "a"
    emit_visemes: bool = True
    enabled: bool = True   # ENGINE_TTS_ENABLED=false → SilenceTTS (CI / no Kokoro)


class VADSpec(BaseModel):
    engine: str
    sample_rate: int = 16_000
    speech_threshold: float = 0.5
    endpoint_silent_frames: int = 8


class AvatarSpec(BaseModel):
    live: dict[str, Any]
    offline: dict[str, Any]


class LatencyBudget(BaseModel):
    """Milliseconds. Ranges are [p50, p95]; `danger_line` is a single ceiling."""

    vad_endpoint: tuple[int, int]
    stt_finalise: tuple[int, int]
    context_retrieval: tuple[int, int]
    llm_ttft: tuple[int, int]
    first_clause_tts: tuple[int, int]
    transport_playback: tuple[int, int]
    cache_miss_total: tuple[int, int]
    cache_hit_total: tuple[int, int]
    danger_line: int


def _resolve_spec(raw: dict[str, Any], schemas: dict[str, Any]) -> LLMSpec:
    d = dict(raw)
    name = d.pop("json_schema", None)
    spec = LLMSpec(**d)
    if name is not None:
        if name not in schemas:
            raise ValueError(f"json_schema {name!r} referenced but not defined in `schemas:`")
        spec.json_schema = name
        spec.json_schema_def = schemas[name]
    return spec


class ModelRegistry(BaseModel):
    """The swap point. Change a `name` in models.yaml and the engine follows."""

    roles: dict[Role, LLMSpec]
    stt: STTSpec
    tts: TTSSpec
    vad: VADSpec
    avatar: AvatarSpec
    latency_budget_ms: LatencyBudget

    @classmethod
    def load(cls, path: Path) -> "ModelRegistry":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        schemas = raw.get("schemas", {})

        roles: dict[str, LLMSpec] = {}
        for role_name, rc in raw["roles"].items():
            primary = dict(rc["primary"])
            if "provider" not in primary and rc.get("provider"):
                primary["provider"] = rc["provider"]
            primary.setdefault("max_vram_mb", rc.get("max_vram_mb"))
            primary.setdefault("unload_before_session", rc.get("unload_before_session", False))

            spec = _resolve_spec(primary, schemas)
            spec.fallback = [_resolve_spec(f, schemas) for f in rc.get("fallbacks", [])]
            roles[role_name] = spec

        return cls(
            roles=roles,
            stt=raw["stt"],
            tts=raw["tts"],
            vad=raw["vad"],
            avatar=raw["avatar"],
            latency_budget_ms=raw["latency_budget_ms"],
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_prefix="ENGINE_", extra="ignore"
    )

    ollama_host: str = "http://localhost:11434"
    groq_api_key: str | None = None
    database_url: str = "postgresql+asyncpg://postgres:dev@localhost:5433/interview"
    media_root: str = "./media"

    barge_in_enabled: bool = False
    tts_enabled: bool = True   # env override; SilenceTTS when false
    models_config_path: Path = _DEFAULT_MODELS_YAML

    @property
    def models(self) -> ModelRegistry:
        return get_models()

    def llm(self, role: Role) -> LLMSpec:
        return get_models().roles[role]


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_models() -> ModelRegistry:
    return ModelRegistry.load(get_settings().models_config_path)
