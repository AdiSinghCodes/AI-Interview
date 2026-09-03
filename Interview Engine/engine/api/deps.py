"""
FastAPI dependency wiring. The ONE place providers / heavy models (and
later the repository + tenant context) are constructed — routes and the
WS handler depend on these, never build their own.

The STT/VAD singletons pull real weights (measured: ~10 s ctranslate2
import for whisper, ~1 s torch-hub for silero). They load lazily on first
use so `pytest` and the text-only turn loop never pay that cost, and tests
override these via `app.dependency_overrides`.
"""

from __future__ import annotations

from engine.inference.routing_provider import RoutingProvider, build_router
from engine.settings import get_models, get_settings

_provider: RoutingProvider | None = None
_vad = None
_stt = None
_tts = None


def get_inference_provider() -> RoutingProvider:
    global _provider
    if _provider is None:
        _provider = build_router()
    return _provider


def get_vad():
    global _vad
    if _vad is None:
        from engine.speech.vad_silero import SileroVAD

        _vad = SileroVAD()
    return _vad


def get_stt():
    global _stt
    if _stt is None:
        from engine.speech.stt_whisper import WhisperSTT

        _stt = WhisperSTT()
    return _stt


def get_tts():
    global _tts
    if _tts is None:
        from engine.speech.tts_kokoro import KokoroTTS, SilenceTTS

        on = get_settings().tts_enabled and get_models().tts.enabled
        _tts = KokoroTTS() if on else SilenceTTS()
    return _tts
