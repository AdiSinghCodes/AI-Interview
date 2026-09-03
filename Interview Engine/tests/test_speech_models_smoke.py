"""Real model loads. Slow (weights + ctranslate2 import). Skipped by
default; run with `pytest -m slow`."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_silero_loads_and_silence_is_not_speech():
    from engine.speech.vad_silero import SileroVAD

    vad = SileroVAD()
    assert vad.is_speech(np.zeros(512, dtype=np.float32)) is False


def test_whisper_loads_and_silence_transcribes_empty():
    from engine.speech.stt_whisper import WhisperSTT

    stt = WhisperSTT()
    assert stt.transcribe(np.zeros(16_000, dtype=np.float32)) == ""


def test_whisper_returns_a_string_on_nonsilent_input():
    from engine.speech.stt_whisper import WhisperSTT

    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(16_000).astype(np.float32)) * 0.2
    assert isinstance(WhisperSTT().transcribe(clip), str)
