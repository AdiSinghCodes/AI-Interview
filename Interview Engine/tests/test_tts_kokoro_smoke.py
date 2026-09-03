import numpy as np
import pytest


async def test_silence_tts_is_sized_to_the_clause():
    """Fast — no model. SilenceTTS keeps the loop runnable without Kokoro."""
    from engine.speech.tts_kokoro import SilenceTTS

    tts = SilenceTTS()
    chunks = [c async for c in tts.synthesize("one two three four five")]
    audio = np.concatenate(chunks)
    assert audio.dtype == np.float32
    assert audio.size == pytest.approx(tts.sample_rate * 5 * 0.34, rel=0.01)


@pytest.mark.slow
async def test_kokoro_synthesizes_real_audio():
    pytest.importorskip("kokoro")
    from engine.speech.tts_kokoro import KokoroTTS

    tts = KokoroTTS()
    try:
        chunks = [c async for c in tts.synthesize("Thanks for joining, let's begin.")]
        audio = np.concatenate(chunks)
        assert audio.dtype == np.float32
        assert audio.size > tts.sample_rate * 0.5   # at least half a second of speech
    finally:
        tts.close()
