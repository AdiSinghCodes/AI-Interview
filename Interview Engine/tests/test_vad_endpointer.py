"""StreamingEndpointer logic — driven with a fake speech test, no torch."""

import numpy as np

from engine.speech.vad_silero import StreamingEndpointer

SILENCE = np.zeros(512, dtype=np.float32)
SPEECH = np.ones(512, dtype=np.float32)


def _is_speech(frame: np.ndarray) -> bool:
    return bool(np.abs(frame).max() > 0.1)


def test_fires_after_exactly_n_silent_frames_following_speech():
    ep = StreamingEndpointer(_is_speech, silent_frames_for_stop=8, min_speech_frames=3)
    for _ in range(5):
        assert ep.feed(SPEECH) is False
    results = [ep.feed(SILENCE) for _ in range(8)]
    assert results == [False] * 7 + [True]


def test_never_fires_on_silence_only():
    ep = StreamingEndpointer(_is_speech, silent_frames_for_stop=4)
    assert not any(ep.feed(SILENCE) for _ in range(50))


def test_short_blip_below_min_speech_never_arms():
    ep = StreamingEndpointer(_is_speech, silent_frames_for_stop=3, min_speech_frames=3)
    ep.feed(SPEECH)
    ep.feed(SPEECH)  # only 2 — not a real utterance
    assert not any(ep.feed(SILENCE) for _ in range(10))


def test_fires_once_then_needs_reset():
    ep = StreamingEndpointer(_is_speech, silent_frames_for_stop=2, min_speech_frames=1)
    ep.feed(SPEECH)
    ep.feed(SILENCE)
    assert ep.feed(SILENCE) is True
    assert ep.feed(SILENCE) is False
    ep.reset()
    ep.feed(SPEECH)
    ep.feed(SILENCE)
    assert ep.feed(SILENCE) is True


def test_new_speech_resets_the_trailing_silence_run():
    ep = StreamingEndpointer(_is_speech, silent_frames_for_stop=4, min_speech_frames=1)
    ep.feed(SPEECH)
    for _ in range(3):
        ep.feed(SILENCE)          # 3 silent, not yet 4
    ep.feed(SPEECH)               # resets the run
    assert not any(ep.feed(SILENCE) for _ in range(3))
    assert ep.feed(SILENCE) is True
