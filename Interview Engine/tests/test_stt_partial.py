"""Incremental STT finalize logic — no model, `_run` stubbed."""

import time

import numpy as np

from engine.speech.stt_whisper import WhisperSTT


def _stt(monkeypatch, run_result="full transcription"):
    s = WhisperSTT()
    calls = []

    def fake_run(audio):
        calls.append(audio.size)
        return run_result

    monkeypatch.setattr(s, "_run", fake_run)
    return s, calls


def test_finalize_reuses_a_fresh_well_covered_partial(monkeypatch):
    s, calls = _stt(monkeypatch)
    s.begin_utterance()
    s.feed_partial(np.ones(10_000, dtype=np.float32))   # partial covers 10k samples
    calls.clear()

    out = s.finalize(np.ones(10_500, dtype=np.float32))  # endpoint audio ~= partial
    assert out == "full transcription"
    assert calls == []                                   # no extra transcription pass


def test_finalize_reruns_when_partial_is_stale(monkeypatch):
    s, calls = _stt(monkeypatch)
    s.begin_utterance()
    s.feed_partial(np.ones(10_000, dtype=np.float32))
    s._partial_at = time.perf_counter() - 5.0            # pretend it's old
    calls.clear()

    s.finalize(np.ones(10_500, dtype=np.float32))
    assert calls == [10_500]                             # had to re-run on the full audio


def test_finalize_reruns_when_a_lot_more_was_said_after_the_partial(monkeypatch):
    s, calls = _stt(monkeypatch)
    s.begin_utterance()
    s.feed_partial(np.ones(5_000, dtype=np.float32))
    calls.clear()

    s.finalize(np.ones(20_000, dtype=np.float32))        # 4x more audio than the partial
    assert calls == [20_000]


def test_finalize_resets_partial_state(monkeypatch):
    s, _ = _stt(monkeypatch)
    s.begin_utterance()
    s.feed_partial(np.ones(10_000, dtype=np.float32))
    s.finalize(np.ones(10_000, dtype=np.float32))
    assert s._partial_text == ""
