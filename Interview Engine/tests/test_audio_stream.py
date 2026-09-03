import numpy as np

from engine.speech.audio_stream import FRAME_SAMPLES, AudioStream


def _pcm(samples) -> bytes:
    return np.asarray(samples, dtype="<f4").tobytes()


def test_frames_emitted_in_512_sample_chunks():
    s = AudioStream()
    frames = s.push(_pcm(np.ones(FRAME_SAMPLES * 2 + 100, dtype=np.float32)))
    assert len(frames) == 2
    assert all(f.shape == (FRAME_SAMPLES,) for f in frames)


def test_sample_remainder_buffered_across_pushes():
    s = AudioStream()
    assert s.push(_pcm(np.ones(300, dtype=np.float32))) == []
    frames = s.push(_pcm(np.ones(300, dtype=np.float32)))  # 600 total
    assert len(frames) == 1  # one 512 frame, 88 samples carried


def test_odd_byte_length_carried():
    s = AudioStream()
    raw = _pcm(np.ones(1024, dtype=np.float32))  # 4096 bytes
    assert s.push(raw[:2047]) == []              # 511 samples + 3 dangling bytes
    frames = s.push(raw[2047:])
    assert len(frames) == 2


def test_take_utterance_accumulates_then_clears():
    s = AudioStream()
    s.push(_pcm(np.ones(FRAME_SAMPLES, dtype=np.float32)))
    s.push(_pcm(np.ones(FRAME_SAMPLES, dtype=np.float32)))
    utt = s.take_utterance()
    assert utt.shape[0] == FRAME_SAMPLES * 2
    assert s.take_utterance().shape[0] == 0


def test_reset_drops_everything():
    s = AudioStream()
    s.push(_pcm(np.ones(700, dtype=np.float32)))
    s.reset()
    assert s.take_utterance().shape[0] == 0
