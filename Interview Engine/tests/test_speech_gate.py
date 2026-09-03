from engine.orchestrator.speech_gate import SpeechGate


def test_accepts_before_any_speech():
    assert SpeechGate().accepting_audio(now=0.0) is True


def test_blocks_while_speaking():
    g = SpeechGate(tail_ms=150)
    g.enter_speaking()
    assert g.accepting_audio(now=1000.0) is False


def test_tail_after_speaking_ends():
    g = SpeechGate(tail_ms=150)
    g.enter_speaking()
    g.leave_speaking(now=1000.0)
    assert g.accepting_audio(now=1000.10) is False   # 100 ms < 150 ms tail
    assert g.accepting_audio(now=1000.20) is True     # 200 ms > 150 ms tail


def test_re_entering_speaking_blocks_again():
    g = SpeechGate(tail_ms=50)
    g.enter_speaking()
    g.leave_speaking(now=10.0)
    assert g.accepting_audio(now=10.1) is True
    g.enter_speaking()
    assert g.accepting_audio(now=10.2) is False
