"""
TEST 9: Kokoro — Text-to-speech (real playback)
====================================================
What it shows:
  - Real load + synthesis time for Kokoro TTS
  - Realtime factor: audio produced vs. wall-clock time taken, which
    is what determines whether TTS can keep up with a live turn
  - Plays the synthesized audio back through your speakers and saves
    it to a WAV file, so you can actually judge voice quality and
    pronunciation instead of just trusting a chunk/sample count.

Interactive: type a sentence, hear it synthesized and played back,
with timing. Empty line or "exit"/"quit" ends the session.

Requirements:
  pip install kokoro soundfile sounddevice

Run:
   python testing_engine_models/text_to_speech/test_09_tts_kokoro.py
"""

import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

VOICE = "af_heart"
SAMPLE_RATE = 24_000  # Kokoro's native output rate
OUTPUT_DIR = Path(__file__).resolve().parent / "tts_output"


def load_pipeline():
    print("\nImporting kokoro + loading pipeline...")
    t0 = time.perf_counter()
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")  # American English
    load_s = time.perf_counter() - t0
    print(f"Loaded: {load_s:.2f}s")
    return pipeline


def synthesize(pipeline, text: str) -> tuple[np.ndarray, float]:
    """Runs the pipeline and concatenates all chunks into one waveform.
    Returns (audio, synth_seconds)."""
    t0 = time.perf_counter()
    pieces = []
    for _, _, audio in pipeline(text, voice=VOICE):
        pieces.append(audio)
    synth_s = time.perf_counter() - t0
    full_audio = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
    return full_audio, synth_s


def run_test():
    print("TEST 9: Kokoro TTS")
    print("=" * 50)

    missing = []
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    if missing:
        print(f"FAILED: missing packages: {', '.join(missing)}")
        print(f"  -> pip install {' '.join(missing)}")
        return

    import soundfile as sf
    import sounddevice as sd

    try:
        pipeline = load_pipeline()
    except Exception as e:
        print(f"\nFAILED to load: {type(e).__name__}: {e}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    default_sentence = "Thanks for walking me through your experience with FastAPI."
    print(f"\nType a sentence to synthesize, or press Enter to use the "
          f"default:\n  {default_sentence!r}")
    print("Type 'exit'/'quit' to stop.")

    round_num = 0
    while True:
        try:
            text = input("\nText: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if text.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not text:
            text = default_sentence

        round_num += 1
        print(f"\nSynthesizing: {text!r}")
        try:
            audio, synth_s = synthesize(pipeline, text)
        except Exception as e:
            print(f"FAILED to synthesize: {type(e).__name__}: {e}")
            continue

        if len(audio) == 0:
            print("WARNING: no audio produced — empty pipeline output.")
            continue

        audio_seconds = len(audio) / SAMPLE_RATE
        rtf = audio_seconds / synth_s if synth_s else 0

        out_path = OUTPUT_DIR / f"round_{round_num}.wav"
        sf.write(out_path, audio, SAMPLE_RATE)

        print(f"Synthesis time:    {synth_s:.2f}s")
        print(f"Audio produced:    {audio_seconds:.2f}s of speech")
        print(f"Realtime factor:   {rtf:.2f}x "
              f"({'faster than realtime — good' if rtf > 1 else 'SLOWER than realtime — will lag live turns'})")
        print(f"Saved to:          {out_path}")

        print("Playing back...")
        try:
            sd.play(audio, SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            print(f"WARNING: playback failed ({type(e).__name__}: {e}) — "
                  f"but the WAV file above is still saved, open it manually.")

    print("\n" + "=" * 50)
    print("RESULT: PASS")


if __name__ == "__main__":
    run_gated(run_test)