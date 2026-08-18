"""
TEST 8: faster-whisper — Speech-to-text (real microphone input)
====================================================
What it shows:
  - Real load time for the model+device the interview-engine uses
    (distil-small.en, CPU)
  - Whether it actually transcribes YOUR voice correctly — records a
    few seconds from your microphone and prints back what it heard,
    so you can visually confirm accuracy instead of just "no crash".
  - KNOWN PAST BUGS on this box:
    1. config previously named "nvidia/parakeet-tdt-0.6b-v3" (a NeMo
       model), but faster-whisper can't load Parakeet at all — it hung
       indefinitely trying to resolve it as a HuggingFace repo.
    2. GPU mode (device="cuda") crashes with "Library cublas64_12.dll
       is not found or cannot be loaded" — the CUDA cuBLAS runtime
       ctranslate2 needs isn't available on this machine. CPU is used
       instead, which also frees the GPU entirely for the LLM.

Interactive: press Enter to record a few seconds of speech, see the
transcription and timing, repeat. Type "exit"/"quit" instead of Enter
to end the session.

Requirements:
  pip install faster-whisper sounddevice numpy

Run:
  python speech_to_text/test_08_stt_whisper.py
"""

import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

MODEL_NAME = "distil-small.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16_000   # faster-whisper expects 16kHz mono
RECORD_SECONDS = 5


def record_audio(seconds: float) -> np.ndarray:
    """Records from the default microphone. Returns float32 mono audio
    at 16kHz, the format faster-whisper wants directly (no wav needed)."""
    import sounddevice as sd

    print(f"Recording for {seconds:.0f}s — speak now...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording done.")
    return audio.flatten()


def load_model():
    print(f"\nImporting faster_whisper / ctranslate2 (measured ~8-17s cold)...")
    t0 = time.perf_counter()
    from faster_whisper import WhisperModel
    import_s = time.perf_counter() - t0
    print(f"Import done: {import_s:.2f}s")

    print("\nLoading model (first-ever run downloads weights — can be 60s+)...")
    t1 = time.perf_counter()
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_s = time.perf_counter() - t1
    print(f"Model loaded: {load_s:.2f}s")

    # Confirm it actually loaded where we asked, not a silent CPU
    # fallback from a requested GPU — relevant given this box's known
    # cuBLAS failure history.
    actual_device = getattr(model.model, "device", DEVICE)
    if str(actual_device) != DEVICE:
        print(f"WARNING: requested device={DEVICE!r} but model reports "
              f"running on {actual_device!r}.")

    return model


def run_test():
    print(f"TEST 8: faster-whisper — model={MODEL_NAME!r} device={DEVICE} "
          f"compute_type={COMPUTE_TYPE}")
    print("=" * 50)

    try:
        import sounddevice  # noqa: F401
    except ImportError:
        print("FAILED: sounddevice not installed.")
        print("  -> pip install sounddevice")
        return

    try:
        model = load_model()
    except Exception as e:
        print(f"\nFAILED to load: {type(e).__name__}: {e}")
        return

    print(f"\nEach round records {RECORD_SECONDS:.0f}s of audio from your "
          "default mic and transcribes it.")
    print("Press Enter to record, or type 'exit'/'quit' to stop.")

    round_num = 0
    while True:
        try:
            cmd = input("\nPress Enter to record: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if cmd in {"exit", "quit"}:
            print("Bye.")
            break

        round_num += 1
        try:
            audio = record_audio(RECORD_SECONDS)
        except Exception as e:
            print(f"FAILED to record: {type(e).__name__}: {e}")
            print("  -> check your default input device / mic permissions")
            continue

        # Skip transcribing near-silence so you don't mistake "nothing
        # heard" for "STT is broken" — most likely just no mic input.
        peak = np.abs(audio).max()
        if peak < 0.005:
            print(f"WARNING: peak amplitude {peak:.4f} is very low — "
                  "check your mic is unmuted and picking up sound.")

        print("Transcribing...")
        t0 = time.perf_counter()
        try:
            segments, info = model.transcribe(
                audio, language="en", vad_filter=True, beam_size=5,
            )
            segments = list(segments)  # materialize to get real timing
        except Exception as e:
            print(f"FAILED to transcribe: {type(e).__name__}: {e}")
            continue
        transcribe_s = time.perf_counter() - t0

        text = " ".join(seg.text.strip() for seg in segments)
        first_word_s = segments[0].start if segments else None

        print(f"\nRound {round_num} — heard: {text!r}")
        if not text:
            print("  (empty — likely silence, or speech too quiet/short)")
        print(f"Detected language: {info.language} "
              f"(confidence {info.language_probability:.2f})")
        print(f"Transcribe time:   {transcribe_s:.2f}s for "
              f"{RECORD_SECONDS:.0f}s of audio "
              f"({RECORD_SECONDS / transcribe_s:.1f}x realtime)")
        if first_word_s is not None:
            print(f"First speech detected at: {first_word_s:.2f}s into clip")


if __name__ == "__main__":
    run_gated(run_test)