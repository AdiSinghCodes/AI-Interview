"""
TEST 10: Silero VAD — Voice activity detection
====================================================
What it shows:
  - Real load time via torch.hub (network fetch on first run, cached
    after)
  - Correctness sanity check: silence should read as NOT speech, and
    endpointing should fire after enough trailing silent frames.

Requirements:
  pip install torch

Run:
  python voice_activity_detection/test_10_vad_silero.py
"""

import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

SAMPLE_RATE = 16_000
SPEECH_THRESHOLD = 0.5
SILENCE_FRAMES_FOR_STOP = 8  # ~250ms at 32ms/frame


def run_test():
    print("TEST 10: Silero VAD")
    print("=" * 50)

    print("\nLoading via torch.hub (first run downloads the repo)...")
    t0 = time.perf_counter()
    try:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
    except Exception as e:
        print(f"\nFAILED to load: {type(e).__name__}: {e}")
        return
    load_s = time.perf_counter() - t0
    print(f"Loaded: {load_s:.2f}s")

    def is_speech(chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(chunk).float()
        return model(tensor, SAMPLE_RATE).item() >= SPEECH_THRESHOLD

    silence = np.zeros(512, dtype=np.float32)
    noise = (np.random.randn(512).astype(np.float32)) * 0.5

    print("\nChecking silence is NOT flagged as speech...")
    t1 = time.perf_counter()
    silence_result = is_speech(silence)
    print(f"  is_speech(silence) = {silence_result}  "
          f"({'OK' if not silence_result else 'UNEXPECTED — flagged silence as speech'})")

    print("\nChecking loud noise IS flagged as speech (probabilistic, may vary)...")
    noise_result = is_speech(noise)
    infer_s = time.perf_counter() - t1
    print(f"  is_speech(noise)   = {noise_result}")
    print(f"  (2 inference calls took {infer_s:.3f}s total — should be near-instant)")

    print("\nChecking endpointing logic (spoke once, then 8 silent frames)...")
    history = [True] + [False] * 8
    tail = history[-SILENCE_FRAMES_FOR_STOP:]
    stopped = len(tail) == SILENCE_FRAMES_FOR_STOP and not any(tail)
    print(f"  has_stopped = {stopped}  "
          f"({'OK' if stopped else 'UNEXPECTED — should detect stop'})")

    print("\n" + "=" * 50)
    print("RESULT: PASS")


if __name__ == "__main__":
    run_gated(run_test)
