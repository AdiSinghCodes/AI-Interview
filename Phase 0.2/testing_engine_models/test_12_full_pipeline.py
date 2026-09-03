"""
TEST 12: Full pipeline — runs every model test and summarizes
====================================================
What it shows:
  Runs test_01 through test_11 each as their own subprocess (so a
  hang or crash in one doesn't take down the rest), and prints a
  single summary table: which models loaded, which are slow, which
  are broken. Skips each script's "Press Enter to start" prompt via
  MODEL_TEST_AUTO=1, so this runs unattended.

Run:
  python test_12_full_pipeline.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

TESTS = [
    ("qwen3:4b (interviewer, primary)", "interviewer/test_01_ollama_interviewer_primary.py"),
    ("phi4-mini (interviewer, fallback 1)", "interviewer/test_02_ollama_interviewer_fallback1.py"),
    ("llama3.2:3b (interviewer, fallback 2)", "interviewer/test_03_ollama_interviewer_fallback2.py"),
    ("qwen3:8b (planner)", "planner/test_04_ollama_planner.py"),
    ("llama-3.1-8b-instant (parser, Groq)", "parser/test_05_groq_parser.py"),
    ("gpt-oss-20b (scorer, Groq)", "scorer/test_06_groq_scorer.py"),
    ("nomic-embed-text (embedder)", "embedder/test_07_ollama_embedder.py"),
    ("faster-whisper (speech-to-text)", "speech_to_text/test_08_stt_whisper.py"),
    ("Kokoro (text-to-speech)", "text_to_speech/test_09_tts_kokoro.py"),
    ("Silero VAD (voice activity)", "voice_activity_detection/test_10_vad_silero.py"),
    ("Interviewer fallback chain (end to end)", "interviewer/test_11_interviewer_fallback_chain.py"),
    ("MuseTalk (avatar lip-sync)", "avatar/test_19_avatar_musetalk.py"),
    ("Wav2Lip (avatar lip-sync)", "avatar/test_20_avatar_wav2lip.py"),
    ("LivePortrait (avatar, video-driven)", "avatar/test_21_avatar_liveportrait.py"),
    ("SadTalker (avatar lip-sync)", "avatar/test_22_avatar_sadtalker.py"),
    ("InfiniteTalk (avatar, 14B backbone, low-VRAM)", "avatar/test_23_avatar_infinitetalk.py"),
]

TEST_DIR = Path(__file__).parent


def main():
    print("TEST 12: Full model pipeline check")
    print("=" * 60)
    print(f"Running {len(TESTS)} tests, each in its own process...\n")

    env = {**os.environ, "MODEL_TEST_AUTO": "1"}
    results = []
    for label, filename in TESTS:
        print(f"--- {label} ---")
        t0 = time.perf_counter()
        try:
            if filename.endswith("test_23_avatar_infinitetalk.py"):
                timeout = 2000  # 14B backbone, heaviest of the avatar tests
            elif filename.startswith("avatar/"):
                timeout = 1200
            else:
                timeout = 300
            proc = subprocess.run(
                [sys.executable, str(TEST_DIR / filename)],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(TEST_DIR), env=env,
            )
            elapsed = time.perf_counter() - t0
            output = proc.stdout + proc.stderr
            passed = "RESULT: PASS" in output
            status = "PASS" if passed else "FAIL"
            if not passed:
                tail = "\n".join(output.strip().splitlines()[-5:])
                print(tail)
        except subprocess.TimeoutExpired:
            elapsed = timeout
            status = f"TIMEOUT ({timeout}s)"
        results.append((label, status, elapsed))
        print(f"[{status}] {elapsed:.1f}s\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<42} {'Status':<15} {'Time'}")
    print("-" * 60)
    for label, status, elapsed in results:
        print(f"{label:<42} {status:<15} {elapsed:.1f}s")

    failed = [r for r in results if r[1] != "PASS"]
    print("\n" + ("All models OK." if not failed else
                   f"{len(failed)} model(s) need attention: " +
                   ", ".join(r[0] for r in failed)))


if __name__ == "__main__":
    main()
