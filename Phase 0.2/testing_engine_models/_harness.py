"""
Shared helpers for the test_NN_*.py scripts in this folder. Not a test
itself.

- run_gated / run_gated_async: waits for Enter before starting (skip
  with MODEL_TEST_AUTO=1, which test_12_full_pipeline.py sets on its
  subprocesses so the full suite still runs unattended), then prints
  wall-clock start/end times around the test body.
- load_groq_key: reads ENGINE_GROQ_API_KEY straight out of the
  interview-engine project's .env file — these scripts live in a
  separate project folder and don't have that package installed.
"""

import os
import re
from datetime import datetime
from pathlib import Path

ENGINE_ENV_FILE = Path(
    r"C:\Users\GHANSHYAM\Desktop\newai-interviewer\interview-engine\.env"
)


def load_groq_key() -> str | None:
    if not ENGINE_ENV_FILE.exists():
        return None
    text = ENGINE_ENV_FILE.read_text(encoding="utf-8")
    m = re.search(r"^ENGINE_GROQ_API_KEY=(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _wait_for_start():
    if not os.environ.get("MODEL_TEST_AUTO"):
        input("Press Enter to start the test... ")
    start = datetime.now()
    print(f"Start time: {start:%Y-%m-%d %H:%M:%S}\n")
    return start


def _report_end(start):
    end = datetime.now()
    print(f"\nEnd time:   {end:%Y-%m-%d %H:%M:%S}")
    print(f"Duration:   {(end - start).total_seconds():.2f}s")


def run_gated(run_fn):
    start = _wait_for_start()
    try:
        run_fn()
    finally:
        _report_end(start)


async def run_gated_async(run_fn):
    start = _wait_for_start()
    try:
        await run_fn()
    finally:
        _report_end(start)
