"""
TEST 6: openai/gpt-oss-20b (Groq) — Scorer role
====================================================
What it shows:
  - Whether the Groq API key (read from interview-engine's .env) is
    valid
  - Real latency for a scoring-shaped call (transcript + rubric -> JSON)
  - This role is off the live-latency path (scoring happens after the
    interview ends), so this is mainly a "does it still work" check.

Requirements:
  ENGINE_GROQ_API_KEY set in newai-interviewer/interview-engine/.env

Run:
  python scorer/test_06_groq_scorer.py
"""

import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated, load_groq_key

MODEL = "openai/gpt-oss-20b"
BASE_URL = "https://api.groq.com/openai/v1"


def run_test():
    print(f"TEST 6: {MODEL} (scorer, via Groq)")
    print("=" * 50)

    key = load_groq_key()
    if not key:
        print("FAILED: ENGINE_GROQ_API_KEY not found in interview-engine/.env")
        return
    print("OK: Groq API key is configured.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": (
                "Score this interview strictly against the given rubric. "
                "Every score must cite the specific turn(s) it is based on."
            )},
            {"role": "user", "content": (
                'Transcript: [{"role": "candidate", "text": "I used FastAPI '
                'and PostgreSQL to build a REST API."}]\n\n'
                'Rubric: {"technical_depth": "1-5 scale"}\n\n'
                'Return JSON: {"technical_depth": 0, "citation": ""}'
            )},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    print("\nSending a real scoring-shaped request...")
    t0 = time.perf_counter()
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload, timeout=60,
        )
    except httpx.ConnectError as e:
        print(f"FAILED: could not reach Groq — {e}")
        return
    elapsed = time.perf_counter() - t0

    if resp.status_code == 401:
        print("FAILED: Groq rejected the API key (401 Unauthorized).")
        return
    if resp.status_code == 429:
        print("FAILED: Groq rate-limited this request (429).")
        print(resp.text[:300])
        return
    if resp.status_code >= 400:
        print(f"FAILED: Groq returned {resp.status_code}")
        print(f"        body: {resp.text[:500]}")
        return
    data = resp.json()
    usage = data.get("usage", {})

    print(f"\nResponse: {data['choices'][0]['message']['content']!r}")
    print(f"Prompt tokens:     {usage.get('prompt_tokens')}")
    print(f"Completion tokens: {usage.get('completion_tokens')}")
    print(f"Total round trip:  {elapsed:.2f}s")

    print("\n" + "=" * 50)
    print("RESULT: PASS")


if __name__ == "__main__":
    run_gated(run_test)
