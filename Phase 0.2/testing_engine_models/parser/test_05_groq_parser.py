"""
TEST 5: llama-3.1-8b-instant (Groq) — Parser role
====================================================
What it shows:
  - Whether the Groq API key (read from interview-engine's .env) is
    valid
  - Real latency for a resume-parsing-shaped call (JSON mode)
  - Groq's free-tier context/TPM ceiling in practice

Requirements:
  ENGINE_GROQ_API_KEY set in newai-interviewer/interview-engine/.env

Run:
  python testing_engine_models/parser/test_05_groq_parser.py

"""

import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated, load_groq_key

MODEL = "llama-3.1-8b-instant"
BASE_URL = "https://api.groq.com/openai/v1"

SAMPLE_RESUME = (
    "Jane Doe\n\nExperience:\nSoftware Intern at Acme Corp, built REST "
    "APIs in Python and FastAPI, worked with PostgreSQL.\n\n"
    "Skills: Python, FastAPI, PostgreSQL, Git\n\nEducation: BE Computer Engineering"
)


def run_test():
    print(f"TEST 5: {MODEL} (parser, via Groq)")
    print("=" * 50)

    key = load_groq_key()
    if not key:
        print("FAILED: ENGINE_GROQ_API_KEY not found in interview-engine/.env")
        return
    print("OK: Groq API key is configured.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Extract resume fields as JSON."},
            {"role": "user", "content": (
                f"Resume:\n{SAMPLE_RESUME}\n\n"
                'Return JSON: {"name": "", "skills": [], "education": ""}'
            )},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    print("\nSending a real resume-parse-shaped request...")
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
        print("FAILED: Groq rate-limited this request (429) — this is the")
        print("        context/TPM ceiling issue. Try a shorter prompt.")
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
