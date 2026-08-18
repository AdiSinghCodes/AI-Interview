"""
TEST 4: qwen3:8b — Planner role
====================================================
What it shows:
  - Whether Ollama has this model pulled
  - Biggest local model (~5.2GB), tight fit on a 6GB GPU — watch
    whether it stays 100% GPU or spills to CPU (`ollama ps` in another
    terminal while this runs)
  - Real JSON-mode generation speed for a planner-shaped call

Requirements:
  Ollama running on localhost:11434
  ollama pull qwen3:8b

Run:
  python testing_engine_models/planner/test_04_ollama_planner.py
"""

import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

MODEL = "qwen3:8b"
HOST = "http://localhost:11434"


def run_test():
    print(f"TEST 4: {MODEL} (planner)")
    print("=" * 50)

    try:
        tags = httpx.get(f"{HOST}/api/tags", timeout=5).json()
    except httpx.ConnectError:
        print("FAILED: Ollama is not reachable at", HOST)
        return

    names = [m["name"] for m in tags.get("models", [])]
    if not any(n.startswith(MODEL) for n in names):
        print(f"FAILED: {MODEL!r} not found in `ollama list`.")
        print(f"  -> run: ollama pull {MODEL}")
        return
    print(f"OK: {MODEL} is pulled.")

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": 'Return JSON only: {"greeting": "<one word hello>"}',
        }],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"num_ctx": 8192, "num_predict": 100, "temperature": 0.8},
    }

    print("\nSending a real JSON-mode chat request (think: false)...")
    t0 = time.perf_counter()
    resp = httpx.post(f"{HOST}/api/chat", json=payload, timeout=180)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    load_s = data.get("load_duration", 0) / 1e9
    eval_s = data.get("eval_duration", 1) / 1e9
    eval_n = data.get("eval_count", 0)
    tok_s = eval_n / eval_s if eval_s else 0

    print(f"\nResponse text: {data['message']['content']!r}")
    print(f"Model load time:   {load_s:.2f}s  (0 if already resident)")
    print(f"Generation speed:  {tok_s:.1f} tok/s  ({eval_n} tokens in {eval_s:.2f}s)")
    print(f"Total round trip:  {elapsed:.2f}s")
    print(f"\nEstimated time for a full 4096-token plan at this speed: "
          f"~{4096 / max(tok_s, 0.1):.0f}s")

    print("\n" + "=" * 50)
    print("RESULT: PASS  -- run `ollama ps` in another terminal while this")
    print("        is generating to check GPU vs CPU processor split.")


if __name__ == "__main__":
    run_gated(run_test)
