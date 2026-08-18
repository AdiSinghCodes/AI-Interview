"""
TEST 7: nomic-embed-text — Embedder role
====================================================
What it shows:
  - Whether Ollama has this model pulled
  - Real embedding latency + vector dimension (must be 768 to match
    the interview-engine's pgvector column)
  - Groq has no embeddings API, which is why this role must stay on
    Ollama regardless of what runs elsewhere.

Requirements:
  Ollama running on localhost:11434
  ollama pull nomic-embed-text

Run:
  python embedder/test_07_ollama_embedder.py
"""

import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

MODEL = "nomic-embed-text"
HOST = "http://localhost:11434"
EXPECTED_DIM = 768  # interview-engine/engine/config.py: ModelRegistry.embedding_dim


def run_test():
    print(f"TEST 7: {MODEL} (embedder)")
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

    print("\nEmbedding a sample sentence...")
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{HOST}/api/embeddings",
        json={"model": MODEL, "prompt": "Backend engineer, 2 years FastAPI experience."},
        timeout=30,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    vector = resp.json()["embedding"]

    dim_ok = len(vector) == EXPECTED_DIM

    print(f"\nVector length:     {len(vector)}  (expected {EXPECTED_DIM})")
    print(f"Dimension match:   {'OK' if dim_ok else 'MISMATCH — pgvector column will not match!'}")
    print(f"Total round trip:  {elapsed:.2f}s")

    print("\n" + "=" * 50)
    print("RESULT:", "PASS" if dim_ok else "FAIL — embedding_dim mismatch")


if __name__ == "__main__":
    run_gated(run_test)
