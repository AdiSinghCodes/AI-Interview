"""
TEST 15: qwen2.5:7b — Interviewer role, CANDIDATE
====================================================
What it shows:
  - Whether Ollama has this model pulled
  - Load + generation speed for Qwen 2.5 7B as a candidate interviewer
    model. At Q4 this is ~4.7 GB — comfortable headroom on a 6 GB card.
  - Unlike Qwen3, Qwen2.5 has no hybrid-thinking mode, so no "think"
    flag is sent for it.

Interactive: type a question, get the model's answer and timing, repeat.
Empty line, Ctrl+C, or "exit"/"quit" ends the session.

Requirements:
  Ollama running on localhost:11434
  ollama pull qwen2.5:7b

Run:
  python interviewer_candidates/test_15_ollama_candidate_qwen25_7b.py
"""

import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

MODEL = "qwen2.5:7b"
HOST = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are an interviewer speaking aloud to a candidate. Your words are "
    "converted directly to speech, so you speak in plain conversational "
    "sentences. Keep each turn to two or three sentences. Ask one thing at "
    "a time and wait for the answer."
)


def strip_markdown(text: str) -> str:
    """TTS safety net — small models emit markdown despite instructions."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # bold
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)  # italic
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)  # numbered
    text = re.sub(r"`+", "", text)                     # code ticks
    return re.sub(r"\n{2,}", " ", text).strip()


def check_model_available() -> bool:
    try:
        tags = httpx.get(f"{HOST}/api/tags", timeout=5).json()
    except httpx.ConnectError:
        print("FAILED: Ollama is not reachable at", HOST)
        print("  -> is the Ollama app/service running?")
        return False

    names = [m["name"] for m in tags.get("models", [])]
    if not any(n.startswith(MODEL.split(":")[0]) for n in names):
        print(f"FAILED: {MODEL!r} not found in `ollama list`.")
        print(f"  -> run: ollama pull {MODEL}")
        return False
    print(f"OK: {MODEL} is pulled.")
    return True


def ask(messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": -1,        # stop the reload cost between turns
        "options": {
            "num_predict": 120,  # a spoken turn, not an essay
            "num_ctx": 4096,     # default is 2048 and silently truncates
            "temperature": 0.7,
        },
    }
    t0 = time.perf_counter()
    resp = httpx.post(f"{HOST}/api/chat", json=payload, timeout=120)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    data["_elapsed"] = elapsed
    data["message"]["content"] = strip_markdown(data["message"]["content"])
    return data


def print_stats(data: dict) -> None:
    load_s = data.get("load_duration", 0) / 1e9
    eval_s = data.get("eval_duration", 1) / 1e9
    eval_n = data.get("eval_count", 0)
    tok_s = eval_n / eval_s if eval_s else 0

    print(f"\n{data['message']['content'].strip()}\n")
    print(f"Model load time:   {load_s:.2f}s  (0 if already resident)")
    print(f"Generation speed:  {tok_s:.1f} tok/s  ({eval_n} tokens in {eval_s:.2f}s)")
    print(f"Total round trip:  {data['_elapsed']:.2f}s")


def run_test():
    print(f"TEST 15: {MODEL} (interviewer — candidate)")
    print("=" * 50)

    if not check_model_available():
        return

    print("\nType your question and press Enter. Empty line or Ctrl+C to quit.")

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        messages.append({"role": "user", "content": question})
        try:
            data = ask(messages)
        except httpx.HTTPStatusError as e:
            print(f"FAILED: {e.response.status_code} {e.response.text}")
            messages.pop()
            continue
        except httpx.ConnectError:
            print("FAILED: lost connection to Ollama.")
            messages.pop()
            continue

        messages.append(data["message"])
        print_stats(data)


if __name__ == "__main__":
    run_gated(run_test)
