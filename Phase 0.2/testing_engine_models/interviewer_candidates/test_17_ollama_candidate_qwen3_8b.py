"""
TEST 17: qwen3:8b — Interviewer role, CANDIDATE
====================================================
What it shows:
  - Whether Ollama has this model pulled
  - Load + generation speed for Qwen3 8B as a candidate interviewer
    model. At Q4 this is ~5.2 GB — lands in the 6-8 GB range in actual
    use once KV cache is loaded, so it will spill off a 6 GB card.
  - qwen3:4b (test 1) narrates its reasoning directly in `content` even
    with think:false — the system prompt below is what actually stops
    that, not the flag alone. Same model family, same fix applied here.
  - This model is already covered as the Planner (test 4) with a
    different system prompt; this file tests it specifically as an
    Interviewer candidate for apples-to-apples comparison with the
    other rows in the candidate table.

Interactive: type a question, get the model's answer and timing, repeat.
Empty line, Ctrl+C, or "exit"/"quit" ends the session.

Requirements:
  Ollama running on localhost:11434
  ollama pull qwen3:8b

Run:
  python interviewer_candidates/test_17_ollama_candidate_qwen3_8b.py
"""

import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

MODEL = "qwen3:8b"
HOST = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are an interviewer speaking aloud to a candidate. Your words are "
    "converted directly to speech, so you speak in plain conversational "
    "sentences. Keep each turn to two or three sentences. Ask one thing at "
    "a time and wait for the answer. Reply with your final answer only — "
    "do not narrate your reasoning or think out loud, and do not use "
    "phrases like 'the user is asking' or 'let me think'."
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
        "think": False,          # the ACTUAL Ollama switch — not a prompt suffix
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
    thinking_leaked = bool(data["message"].get("thinking", "").strip())

    print(f"\n{data['message']['content'].strip()}\n")
    print(f"Thinking trace present: {'YES (bug!)' if thinking_leaked else 'no (correct)'}")
    print(f"Model load time:   {load_s:.2f}s  (0 if already resident)")
    print(f"Generation speed:  {tok_s:.1f} tok/s  ({eval_n} tokens in {eval_s:.2f}s)")
    print(f"Total round trip:  {data['_elapsed']:.2f}s")


def run_test():
    print(f"TEST 17: {MODEL} (interviewer — candidate)")
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
