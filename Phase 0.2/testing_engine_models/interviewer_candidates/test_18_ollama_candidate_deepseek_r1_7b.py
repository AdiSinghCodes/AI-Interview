"""
TEST 18: deepseek-r1:7b — Interviewer role, CANDIDATE
====================================================
What it shows:
  - Whether Ollama has this model pulled
  - Load + generation speed for the DeepSeek-R1 7B distill as a
    candidate interviewer model. At Q4 this is ~4.5 GB — fits, but
    it's a reasoning model: it burns tokens on a <think> block before
    it ever gets to the spoken answer, which is exactly the latency
    tradeoff this test is meant to expose.
  - "think" is deliberately left UNSET here (unlike test 1 and test 17)
    so the reasoning trace runs and its real cost shows up in the
    timing, instead of being suppressed.
  - Some Ollama builds return the reasoning as a separate
    message.thinking field; others inline it as a <think>...</think>
    block inside content. This script handles both so the split
    between "thinking time" and "speaking time" is visible either way.

Interactive: type a question, get the model's answer and timing, repeat.
Empty line, Ctrl+C, or "exit"/"quit" ends the session.

Requirements:
  Ollama running on localhost:11434
  ollama pull deepseek-r1:7b

Run:
  python interviewer_candidates/test_18_ollama_candidate_deepseek_r1_7b.py
"""

import re
import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

MODEL = "deepseek-r1:7b"
HOST = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are an interviewer speaking aloud to a candidate. Your words are "
    "converted directly to speech, so you speak in plain conversational "
    "sentences. Keep each turn to two or three sentences. Ask one thing at "
    "a time and wait for the answer."
)


def strip_markdown(text: str) -> str:
    """TTS safety net — small models emit markdown despite instructions."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # bold
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)  # italic
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)  # numbered
    text = re.sub(r"`+", "", text)                     # code ticks
    return re.sub(r"\n{2,}", " ", text).strip()


def split_inline_thinking(text: str) -> tuple[str, str]:
    """Some Ollama builds don't route deepseek-r1's <think> block to the
    dedicated `thinking` field — it shows up inline in `content` instead."""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if not m:
        return "", text.strip()
    return m.group(1).strip(), text[m.end():].strip()


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
            "num_predict": 120,  # a spoken turn's worth — reasoning eats into this
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

    thinking = data["message"].get("thinking", "") or ""
    content = data["message"]["content"]
    if not thinking.strip():
        thinking, content = split_inline_thinking(content)

    data["_thinking"] = thinking
    data["message"]["content"] = strip_markdown(content)
    return data


def print_stats(data: dict) -> None:
    load_s = data.get("load_duration", 0) / 1e9
    eval_s = data.get("eval_duration", 1) / 1e9
    eval_n = data.get("eval_count", 0)
    tok_s = eval_n / eval_s if eval_s else 0

    if data["_thinking"]:
        print(f"\n[reasoning, {len(data['_thinking'].split())} words — not spoken]")
    if data["message"]["content"]:
        print(f"\n{data['message']['content']}\n")
    else:
        print("\n(no spoken answer — num_predict was spent entirely on reasoning)\n")
    print(f"Model load time:   {load_s:.2f}s  (0 if already resident)")
    print(f"Generation speed:  {tok_s:.1f} tok/s  ({eval_n} tokens in {eval_s:.2f}s)")
    print(f"Total round trip:  {data['_elapsed']:.2f}s")


def run_test():
    print(f"TEST 18: {MODEL} (interviewer — candidate)")
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
