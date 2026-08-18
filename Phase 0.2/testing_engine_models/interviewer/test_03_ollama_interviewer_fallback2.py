"""
TEST 3: llama3.2:3b — Interviewer role, FALLBACK #2
====================================================
What it shows:
  - Whether Ollama has this model pulled
  - Load + generation speed for the LAST-resort model in the
    interviewer fallback chain (qwen3:4b -> phi4-mini -> llama3.2:3b)
  - Same JSON-schema hardening as test_01/test_02: the model must return
    {"answer": "..."} so reasoning leakage (qwen3:4b) or question-echo
    (phi4-mini) is structurally impossible here too. llama3.2:3b hasn't
    shown either failure mode in earlier manual testing, but the schema
    stays on as a safety net rather than assuming it never will.

Interactive: type a question, get the model's answer and timing, repeat.
Empty line, Ctrl+C, or "exit"/"quit" ends the session.

Requirements:
  Ollama running on localhost:11434
  ollama pull llama3.2:3b

Run:
  python interviewer/test_03_ollama_interviewer_fallback2.py
"""

import re
import time
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated

MODEL = "llama3.2:3b"
HOST = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are an interviewer speaking aloud to a candidate. You must "
    "respond with a JSON object containing one field, 'answer', whose "
    "value is exactly what you would say out loud. The value of 'answer' "
    "is converted directly to speech, so write it as plain conversational "
    "sentences with no markdown, no lists, and no formatting — just the "
    "words you'd speak. Keep it to two or three sentences. Give a direct, "
    "concrete answer or statement — never respond to a question with "
    "another question, and never just restate or rephrase what was asked "
    "back to the speaker. If a question has a false or flawed premise, "
    "say so briefly in one sentence and correct it. If a question bundles "
    "more than one thing, pick the most important part and answer that "
    "directly."
)

# Same schema as test_01/test_02 — forces valid JSON so monologue, echo,
# or rephrasing can't be produced as output.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def check_model_available() -> bool:
    try:
        tags = httpx.get(f"{HOST}/api/tags", timeout=5).json()
    except httpx.ConnectError:
        print("FAILED: Ollama is not reachable at", HOST)
        print("  -> is the Ollama app/service running?")
        return False

    names = [m["name"] for m in tags.get("models", [])]
    if not any(n.startswith("llama3.2") for n in names):
        print(f"FAILED: {MODEL!r} not found in `ollama list`.")
        print(f"  -> run: ollama pull {MODEL}")
        return False
    print(f"OK: {MODEL} is pulled.")
    return True


def strip_markdown(text: str) -> str:
    """TTS safety net — small models emit markdown despite instructions."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # bold
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)  # italic
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)  # numbered
    text = re.sub(r"`+", "", text)                     # code ticks
    return re.sub(r"\n{2,}", " ", text).strip()


def extract_answer(raw_content: str) -> str:
    """Pull the answer out of the model's JSON response. Falls back to the
    raw text if the model still failed to produce valid JSON (rare with
    the schema constraint, but don't crash the test loop over it)."""
    import json
    try:
        parsed = json.loads(raw_content)
        return parsed["answer"].strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return raw_content.strip()


def ask(messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": -1,        # stop reload between turns
        "format": ANSWER_SCHEMA, # forces valid JSON — same hardening as test_01/test_02
        "options": {
            "num_predict": 160,  # JSON wrapper + a spoken turn
            "num_ctx": 4096,     # default is 2048 and silently truncates
            "temperature": 0.4,  # less rambling, more deterministic for an interviewer
        },
    }
    t0 = time.perf_counter()
    resp = httpx.post(f"{HOST}/api/chat", json=payload, timeout=120)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    data["_elapsed"] = elapsed
    content = extract_answer(data["message"]["content"])
    content = strip_markdown(content)
    data["message"]["content"] = content
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
    print(f"TEST 3: {MODEL} (interviewer — fallback #2)")
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