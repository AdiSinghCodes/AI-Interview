"""
TEST 11: Interviewer fallback chain — end to end (real engine code)
====================================================
What it shows:
  - Proves interview-engine's RoutingProvider actually walks
    LLMSpec.fallback in order, using the REAL RoutingProvider /
    OllamaProvider classes from the engine project — not a
    reimplementation. This is the one test in this folder that reaches
    into the other project, specifically so it stays authoritative.

  This test does NOT reimplement the JSON-schema / system-prompt /
  keep_alive fixes that tests 1-3 apply locally. Those belong in
  engine/inference/ollama_provider.py in the interview-engine project
  itself — once they're there, this test exercises them for free
  because it calls the real provider. Duplicating that logic here would
  defeat the point of this file, which is to stay authoritative against
  the real code path rather than a parallel copy of it.

  Three checks:
  1. Real config: primary (qwen3:4b) should just succeed.
  2. Deliberately broken primary (nonexistent model name) with the
     real fallback list attached — should recover via phi4-mini.
  3. Deliberately broken primary AND first fallback — should recover
     via the second fallback (llama3.2:3b), proving the chain walks
     more than one level deep, not just primary -> fallback[0].

  KNOWN PAST BUG this test caught: ollama_provider.py used to send
  "think": true/false to EVERY model unconditionally. Ollama 400s with
  "does not support thinking" for models that have no thinking mode at
  all (phi4-mini, llama3.2:3b) — which meant the fallback chain was
  completely non-functional: if qwen3:4b ever failed mid-interview,
  every fallback would 400 too. Fixed to only send `think` when
  spec.no_think is explicitly set.

Requirements:
  Ollama running, with qwen3:4b, phi4-mini, and llama3.2:3b all pulled.
  The newai-interviewer/interview-engine project present at the path
  below (this test imports its real engine package).

Run:
  python interviewer/test_11_interviewer_fallback_chain.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\GHANSHYAM\Desktop\newai-interviewer\interview-engine")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated_async

from engine.config import get_settings, ModelRole, LLMSpec
from engine.inference.ollama_provider import OllamaProvider
from engine.inference.routing_provider import RoutingProvider
from engine.inference.provider import ChatMessage


async def run_test():
    print("TEST 11: Interviewer fallback chain (real RoutingProvider)")
    print("=" * 50)

    settings = get_settings()
    ollama = OllamaProvider(settings.ollama_host)
    router = RoutingProvider(ollama, groq=None)
    messages = [ChatMessage(role="user", content="Say hello in one word.")]

    real_spec = settings.llm(ModelRole.INTERVIEWER)
    print(f"\n[1] Real config — primary={real_spec.name!r}, "
          f"fallbacks={[f.name for f in real_spec.fallback]}")
    try:
        result = await router.generate(real_spec, messages)
        print(f"    -> succeeded on primary: {result.text!r}")
    except Exception as e:
        print(f"    -> FAILED even with fallback chain: {type(e).__name__}: {e}")
        return

    if len(real_spec.fallback) < 2:
        print(f"\n[!] Only {len(real_spec.fallback)} fallback(s) configured — "
              f"skipping check [3] (needs at least 2 to test depth).")

    broken_spec = LLMSpec(
        name="this-model-does-not-exist:latest",
        num_ctx=4096, temperature=0.7, max_tokens=50,
        stream=False, json_mode=False,
        fallback=list(real_spec.fallback),
    )
    print(f"\n[2] Broken primary={broken_spec.name!r} -> should fall back to "
          f"{broken_spec.fallback[0].name!r}")
    try:
        result = await router.generate(broken_spec, messages)
        print(f"    -> recovered via fallback: {result.text!r}")
        print("    -> RESULT: PASS — fallback chain works")
    except Exception as e:
        print(f"    -> FAILED: every candidate errored: {type(e).__name__}: {e}")
        print("    -> RESULT: FAIL")
        return

    if len(real_spec.fallback) >= 2:
        # RoutingProvider.generate() only flattens ONE level:
        # `for candidate in (spec, *spec.fallback)` — it never recurses
        # into a candidate's own .fallback. That matches how config.py
        # actually shapes the real chain: a flat sibling list under the
        # primary, not nested. So the second real fallback must be a
        # SIBLING of the second broken model here, not nested inside it
        # (nesting it was the bug in an earlier version of this test —
        # it made check [3] fail even though RoutingProvider was fine).
        second_level_fallback = list(real_spec.fallback[1:])
        double_broken_spec = LLMSpec(
            name="this-model-does-not-exist:latest",
            num_ctx=4096, temperature=0.7, max_tokens=50,
            stream=False, json_mode=False,
            fallback=[
                LLMSpec(
                    name="this-one-also-does-not-exist:latest",
                    num_ctx=4096, temperature=0.7, max_tokens=50,
                    stream=False, json_mode=False,
                ),
                *second_level_fallback,
            ],
        )
        print(f"\n[3] Primary AND fallback[0] both broken -> should recover "
              f"via fallback[1]={second_level_fallback[0].name!r}")
        try:
            result = await router.generate(double_broken_spec, messages)
            print(f"    -> recovered two levels deep: {result.text!r}")
            print("    -> RESULT: PASS — chain walks past a single fallback")
        except Exception as e:
            print(f"    -> FAILED: chain did not walk past one level: "
                  f"{type(e).__name__}: {e}")
            print("    -> RESULT: FAIL")
            return

    print("\n" + "=" * 50)
    print("RESULT: PASS")


if __name__ == "__main__":
    asyncio.run(run_gated_async(run_test))