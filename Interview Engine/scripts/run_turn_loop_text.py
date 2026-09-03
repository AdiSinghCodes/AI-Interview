"""
NTRVSTA Phase 2 — build-order step 1 demo.

Runs the SessionOrchestrator text-only against a hardcoded InterviewPlan:
type the candidate's answer, the orchestrator walks LISTENING -> THINKING
-> SPEAKING, the interviewer model (real Ollama call, real TTFT) replies,
and the per-turn latency trace prints. No audio, no WebSocket.

    cd "Interview Engine" && pip install -e .
    python scripts/run_turn_loop_text.py

Needs Ollama running with the interviewer model from config/models.yaml
pulled (default qwen3:4b).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx

# The interviewer speaks in em dashes and curly quotes; the legacy Windows
# console is cp1252 and would mangle them (same class of bug as Phase 0.2's
# LivePortrait emoji crash).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.inference.provider import InferenceError
from engine.plans import get_plan
from engine.inference.routing_provider import build_router
from engine.orchestrator import SessionOrchestrator, TurnState
from engine.settings import Role, get_models, get_settings


def _preflight() -> bool:
    settings = get_settings()
    spec = get_models().roles[Role.INTERVIEWER]
    try:
        tags = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5).json()
    except httpx.HTTPError:
        print(f"FAILED: Ollama not reachable at {settings.ollama_host}")
        return False

    pulled = {m["name"] for m in tags.get("models", [])}

    def is_pulled(model: str) -> bool:
        base = model.split(":")[0]
        return model in pulled or any(n.split(":")[0] == base for n in pulled)

    if not is_pulled(spec.name):
        print(f"FAILED: interviewer model {spec.name!r} not pulled. Run: ollama pull {spec.name}")
        return False

    fallbacks = [f.name for f in spec.fallback if is_pulled(f.name)]
    print(f"interviewer: {spec.name}   fallbacks pulled: {fallbacks or 'none'}")
    return True


def _format_trace(result, wall_ms: float) -> str:
    ttft = f"{result.ttft_ms:.0f} ms" if result.ttft_ms is not None else "n/a"
    rr = result.trace_marks.get("response_ready")
    rr_s = f"{rr:.0f} ms" if rr is not None else "n/a"
    return (
        f"    [{result.kind.value}]  ttft {ttft}  |  response_ready {rr_s}  "
        f"|  wall {wall_ms:.0f} ms  |  {result.degrade_level}"
    )


async def main() -> int:
    if not _preflight():
        return 1

    provider = build_router()
    orch = SessionOrchestrator("demo-round", get_plan("technical"), provider)

    print("\n" + "=" * 70)
    print(f"interviewer > {await orch.start()}")
    print("=" * 70)
    print("(type an answer and press Enter; empty line, 'exit', or Ctrl+C ends it)")

    try:
        while orch.state == TurnState.LISTENING:
            try:
                answer = input("\ncandidate  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                orch.end()
                break
            if not answer or answer.lower() in {"exit", "quit"}:
                orch.end()
                break

            t0 = time.perf_counter()
            try:
                result = await orch.submit_answer(answer)
            except InferenceError as e:
                print(f"    inference failed (all fallbacks exhausted): {e}")
                orch.end()
                break
            wall = (time.perf_counter() - t0) * 1000.0

            print(f"\ninterviewer > {result.interviewer_text}")
            print(_format_trace(result, wall))

            if result.interview_over:
                print("\n-- planned questions exhausted, interview complete --")
                break
    finally:
        print("\n" + orch.tracer.dump())
        await provider.aclose()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
