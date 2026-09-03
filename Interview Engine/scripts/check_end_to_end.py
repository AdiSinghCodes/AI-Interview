"""
End-to-end smoke test for the whole interview, no browser and no mic.

Synthesises a few "candidate answers" with Kokoro, streams them into the
live WebSocket as real utterances, and checks the interview runs start to
finish: opening -> acknowledged Q&A turns -> closing -> transcript ->
summary. Prints PASS / FAIL.

    # 1. start the engine in another terminal:
    python -m engine.api.main

    # 2. once it says it's up (~60-90s warmup), run:
    python scripts/check_end_to_end.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ENGINE = "http://localhost:8000"
WS = "ws://localhost:8000"
SR, FRAME = 16_000, 512

ANSWERS = [
    "I built a service that ingests bank statement files and reconciles them against our "
    "internal ledger. It runs on FastAPI with a Postgres database and flags mismatches for review.",
    "A process has its own isolated memory while threads share memory within a process. "
    "I use threads for I O bound work and separate processes for C P U bound work because of the global interpreter lock.",
    "I would use cursor based pagination keyed on an indexed column, return a next cursor token "
    "instead of an offset, and avoid an exact total count because it forces a full scan.",
    "I would add a timeout so the call fails fast, a circuit breaker so we stop hammering a failing "
    "dependency, and a bulkhead to isolate its thread pool.",
    "That mostly covers it, thanks.",
]


def synth_answers(outdir: Path) -> list[Path]:
    from kokoro import KPipeline

    print("synthesising candidate answers with Kokoro (~40s the first time)...")
    pipe = KPipeline(lang_code="a")
    paths = []
    for i, text in enumerate(ANSWERS):
        import soundfile as sf

        chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice="af_heart")]
        audio = np.concatenate(chunks)
        p = outdir / f"ans{i}.wav"
        sf.write(str(p), audio, 24_000)
        paths.append(p)
    return paths


def load_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio.functional as AF

        a = AF.resample(torch.from_numpy(a), sr, SR).numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


async def stream_utterance(ws, audio: np.ndarray) -> None:
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        await ws.send(audio[i : i + FRAME].tobytes())
        await asyncio.sleep(FRAME / SR)
    for _ in range(45):  # ~1.4s trailing silence -> VAD endpoint
        await ws.send(np.zeros(FRAME, dtype=np.float32).tobytes())
        await asyncio.sleep(FRAME / SR)


async def main() -> int:
    import websockets

    try:
        async with httpx.AsyncClient(timeout=10) as h:
            await h.get(f"{ENGINE}/health")
    except Exception:
        print(f"FAIL: engine not reachable at {ENGINE} — start it with `python -m engine.api.main`")
        return 1

    with tempfile.TemporaryDirectory() as td:
        clips = [load_16k(p) for p in synth_answers(Path(td))]

    async with httpx.AsyncClient(timeout=180) as h:
        s = (await h.post(f"{ENGINE}/v1/sessions", json={"role": "technical", "num_questions": 4})).json()
        sid, rid = s["session_id"], s["round_id"]
        print(f"session {sid}  round {rid}  ({s['num_questions']} questions)")
        for _ in range(90):
            st = (await h.get(f"{ENGINE}/v1/sessions/{sid}/status")).json()
            if st["ready"]:
                break
            await asyncio.sleep(2)
        print(f"prefetch ready ({st['done']}/{st['total']})\n" + "=" * 70)

    turns_seen, cache_hits, closed = [], 0, False
    reply = asyncio.Event()

    async with websockets.connect(f"{WS}/v1/rounds/{rid}/live", max_size=None, ping_timeout=None) as ws:
        async def rx():
            nonlocal cache_hits, closed
            async for m in ws:
                if isinstance(m, bytes):
                    continue
                d = json.loads(m)
                if d["type"] == "opening":
                    print(f"\nINTERVIEWER: {d['text'][:100]}...")
                elif d["type"] == "transcript":
                    print(f"\n  heard: {d['text'][:90]!r}")
                elif d["type"] == "interviewer":
                    cache_hits += int(d.get("cache_hit", False))
                    turns_seen.append(d["kind"])
                    print(f"INTERVIEWER [{d['kind']}]: {d['text'][:90]}")
                    reply.set()
                elif d["type"] == "end":
                    closed = True
                    reply.set()

        rxt = asyncio.create_task(rx())
        await ws.send(np.zeros(FRAME, dtype=np.float32).tobytes())
        await asyncio.sleep(8)  # opening plays

        for clip in clips:
            if closed:
                break
            reply.clear()
            await stream_utterance(ws, clip)
            try:
                await asyncio.wait_for(reply.wait(), timeout=120)
            except asyncio.TimeoutError:
                print("  FAIL: no interviewer reply within 120s")
                rxt.cancel()
                return 1
            await asyncio.sleep(2)
        rxt.cancel()

    async with httpx.AsyncClient(timeout=180) as h:
        transcript = (await h.get(f"{ENGINE}/v1/rounds/{rid}/transcript")).json()["turns"]
        summary = (await h.post(f"{ENGINE}/v1/rounds/{rid}/summary")).json().get("summary", "")

    print("\n" + "=" * 70)
    checks = {
        "opening spoken": transcript and transcript[0]["kind"] == "opening",
        "candidate answers transcribed": any(t["role"] == "candidate" and t["text"] for t in transcript),
        "interviewer asked questions": turns_seen.count("question") >= 2,
        "planned questions served from cache": cache_hits >= 2,
        "interview reached a closing line": "closing" in turns_seen or closed,
        "transcript endpoint works": len(transcript) >= 4,
        "summary generated (no reasoning leak)": len(summary) > 80 and not summary.lower().startswith("okay,"),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nsummary:", summary[:400])

    ok = all(checks.values())
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
