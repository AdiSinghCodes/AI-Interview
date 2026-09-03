# Interview Engine — NTRVSTA

A voice interview you can sit through end to end: pick a round, pass a device
check, and an AI interviewer talks to you — asks a question aloud, you answer
aloud, it acknowledges and moves on, for the length of the interview, then shows
a transcript and a written read on how it went.

Model findings behind it: `../Phase 0.2/testing_engine_models/`. Architecture:
`../../Downloads/NTRVSTA_Phase2_Integration.pdf`.

## Run it

```powershell
pwsh -File run.ps1
```

Starts the engine (`:8000`) and the web app (`:5173`) and opens the browser. The
engine warms its speech models on startup (~60–90 s the first time). Or manually:

```bash
# engine
cd "Interview Engine" && pip install -e ".[speech,dev]" && python -m engine.api.main
# web  (separate terminal)
cd "Interview Engine/web" && npm install && npm run dev
```

Then open http://localhost:5173 — **headphones required** (the pre-flight check
enforces it; without them the interviewer's voice leaks into your mic).

## The flow

1. **Setup** — pick a round (general technical / Python / React / DSA / HR) and a
   question count. Creates a session; the engine starts synthesising the
   questions in the background.
2. **Device check** — camera + mic test, headphone confirm, and a "preparing your
   interview" bar (the prefetch). "Start" unlocks when the essentials are ready.
3. **Interview** — the avatar speaks; you answer out loud. Silero decides when
   you've stopped, whisper transcribes, the interviewer acknowledges and asks the
   next question. Planned questions play **instantly** from the prefetch cache;
   the occasional live follow-up takes a few seconds. Live transcript on the right.
4. **Summary** — full transcript + a short written read from the interviewer.

## How the latency works

Most turns are planned questions, and their audio is generated up front (during
the device check) and again in the background during the interview
(`engine/orchestrator/prefetch_cache.py`). So a planned-question turn costs only
the STT time — no LLM, no TTS wait. STT is `base.en` on CPU with incremental
transcription during the answer, so finalising at the endpoint is quick.

Genuine follow-ups (pacing decides when — after a substantive answer, capped per
round) run the live LLM → clause-chunker → Kokoro path and take a few seconds —
which reads as a natural pause.

On this CPU-only dev box, warm: planned turn ≈ 3–5 s (all STT), follow-up ≈ +4 s.
A CUDA torch build would take both well down (Kokoro-82M is ~0.5 GB).

## Layout

| Path | Role |
|------|------|
| `config/models.yaml` | **the only place a model name appears** — swap a model here, nothing else |
| `engine/settings.py` | loads + validates that YAML into typed specs |
| `engine/plans/` | the curated interview plans, one per role (stand-in for the generative planner) |
| `engine/orchestrator/` | the turn state machine, latency tracer, prefetch cache, degrade ladder, pacing, echo gate |
| `engine/inference/` | Ollama + Groq + routing with the Phase-0.2 fallback chain |
| `engine/speech/` | Silero VAD + endpointer, faster-whisper (incremental), Kokoro TTS (worker process), clause chunker |
| `engine/agents/interviewer.py` | live turn generation, streamed clause-by-clause |
| `engine/transport/ws_live.py` | the duplex WebSocket — PCM in, transcript + audio out |
| `engine/api/` | `POST /v1/sessions`, `/status`, `WS /v1/rounds/{id}/live`, `/transcript`, `/summary`, `/preflight` |
| `web/` | the React app (Setup → Preflight → InterviewRoom → Summary) |
| `scripts/run_turn_loop_text.py` | text-only turn loop (no audio) for quick backend checks |
| `scripts/run_ws_client.py` | drive the live socket from a WAV or the mic, no browser |

Stubs still to fill (PDF build order): generative planner + resume upload, rubric
scoring, proctoring integration, barge-in, neural lip-sync, DB persistence, the
DSA code editor.

## Tests

```bash
pytest              # 68 fast tests, no models
pytest -m slow      # +4 that load real Silero / whisper / Kokoro (~1 min)
```
