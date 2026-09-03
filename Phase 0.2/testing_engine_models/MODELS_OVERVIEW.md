# Phase 0.2 — Every Model Used, What It Does, and How It's Tested

This documents every model evaluated in `testing_engine_models/` for the `newai-interviewer`
interview-engine: its functionality, why it fills that role, how its test script exercises it, and
what's been found running it on this machine (RTX 4050 Laptop, 6GB VRAM, ~16GB system RAM).

Each model has its own standalone test script — no shared "run everything" logic reimplements a
model's behavior; every script either calls the real API (Ollama/Groq) or shells out to a repo's
own inference CLI, so results stay authoritative against the real thing instead of a parallel copy
of its logic.

---

## 1. Interviewer role — asks questions, responds live during the interview

The interviewer is the model on the live-latency path: candidate speaks, STT transcribes it, this
model responds, TTS speaks it back. Speed and "sounding right when spoken aloud" both matter here
more than raw capability.

### Production chain (primary → fallback → fallback)

| Test | Model | Size (Q4) | Functionality |
|------|-------|-----------|----------------|
| [1](interviewer/test_01_ollama_interviewer_primary.py) | **qwen3:4b** (primary) | ~2.6GB | Hybrid-thinking model (`think: false` suppresses its `<think>` block via Ollama's real API flag — the old `/no_think` prompt trick does **not** work). Still narrates reasoning in plain prose despite `think:false`, which a JSON-schema-constrained response (`{"answer": "..."}`) structurally prevents, since a monologue can't validate against that schema. |
| [2](interviewer/test_02_ollama_interviewer_fallback1.py) | **phi4-mini** (fallback #1) | ~2.5GB | No hybrid-thinking mode at all — sending Ollama a `think` flag for this model 400s (this is exactly the bug test 11 caught). Its own failure mode is echoing/rephrasing the question back instead of answering; same JSON-schema fix applies. |
| [3](interviewer/test_03_ollama_interviewer_fallback2.py) | **llama3.2:3b** (fallback #2) | ~2GB | Last-resort model in the chain. Hasn't shown the reasoning-leak or echo failure modes seen in the other two, but keeps the same JSON-schema safety net rather than assuming it never will. |
| [11](interviewer/test_11_interviewer_fallback_chain.py) | *(all three, via real engine code)* | — | Not a model test — imports the **actual** `RoutingProvider`/`OllamaProvider` classes from the interview-engine project (not a reimplementation) and proves the fallback chain really walks primary → fallback[0] → fallback[1] on real failures. **Caught a real bug**: `ollama_provider.py` used to send `think: true/false` to *every* model unconditionally, so any model with no thinking mode (phi4-mini, llama3.2:3b) 400'd — meaning if qwen3:4b ever failed mid-interview, the entire fallback chain was non-functional. Fixed to only send `think` when the spec explicitly sets it. |

All three interviewer scripts share the same defensive pattern: a system prompt telling the model
it's speaking aloud (plain sentences, no markdown, 2-3 sentences, answer directly rather than
re-asking or restating), a regex `strip_markdown()` safety net (small models emit `**bold**` /
bullets despite instructions), and `keep_alive: -1` so the model stays resident between turns
instead of reloading (~15-20s) on every question.

### Candidate models (evaluated, not yet adopted for this role)

| Test | Model | Size (Q4) | Notes |
|------|-------|-----------|-------|
| [13](interviewer_candidates/test_13_ollama_candidate_llama31_8b.py) | llama3.1:8b | ~5.0GB | Tight on 6GB — little headroom left for KV cache on long transcripts. |
| [14](interviewer_candidates/test_14_ollama_candidate_mistral_7b.py) | mistral:7b | ~4.4GB | Comfortable headroom on a 6GB card. |
| [15](interviewer_candidates/test_15_ollama_candidate_qwen25_7b.py) | qwen2.5:7b | ~4.7GB | No hybrid-thinking mode (unlike Qwen3), so no `think` flag sent. |
| [16](interviewer_candidates/test_16_ollama_candidate_gemma3_4b.py) | gemma3:4b | ~2.9GB | Comfortable headroom. |
| [17](interviewer_candidates/test_17_ollama_candidate_qwen3_8b.py) | qwen3:8b (as interviewer) | ~5.2GB | Same model as the Planner (test 4) but evaluated here with an interviewer-shaped system prompt, for apples-to-apples comparison against the other candidate rows. Lands in the 6-8GB range once KV cache loads, so it spills off this 6GB card. |
| [18](interviewer_candidates/test_18_ollama_candidate_deepseek_r1_7b.py) | deepseek-r1:7b | ~4.5GB | A genuine reasoning model — deliberately left with `think` **unset** so its real `<think>` block cost shows up in the timing (the whole point of this test is exposing that latency tradeoff). Handles both response shapes Ollama may return (a separate `thinking` field, or an inline `<think>...</think>` block in `content`) so the split between "thinking time" and "speaking time" is visible either way. |

**None of the 6 candidates have been pulled yet** — the Readme notes this as "not yet run."

---

## 2. Planner role — plans interview structure / question sequencing

| Test | Model | Functionality |
|------|-------|----------------|
| [4](planner/test_04_ollama_planner.py) | **qwen3:8b** | Largest local model (~5.2GB at Q4) — the test specifically watches whether it stays 100% GPU-resident or spills to CPU on this 6GB card (checked via `ollama ps` in a separate terminal while the test runs). Off the live-turn latency path (interviewer is), so it can tolerate being slower/JSON-mode. |

---

## 3. Parser role — extracts structured fields from resume/transcript text

| Test | Model | Functionality |
|------|-------|----------------|
| [5](parser/test_05_groq_parser.py) | **llama-3.1-8b-instant** (Groq API) | Cloud-hosted, not local — no VRAM cost. Extracts resume fields (name/skills/education) as JSON via Groq's `response_format: json_object`. Test specifically probes Groq's free-tier context/TPM rate-limit ceiling (429 responses) since that's a real constraint in practice, not just whether the key is valid. |

---

## 4. Scorer role — scores the completed interview against a rubric

| Test | Model | Functionality |
|------|-------|----------------|
| [6](scorer/test_06_groq_scorer.py) | **openai/gpt-oss-20b** (Groq API) | Cloud-hosted. Takes a transcript + rubric, returns a JSON score that must cite the specific turn(s) it's based on. Runs **after** the interview ends, so it's off the live-latency path entirely — this test is mainly a "does it still work" check, not a speed benchmark. |

---

## 5. Embedder role — vectorizes text for retrieval/similarity (pgvector)

| Test | Model | Functionality |
|------|-------|----------------|
| [7](embedder/test_07_ollama_embedder.py) | **nomic-embed-text** (Ollama) | Must stay on Ollama regardless of what else runs on Groq, because **Groq has no embeddings API at all**. Test checks the returned vector is exactly 768-dimensional — the interview-engine's pgvector column is fixed at that width, so any mismatch silently breaks retrieval rather than erroring loudly. |

---

## 6. Speech-to-text (STT) — transcribes the candidate's spoken answers

| Test | Model | Functionality |
|------|-------|----------------|
| [8](speech_to_text/test_08_stt_whisper.py) | **faster-whisper** (`distil-small.en`, CPU, int8) | Records a few real seconds from your microphone and transcribes it, so accuracy is visually confirmed rather than just "didn't crash." Runs on **CPU**, not GPU — two known past bugs drove this: (1) config once pointed at `nvidia/parakeet-tdt-0.6b-v3`, a NeMo model faster-whisper can't load at all, which hung indefinitely; (2) GPU mode crashes with `cublas64_12.dll is not found` — the CUDA cuBLAS runtime ctranslate2 needs isn't present on this machine. Running STT on CPU also has the side benefit of freeing the entire 6GB of VRAM for the LLM. |

---

## 7. Text-to-speech (TTS) — speaks the interviewer's responses aloud

| Test | Model | Functionality |
|------|-------|----------------|
| [9](text_to_speech/test_09_tts_kokoro.py) | **Kokoro** (voice: `af_heart`) | Synthesizes typed text to speech, plays it back through real speakers, and saves a WAV — so voice quality/pronunciation is judged by ear, not just "a waveform came out." Reports a **realtime factor** (audio-seconds produced ÷ wall-clock seconds taken): the number that determines whether TTS can actually keep up with a live conversational turn. This is also where `round_2.wav` gets generated — the driving-audio file every avatar/lip-sync test below reuses. |

---

## 8. Voice activity detection (VAD) — detects when the candidate starts/stops speaking

| Test | Model | Functionality |
|------|-------|----------------|
| [10](voice_activity_detection/test_10_vad_silero.py) | **Silero VAD** (via `torch.hub`) | Correctness checks rather than a speed benchmark: silence must NOT register as speech, loud noise should, and endpointing (detecting the candidate has *stopped* talking) must correctly fire after ~8 consecutive silent frames (~250ms) so the system knows when to let the interviewer respond. |

---

## 9. Avatar / lip-sync — animates a still portrait to match the TTS audio

Unlike every model above, these aren't pip packages — each is a full GitHub repo with its own
inference CLI and its own multi-GB checkpoint set, kept **outside** this git repo at
`C:\Users\GHANSHYAM\Desktop\lipsync_models\<RepoName>`. Every test here shells out to that repo's
real inference script rather than reimplementing it.

| Test | Model | How it works | Functionality / verdict |
|------|-------|---------------|---------------------------|
| [19](avatar/test_19_avatar_musetalk.py) | **MuseTalk** | GAN-based; takes a short **source video** + audio, generates a lip-synced output video via the repo's `scripts.inference` (YAML-configured video/audio pair) | Current pick for this project — claimed real-time-capable at ~4-6GB VRAM, which is exactly what this test exists to confirm or kill on this specific 6GB card. Needs 6 separate checkpoint sets (musetalk, sd-vae, whisper, dwpose, syncnet, face-parse-bisent) downloaded via the repo's own script. **Not yet run** — repo not cloned locally yet. |
| [20](avatar/test_20_avatar_wav2lip.py) | **Wav2Lip** | Older GAN; takes a still photo or short video + audio, shells out to `inference.py` | Lightweight fallback (~2GB VRAM) if MuseTalk turns out too tight alongside the interviewer LLM. **Confirmed real bug on Windows**: its image/video-type detection does a naive `path.split('.')[1]`, and its audio+video mux shells out to ffmpeg via an *unquoted* string — both break on this project's own path (`Phase 0.2` has a literal dot, and a space). Fixed by staging every input/output file into the repo's own directory (no dots/spaces there) rather than patching Wav2Lip itself. **Not yet run.** |
| [21](avatar/test_21_avatar_liveportrait.py) | **LivePortrait** | Takes a still photo + a **driving video** (someone else's face moving), transfers that motion onto the portrait, via `inference.py` | **Important caveat this test exists to prove**: LivePortrait is video-driven, not audio-driven — it has no built-in audio-to-motion step, so it does **not** solve audio-driven lip-sync on its own. This test only confirms the video-driven motion-transfer path works, and was the reasoning basis for ruling it out as *the* avatar solution for a TTS-only pipeline. **Actually run**: passed after two environment fixes — missing `onnx` package (insightface's ONNX model zoo needs it, only `onnxruntime` was present, pulled in transitively by faster-whisper) and a 🚀 emoji in a progress-bar label that crashed on Windows' legacy console encoding (`cp1252` can't encode it) during teardown, patched out of `live_portrait_pipeline.py:273`. |
| [22](avatar/test_22_avatar_sadtalker.py) | **SadTalker** | Diffusion-based; still photo + audio → animates **head pose as well as lips**, via `inference.py --still --preprocess full` | Slowest of the models here (diffusion, moderate-high VRAM) but potentially the best visual quality since it moves the head, not just the mouth. **Actually run** in this session. |
| [23](avatar/test_23_avatar_infinitetalk.py) | **InfiniteTalk** | Audio-native end to end; still photo + audio (no driving video, no motion prompt needed) — built on the 14B-parameter **Wan2.1-I2V-14B-480P** video-diffusion backbone | The only model here that's audio-driven natively rather than needing a driving video. But the full backbone is **82.3GB** (this machine has ~74GB free disk) and even the smallest practical quantization needs more system RAM than this machine's 16GB in community reports — see [INFINITETALK_COMFYUI_SETUP.md](avatar/INFINITETALK_COMFYUI_SETUP.md) for the full low-VRAM/GGUF-via-ComfyUI setup path being attempted instead of the native script. **Not yet run** — weights not downloaded (native script needs them; the GGUF/ComfyUI path is the workaround in progress). |

All 5 avatar tests share driving audio: `text_to_speech/tts_output/round_2.wav`, generated once by
running test 9. Source face assets live in `avatar/assets/` (see `avatar/assets/README.md`).

---

## Known state as of last full run (from the suite's own Readme)

- **Ollama LLM roles** (interviewer primary + planner + embedder, tests 1-4, 7): all pass.
- **Groq roles** (parser + scorer, tests 5-6): pass; a transient `400` seen once reproduced clean
  on retry.
- **STT** (test 8): fails on GPU (missing `cublas64_12.dll`) — engine config now runs it on CPU,
  which also frees VRAM for the LLM.
- **TTS/VAD** (9-10): pass.
- **Fallback chain** (test 11): pass, after fixing the `think`-parameter bug described above and
  removing `json_mode=True` from the interviewer role (it made the model echo the input back as
  JSON instead of speaking an answer).
- **Interviewer candidates** (13-18): none pulled yet.
- **Avatar/lip-sync**: LivePortrait (21) and SadTalker (22) have actually been run in this session;
  MuseTalk (19), Wav2Lip (20), and InfiniteTalk (23) have not — each currently reports `FAILED:
  repo not found` or `FAILED: missing weight set(s)` with the exact setup commands until that
  happens.

## Run everything at once

```bash
python test_12_full_pipeline.py
```

Runs every test above as its own subprocess (so one hang/crash doesn't take down the rest),
skipping each script's "Press Enter" prompt via `MODEL_TEST_AUTO=1`, and prints one PASS/FAIL/
TIMEOUT summary table.
