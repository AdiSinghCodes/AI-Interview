# Phase 0.2 — Interview-Engine Model Test Suite

Same idea as Phase 0.1's `testingpretrainedmodels/` — one standalone script
per model, run independently, clear pass/fail output. These test the LLM /
STT / TTS / VAD models behind the `newai-interviewer` interview-engine
project (a separate repo from this one).

## Why this exists

Ad-hoc one-liners in a terminal scroll away and are hard to re-run. These
scripts are saved, reusable, and each isolates exactly one model so a hang
or slowdown is attributable to a single piece, not "the app is slow."

Every script waits for **Enter** before starting and prints wall-clock
**start/end times** plus a duration, so timing is obvious when running by
hand. `test_12_full_pipeline.py` skips the Enter-wait automatically so the
whole suite can run unattended.

## Setup (do this once)

```bash
ollama pull qwen3:4b
ollama pull phi4-mini
ollama pull llama3.2:3b
ollama pull qwen3:8b
ollama pull nomic-embed-text
pip install httpx faster-whisper kokoro torch
```

Tests 19-23 (avatar/lip-sync) are a separate table below, since each of
those is a cloned repo + checkpoint download rather than a pip package.

`ENGINE_GROQ_API_KEY` must be set in
`newai-interviewer/interview-engine/.env` for tests 5 and 6 — these scripts
read it directly from that file (see `_harness.py`), since this project
doesn't have the engine package installed.

## Folder layout

Scripts are grouped by role so it's obvious what each one tests at a glance.
`_harness.py` stays at the top level — every script adds the parent folder
to `sys.path` so `from _harness import ...` still resolves from inside a
subfolder.

```
interviewer/              tests 1-3, 11 — production interviewer + fallback chain
interviewer_candidates/   tests 13-18 — alternate models being evaluated for the interviewer role
planner/                  test 4
parser/                   test 5
scorer/                   test 6
embedder/                 test 7
speech_to_text/           test 8
text_to_speech/           test 9
voice_activity_detection/ test 10
avatar/                   tests 19-23 — lip-sync/avatar model candidates
```

## Run each test

```bash
python interviewer/test_01_ollama_interviewer_primary.py    # qwen3:4b
python interviewer/test_02_ollama_interviewer_fallback1.py  # phi4-mini
python interviewer/test_03_ollama_interviewer_fallback2.py  # llama3.2:3b
python planner/test_04_ollama_planner.py                    # qwen3:8b
python parser/test_05_groq_parser.py                        # llama-3.1-8b-instant
python scorer/test_06_groq_scorer.py                        # gpt-oss-20b
python embedder/test_07_ollama_embedder.py                  # nomic-embed-text
python speech_to_text/test_08_stt_whisper.py                # faster-whisper
python text_to_speech/test_09_tts_kokoro.py                 # Kokoro
python voice_activity_detection/test_10_vad_silero.py       # Silero VAD
python interviewer/test_11_interviewer_fallback_chain.py    # real RoutingProvider, real chain
```

Candidate interviewer models under evaluation (none pulled yet):

```bash
python interviewer_candidates/test_13_ollama_candidate_llama31_8b.py     # llama3.1:8b
python interviewer_candidates/test_14_ollama_candidate_mistral_7b.py     # mistral:7b
python interviewer_candidates/test_15_ollama_candidate_qwen25_7b.py      # qwen2.5:7b
python interviewer_candidates/test_16_ollama_candidate_gemma3_4b.py      # gemma3:4b
python interviewer_candidates/test_17_ollama_candidate_qwen3_8b.py       # qwen3:8b (as interviewer, not planner)
python interviewer_candidates/test_18_ollama_candidate_deepseek_r1_7b.py # deepseek-r1:7b
```

Avatar / lip-sync candidates under evaluation (repos not cloned yet —
see "Avatar model setup" below):

```bash
python avatar/test_19_avatar_musetalk.py       # MuseTalk — current pick, real-time capable, ~4-6GB
python avatar/test_20_avatar_wav2lip.py        # Wav2Lip — lightweight fallback, ~2GB
python avatar/test_21_avatar_liveportrait.py   # LivePortrait — video-driven only, NOT audio-native
python avatar/test_22_avatar_sadtalker.py      # SadTalker — diffusion, slowest, moderate-high VRAM
python avatar/test_23_avatar_infinitetalk.py    # InfiniteTalk — audio-native, 14B backbone, likely RAM-bound on this 16GB box
```

## Run everything at once

```bash
python test_12_full_pipeline.py
```

## Avatar model setup (tests 19-23)

Unlike the LLM/STT/TTS/VAD tests above, these aren't pip-installable —
each is a GitHub repo with its own inference CLI and its own multi-GB
checkpoint set. The test scripts shell out to each repo's real
`inference.py` (or `scripts.inference` for MuseTalk, `generate_infinitetalk.py`
for InfiniteTalk) rather than reimplementing it, so they stay correct as
those repos evolve — but that also means each one needs to actually be
cloned + have weights downloaded before its test will do more than print
a `FAILED` setup checklist.

Repos and checkpoints go **outside** this git repo (they're multi-GB) at
`C:\Users\GHANSHYAM\Desktop\lipsync_models\<RepoName>` — same convention
as `_harness.py`'s `ENGINE_ENV_FILE` pointing at the separate
interview-engine project. See each `test_1[9]`/`test_2[0-3]` script's
docstring for the exact `git clone` / checkpoint-download commands for
that model.

Test assets (source face photo/video, driving video for LivePortrait) go
in `avatar/assets/` — see `avatar/assets/README.md`. Driving audio is
shared: tests 19, 20, 22, and 23 reuse
`text_to_speech/tts_output/round_2.wav` (generate it once via
`test_09_tts_kokoro.py`).

| Test | Model | What it catches |
|------|-------|------------------|
| 19 | MuseTalk | repo/checkpoints present?, real inference time + realtime factor on this project's TTS audio — this is what confirms whether the "real-time capable at ~4-6GB" claim holds on this 6GB card |
| 20 | Wav2Lip | repo/checkpoints present?, real inference time — lightweight fallback if MuseTalk is too tight alongside the interviewer LLM |
| 21 | LivePortrait | repo/weights present?, real inference time on the video-driven path only — this test does NOT prove audio-driven lip-sync, it just confirms the video-driven motion-transfer path this project already ruled out for audio |
| 22 | SadTalker | repo/checkpoints present?, real inference time — diffusion-based, animates head pose too, expected to be the slowest of the four |
| 23 | InfiniteTalk | repo/3 weight sets present?, real inference time in low-VRAM mode on a 14B-parameter backbone (Wan2.1-I2V-14B) — the only audio-native option evaluated (no separate driving video needed) but public reports put its low-VRAM path's system-RAM need at 24-32GB against this machine's ~16GB, so `FAILED` here most likely means "hit that wall," not "broken setup" |

## What each test actually checks

| Test | Model | Role | What it catches |
|------|-------|------|------------------|
| 1 | qwen3:4b | interviewer (primary) | pulled?, `think: false` actually suppresses reasoning trace |
| 2 | phi4-mini | interviewer (fallback 1) | pulled?, load + gen speed |
| 3 | llama3.2:3b | interviewer (fallback 2) | pulled?, load + gen speed |
| 4 | qwen3:8b | planner | pulled?, JSON-mode speed, GPU vs CPU spillover (tight 6GB fit) |
| 5 | llama-3.1-8b-instant | parser | Groq key valid?, context/TPM ceiling |
| 6 | gpt-oss-20b | scorer | Groq key valid? (off live-latency path) |
| 7 | nomic-embed-text | embedder | pulled?, vector dim matches engine's `embedding_dim` (768) |
| 8 | faster-whisper | STT | real load time — this is what hung when config named an incompatible Parakeet model, and what crashed when forced onto a GPU missing `cublas64_12.dll` |
| 9 | Kokoro | TTS | load + synth time, realtime factor |
| 10 | Silero VAD | VAD | load time, silence-vs-speech correctness |
| 11 | fallback chain | interviewer | proves the real `RoutingProvider` walks `LLMSpec.fallback` — caught a bug where non-reasoning fallback models (phi4-mini, llama3.2:3b) 400'd on every call because the code sent them an unsupported `think` parameter |

## Known state as of last full run

- Ollama LLM roles (tests 1-4, 7): all pass.
- Groq roles (tests 5-6): pass; a `400` seen once was a transient blip,
  reproduced clean on retry — script now prints the full error body if it
  happens again.
- STT (test 8): fails on GPU (`cublas64_12.dll` missing) — engine config
  now runs it on CPU instead, which also frees VRAM for the LLM.
- TTS/VAD (9-10): pass.
- Fallback chain (11): pass, after fixing the `think` parameter bug above
  and removing `json_mode=True` from the interviewer role (it was making
  the model echo the input back as JSON instead of a spoken answer).
- Avatar/lip-sync (19-22): not yet run — none of the 4 repos are cloned
  locally yet. Each test currently reports `FAILED: repo not found` with
  the exact clone command until setup happens (see "Avatar model setup").
