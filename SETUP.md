# SETUP — what to install so every model runs

This repo is several sub-projects, each exercising different ML models. This file
is the single "new machine" checklist: install the prerequisites once, then do
the section(s) for the phase you're working on.

> Reference machine for all timing/VRAM notes: RTX 4050 Laptop (6 GB VRAM),
> ~16 GB system RAM, Windows 11, Python 3.11.

---

## 1. System prerequisites (install once, all phases)

| Tool | Why | Get it |
|------|-----|--------|
| **Python 3.11** | every backend / test script (`.pyc` files are `cpython-311`) | https://www.python.org/downloads/ |
| **Git** | cloning this repo + the external avatar repos | https://git-scm.com |
| **FFmpeg** (on `PATH`) | faster-whisper audio decode, Kokoro/`soundfile` I/O, every avatar lip-sync mux | https://ffmpeg.org/download.html — or `winget install Gyan.FFmpeg` |
| **Ollama** (running) | all local LLM roles: interviewer, planner, embedder, Groq fallbacks | https://ollama.com/download — then `ollama serve` |
| **Node.js 18+** | the Phase 1 / Interview Engine web frontends only | https://nodejs.org |
| **NVIDIA CUDA toolkit + cuDNN** *(optional)* | GPU torch / avatar models. STT deliberately stays on CPU — see note below | https://developer.nvidia.com/cuda-downloads |

### GPU / CUDA note

- `pip install -r requirements.txt` pulls the **default torch wheel**. For a
  specific CUDA build, install torch/torchaudio **first** from
  https://pytorch.org/get-started/locally/ then run the requirements file.
- **faster-whisper is configured to run on CPU on purpose.** GPU mode needs
  `cublas64_12.dll` (CUDA cuBLAS runtime); if it's missing STT crashes with
  `cublas64_12.dll is not found`. CPU (`int8`) is fast enough for a live turn and
  frees all 6 GB of VRAM for the LLM.

---

## 2. Python environment

```powershell
# from the repo root
python -m venv myenv
myenv\Scripts\Activate.ps1          # Windows PowerShell
# source myenv/bin/activate         # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt     # aggregated superset for the whole repo
```

`requirements.txt` at the root is a convenience superset. Each sub-project also
ships a smaller list if you only need one:

| Sub-project | Narrower install |
|-------------|------------------|
| `Interview Engine/` | `pip install -e ".[groq,speech,dev]"` (see its `pyproject.toml`) |
| `Phase 1/backend/` | `pip install -r "Phase 1/backend/requirements.txt"` |
| `Phase 0/voice_clarity_detection/` | `pip install -r "Phase 0/voice_clarity_detection/requirements.txt"` |
| `Phase 0.1/testingpretrainedmodels/` | `python "Phase 0.1/testingpretrainedmodels/install_deps.py"` |
| `Procturing In AI-Interview/` | `pip install -r "Procturing In AI-Interview/requirements.txt"` |

---

## 3. Ollama models to pull

The Interview Engine + Phase 0.2 tests need these local models. `config/models.yaml`
in the Interview Engine is the source of truth; keep this list in sync with it.

```bash
# --- production roles (required) ---
ollama pull qwen3:4b            # interviewer — primary (live path, ~2.6 GB)
ollama pull phi4-mini           # interviewer — fallback 1 (~2.5 GB)
ollama pull llama3.2:3b         # interviewer — fallback 2 (~2 GB)
ollama pull qwen3:8b            # planner + parser/scorer local fallback (~5.2 GB)
ollama pull nomic-embed-text    # embedder — 768-dim, no cloud alternative

# --- interviewer candidates (only if running Phase 0.2 tests 13–18) ---
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull gemma3:4b
ollama pull deepseek-r1:7b
```

---

## 4. API keys & tokens

Copy each `.env.example` to `.env` next to it and fill in:

| Variable | Used by | Needed? | Where |
|----------|---------|---------|-------|
| `ENGINE_GROQ_API_KEY` / `GROQ_API_KEY` | parser + scorer roles (Groq-first, local fallback) | Optional — without it those roles use the local `qwen3:8b` fallback | https://console.groq.com/keys |
| `HF_TOKEN` (HuggingFace) | Phase 0.1 `test_07_pyannote.py` speaker diarization | Only for that test | https://huggingface.co/settings/tokens — also click **Agree** on https://huggingface.co/pyannote/speaker-diarization-3.1 |
| `ENGINE_DATABASE_URL` | Interview Engine retrieval (pgvector) | Only from the retrieval pass onward | Postgres + pgvector, e.g. `postgresql+asyncpg://postgres:dev@localhost:5433/interview` |

`.env` files are git-ignored; `.env.example` files are committed.

---

## 5. Model weights that download themselves

These need **no manual step** — first run pulls them to a local cache (needs
internet the first time):

| Model | Trigger | Cache |
|-------|---------|-------|
| YOLOv8n (`yolov8n.pt`) | `ultralytics` first use | already committed at `Phase 0.1/testingpretrainedmodels/yolov8n.pt` |
| InsightFace `buffalo_l` | `insightface` FaceAnalysis init | `~/.insightface/models/` |
| Silero VAD | `torch.hub.load('snakers4/silero-vad')` | `~/.cache/torch/hub/` |
| faster-whisper `base.en` | first transcription | `~/.cache/huggingface/` |
| Kokoro-82M | first synthesis | `~/.cache/huggingface/` |
| MediaPipe face mesh / pose | first use | bundled with the wheel |

`insightface` also needs the `onnx` **and** `onnxruntime` packages present —
both are in `requirements.txt`. (`onnxruntime-gpu` is optional, for GPU.)

---

## 6. Avatar / lip-sync models (Phase 0.2 tests 19–23) — manual, external

These are **not pip packages**. Each is a full GitHub repo with its own inference
CLI and multi-GB checkpoints, cloned **outside this git repo** at
`C:\Users\<you>\Desktop\lipsync_models\<RepoName>`. The test scripts shell out to
each repo's real `inference.py`. See each `avatar/test_19..23` docstring for the
exact per-model checkpoint commands.

| Test | Model | Clone | VRAM | Notes |
|------|-------|-------|------|-------|
| 19 | **MuseTalk** | `github.com/TMElyralab/MuseTalk` | ~4–6 GB | current pick; needs 6 checkpoint sets via the repo's `download_weights` script |
| 20 | **Wav2Lip** | `github.com/Rudrabha/Wav2Lip` | ~2 GB | lightweight fallback; needs `wav2lip_gan.pth` + `s3fd.pth` |
| 21 | **LivePortrait** | `github.com/KwaiVGI/LivePortrait` | moderate | video-driven only (not audio-native); `huggingface-cli download KwaiVGI/LivePortrait` |
| 22 | **SadTalker** | `github.com/OpenTalker/SadTalker` | moderate–high | diffusion, animates head pose; run its `download_models` script |
| 23 | **InfiniteTalk** | `github.com/MeiGen-AI/InfiniteTalk` | very high | 14B backbone (~82 GB weights); needs its own CUDA 12.1 torch + `flash_attn` + `xformers` — see `avatar/INFINITETALK_COMFYUI_SETUP.md` |

For each: `git clone <url> C:\Users\<you>\Desktop\lipsync_models\<RepoName>` then
`cd` in and `pip install -r requirements.txt` (ideally in a **separate venv** —
several pin old torch versions that conflict with this repo's).

Shared inputs for these tests:
- driving audio: run `text_to_speech/test_09_tts_kokoro.py` once to produce
  `text_to_speech/tts_output/round_2.wav`
- source face: drop `avatar/assets/source_face.jpg` (or `.mp4`); LivePortrait
  also needs `avatar/assets/driving_video.mp4` — see `avatar/assets/README.md`

---

## 7. Web frontends (Node)

```bash
# Interview Engine web app
cd "Interview Engine/web" && npm install && npm run dev      # :5173

# Phase 1 frontend (Vite + React + three.js)
cd "Phase 1/frontend" && npm install && npm run dev          # :5173
```

The interview flows need **headphones** — the pre-flight check enforces it so the
interviewer's TTS voice doesn't leak into the mic.

---

## 8. Verify

| Phase | Command | Expect |
|-------|---------|--------|
| Interview Engine | `cd "Interview Engine" && pytest` | 68 fast tests pass (no models) |
| Interview Engine (slow) | `pytest -m slow` | +4 that load real Silero / whisper / Kokoro (~1 min) |
| Phase 0.2 full suite | `cd "Phase 0.2/testing_engine_models" && python test_12_full_pipeline.py` | PASS/FAIL table; avatar rows FAIL until section 6 is done |
| Phase 0.1 proctoring | `cd "Phase 0.1/testingpretrainedmodels" && python test_01_yolov8.py` | webcam window, person boxes |
| Phase 1 backend | `cd "Phase 1/backend" && uvicorn main:app --reload` | starts on `:8000` |

See `Phase 0.2/testing_engine_models/MODELS_OVERVIEW.md` for a per-model
breakdown of what each test checks and the known-good state.
