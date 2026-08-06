"""
TEST 7: PyAnnote — Speaker Diarization
========================================
What it shows:
  • Detects how many distinct speakers are in the audio
  • Flags if more than 1 speaker is detected (coaching)
  • Shows speaker timeline

⚠  SETUP REQUIRED (free):
  1.  pip install pyannote.audio sounddevice
  2.  Sign up at huggingface.co (free)
  3.  Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
      Click "Agree and access repository"
  4.  Generate token: https://huggingface.co/settings/tokens
  5.  Set environment variable:
      export HF_TOKEN=hf_your_token_here
      (Windows: set HF_TOKEN=hf_your_token_here)

Controls:
  Q  - Quit
  S  - Screenshot
"""

import os
import time
import threading
import numpy as np
from collections import deque

try:
    import cv2
except ImportError:
    print("❌  pip install opencv-python")
    exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("❌  pip install sounddevice")
    exit(1)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("⚠  HF_TOKEN not set.")
    print("   Set it with:  export HF_TOKEN=hf_your_token_here")
    print("   Get token at: https://huggingface.co/settings/tokens\n")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠  pyannote.audio not installed. Run:  pip install pyannote.audio")
    print("   Running in DEMO mode (simulating diarization output)\n")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_SECONDS  = 3          # process 3-second windows
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_SECONDS
SECOND_SPEAKER_THRESH = 2.0  # flag if second speaker > 2 seconds

class DiarizationState:
    def __init__(self):
        self.audio_buffer  = deque(maxlen=CHUNK_SAMPLES * 2)
        self.speakers      = {}   # {speaker_id: cumulative_seconds}
        self.current_chunk_speakers = []
        self.flags         = deque(maxlen=20)
        self.processing    = False
        self.last_result   = "Waiting for audio…"
        self.num_speakers  = 0
        self.lock          = threading.Lock()
        self.waveform      = deque(maxlen=400)

state = DiarizationState()
pipeline = None

def load_pipeline():
    global pipeline
    if not PYANNOTE_AVAILABLE or not HF_TOKEN:
        return False
    try:
        print("⏳  Loading PyAnnote pipeline (downloads ~1GB on first run) …")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        print("✅  PyAnnote loaded")
        return True
    except Exception as e:
        print(f"❌  Failed to load PyAnnote: {e}")
        return False

def process_audio_chunk(audio):
    """Run diarization on a 3-second audio chunk."""
    if pipeline is None:
        # Demo mode: simulate output
        time.sleep(0.2)
        speakers = ["SPEAKER_00"]
        if np.random.random() < 0.1:
            speakers.append("SPEAKER_01")
        return speakers

    try:
        import torch
        from pyannote.core import Segment
        tensor = torch.FloatTensor(audio).unsqueeze(0)
        waveform = {"waveform": tensor, "sample_rate": SAMPLE_RATE}
        diarization = pipeline(waveform)

        speakers_in_chunk = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_in_chunk.add(speaker)

        return list(speakers_in_chunk)
    except Exception as e:
        return ["SPEAKER_00"]

def diarization_worker():
    """Background thread: processes audio every 3 seconds."""
    while True:
        time.sleep(CHUNK_SECONDS)
        with state.lock:
            if len(state.audio_buffer) < CHUNK_SAMPLES:
                continue
            audio = np.array(list(state.audio_buffer)[-CHUNK_SAMPLES:], dtype=np.float32)
            state.processing = True

        speakers = process_audio_chunk(audio)
        now = time.time()

        with state.lock:
            state.processing = False
            state.current_chunk_speakers = speakers
            state.num_speakers = len(speakers)
            for s in speakers:
                state.speakers[s] = state.speakers.get(s, 0) + CHUNK_SECONDS

            # Flag if second speaker
            if len(speakers) > 1:
                second_time = sum(v for k, v in state.speakers.items() if k != "SPEAKER_00")
                if second_time >= SECOND_SPEAKER_THRESH:
                    state.flags.append({
                        'type': 'second_speaker',
                        'speakers': speakers,
                        'time': now,
                        'duration': second_time
                    })
            state.last_result = ", ".join(speakers)

def audio_callback(indata, frames, time_info, status):
    chunk = indata[:, 0]
    with state.lock:
        state.audio_buffer.extend(chunk.tolist())
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        state.waveform.append(rms)

def main():
    pipeline_loaded = load_pipeline()
    if not pipeline_loaded:
        print("ℹ  Running in DEMO MODE — simulating diarization\n")

    # Start diarization worker thread
    worker = threading.Thread(target=diarization_worker, daemon=True)
    worker.start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=1024,
        callback=audio_callback
    )

    W, H = 860, 520
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    print("\n🎤  Microphone started.")
    print("ℹ  Speak for 3 seconds, then have someone else speak")
    print("   to test second-speaker detection.\n")
    print("   Q=Quit  S=Screenshot\n")

    SPEAKER_COLORS = [
        (50,  200, 50),   # green
        (50,  50,  220),  # red
        (255, 165, 50),   # orange
        (200, 50,  200),  # purple
    ]

    with stream:
        while True:
            canvas[:] = (15, 15, 25)

            with state.lock:
                speakers       = dict(state.speakers)
                current        = list(state.current_chunk_speakers)
                flags          = list(state.flags)
                processing     = state.processing
                last_result    = state.last_result
                num_spk        = state.num_speakers
                waveform       = list(state.waveform)

            # ── Title ──────────────────────────────────────────────────────────
            mode = "PyAnnote" if pipeline_loaded else "DEMO MODE"
            cv2.putText(canvas, f"TEST 7: Speaker Diarization  [{mode}]",
                        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 255), 1)
            cv2.putText(canvas, f"Window: {CHUNK_SECONDS}s  |  Second speaker flag: >{SECOND_SPEAKER_THRESH}s",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # ── Waveform ───────────────────────────────────────────────────────
            wy, wh = 70, 50
            cv2.rectangle(canvas, (10, wy), (W - 10, wy + wh), (25, 25, 40), -1)
            if waveform:
                max_rms = max(waveform) + 1e-6
                for i in range(1, len(waveform)):
                    x = 10 + int(i / len(waveform) * (W - 20))
                    amp = int(waveform[i] / max_rms * (wh // 2))
                    cy_pt = wy + wh // 2
                    cv2.line(canvas, (x, cy_pt - amp), (x, cy_pt + amp), (80, 160, 80), 1)

            # ── Processing indicator ───────────────────────────────────────────
            if processing:
                cv2.putText(canvas, "⏳ Analyzing 3s window…",
                            (W - 220, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 100), 1)

            # ── Current result ─────────────────────────────────────────────────
            ry = 145
            cv2.rectangle(canvas, (10, ry), (W - 10, ry + 55), (20, 20, 35), -1)
            cv2.putText(canvas, "Last 3s window:", (15, ry + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 180), 1)
            result_col = (50, 50, 220) if num_spk > 1 else (50, 200, 80)
            cv2.putText(canvas, f"Speakers detected: {num_spk}  ({last_result})",
                        (15, ry + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_col, 1)

            # ── Speaker cumulative times ───────────────────────────────────────
            ty = 220
            cv2.rectangle(canvas, (10, ty), (440, ty + 30 + len(speakers) * 36), (20, 20, 35), -1)
            cv2.putText(canvas, "Cumulative speaker time:", (15, ty + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 180), 1)
            for i, (spk, secs) in enumerate(sorted(speakers.items())):
                col = SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
                bx = 15
                by2 = ty + 30 + i * 36
                bw = int(min(1.0, secs / 60.0) * 360)
                cv2.rectangle(canvas, (bx, by2), (bx + 380, by2 + 20), (40, 40, 40), -1)
                cv2.rectangle(canvas, (bx, by2), (bx + bw, by2 + 20), col, -1)
                label = "YOU" if i == 0 else f"EXTRA VOICE #{i}"
                cv2.putText(canvas, f"{label}: {secs:.1f}s",
                            (bx + 2, by2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # ── Flags ──────────────────────────────────────────────────────────
            fy = 350
            cv2.rectangle(canvas, (10, fy), (W - 10, H - 60), (30, 15, 15), -1)
            cv2.putText(canvas, "FLAGS:", (15, fy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 130, 130), 1)
            if flags:
                for i, fl in enumerate(reversed(flags[-4:])):
                    elapsed = time.time() - fl['time']
                    msg = f"⚠ Second speaker detected — {fl['duration']:.1f}s total ({elapsed:.0f}s ago)"
                    cv2.putText(canvas, msg, (15, fy + 44 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 100, 100), 1)
            else:
                cv2.putText(canvas, "No violations detected", (15, fy + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 150, 80), 1)

            # ── Alert banner ───────────────────────────────────────────────────
            if num_spk > 1:
                cv2.rectangle(canvas, (0, H - 52), (W, H), (0, 0, 170), -1)
                cv2.putText(canvas, "⚠  MULTIPLE SPEAKERS — Possible coaching detected",
                            (10, H - 18), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(canvas, "Q=Quit  S=Screenshot",
                            (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 100), 1)

            cv2.imshow("Proctoring — Test 7: Speaker Diarization", canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"diarization_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, canvas)
                print(f"📸  Saved {fname}")

    cv2.destroyAllWindows()
    print("\n✅  Done.")

if __name__ == "__main__":
    main()