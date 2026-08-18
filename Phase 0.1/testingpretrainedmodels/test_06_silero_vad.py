"""
TEST 6: Silero VAD — Voice Activity Detection
===============================================
What it shows:
  • Real-time voice probability (0.0 to 1.0)
  • Speech vs silence detection
  • Unusual silence detection (candidate frozen)
  • Short utterance detection (just saying "yes" to coach)

No GPU needed. Tiny model (<2MB). Runs at 1000+ FPS equivalent.

Requirements:
  pip install silero-vad sounddevice numpy

Controls:
  Q  - Quit
  S  - Screenshot of current state
"""

import time
import threading
import numpy as np
from collections import deque

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import sounddevice as sd
except ImportError:
    print("❌  sounddevice not installed. Run:  pip install sounddevice")
    exit(1)

try:
    import cv2
except ImportError:
    print("❌  opencv not installed. Run:  pip install opencv-python")
    exit(1)

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch
    VAD_AVAILABLE = True
    print("✅  Silero VAD (pip package) found")
except ImportError:
    VAD_AVAILABLE = False
    # Try torch hub fallback
    try:
        import torch
        print("⏳  Loading Silero VAD from torch hub…")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        VAD_AVAILABLE = True
        print("✅  Silero VAD loaded from hub")
    except Exception as e:
        print(f"⚠  Silero VAD not available ({e})")
        print("   Install: pip install silero-vad")
        print("   Running in DEMO mode with volume-based detection instead\n")
        VAD_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
CHUNK_SIZE   = 512          # ~32ms at 16kHz (Silero requirement)
VAD_THRESH   = 0.5
SILENCE_FLAG = 8.0          # flag if silent > 8 seconds
SHORT_UTTE   = 0.5          # flag if utterance < 0.5 seconds

class AudioState:
    def __init__(self):
        self.voice_prob   = 0.0
        self.is_speaking  = False
        self.speech_start = None
        self.silence_start = time.time()
        self.silence_dur  = 0.0
        self.utterance_dur = 0.0
        self.prob_history  = deque(maxlen=200)
        self.flags         = deque(maxlen=10)
        self.total_speech  = 0.0
        self.total_silence = 0.0
        self.lock          = threading.Lock()

state = AudioState()

def volume_vad(chunk):
    """Fallback VAD using RMS volume."""
    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
    return min(1.0, rms / 1000.0)

def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio chunk."""
    global model
    chunk = indata[:, 0]  # mono
    chunk_int16 = (chunk * 32767).astype(np.int16)

    if VAD_AVAILABLE:
        try:
            tensor = torch.FloatTensor(chunk).unsqueeze(0)
            prob   = float(model(tensor, SAMPLE_RATE).item())
        except Exception:
            prob = volume_vad(chunk_int16)
    else:
        prob = volume_vad(chunk_int16)

    now = time.time()
    with state.lock:
        state.voice_prob = prob
        state.prob_history.append(prob)
        speaking = prob >= VAD_THRESH

        if speaking and not state.is_speaking:
            state.is_speaking  = True
            state.speech_start = now
            state.silence_dur  = now - (state.silence_start or now)
            if state.silence_dur > SILENCE_FLAG:
                state.flags.append({
                    'type': 'long_silence',
                    'dur':  state.silence_dur,
                    'time': now
                })

        elif not speaking and state.is_speaking:
            state.is_speaking  = False
            dur = now - (state.speech_start or now)
            state.utterance_dur = dur
            state.total_speech += dur
            if dur < SHORT_UTTE:
                state.flags.append({
                    'type': 'short_utterance',
                    'dur':  dur,
                    'time': now
                })
            state.silence_start = now

        if not state.is_speaking:
            state.total_silence += frames / SAMPLE_RATE

def main():
    global model

    if VAD_AVAILABLE:
        try:
            model = load_silero_vad()
        except Exception:
            # hub version already loaded as `model`
            pass

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )

    W, H = 800, 500
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    print("\n🎤  Microphone started. Speak to test VAD.")
    print("   Q=Quit  S=Screenshot\n")

    with stream:
        while True:
            canvas[:] = (15, 15, 25)

            with state.lock:
                prob     = state.voice_prob
                speaking = state.is_speaking
                history  = list(state.prob_history)
                flags    = list(state.flags)
                t_speech  = state.total_speech
                t_silence = state.total_silence
                utt_dur   = state.utterance_dur

            # ── Title ─────────────────────────────────────────────────────────
            cv2.putText(canvas, "TEST 6: Silero VAD — Voice Activity Detection",
                        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 255), 1)
            mode_str = "Silero VAD" if VAD_AVAILABLE else "Volume fallback"
            cv2.putText(canvas, f"Mode: {mode_str}  |  Threshold: {VAD_THRESH}",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # ── Voice probability bar ──────────────────────────────────────────
            bx, by, bw, bh = 10, 80, W - 20, 40
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
            bar_w = int(bw * prob)
            bar_color = (50, 220, 80) if speaking else (80, 120, 80)
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bh), bar_color, -1)
            # Threshold line
            tx = bx + int(bw * VAD_THRESH)
            cv2.line(canvas, (tx, by - 4), (tx, by + bh + 4), (255, 255, 100), 2)
            cv2.putText(canvas, f"Voice Probability: {prob:.3f}",
                        (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, f"{'SPEAKING' if speaking else 'SILENT'}",
                        (bx + bw - 120, by + 28), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        (50, 220, 80) if speaking else (150, 80, 80), 2)

            # ── Waveform history ───────────────────────────────────────────────
            wave_y = 160
            wave_h = 80
            cv2.rectangle(canvas, (10, wave_y), (W - 10, wave_y + wave_h), (25, 25, 40), -1)
            cv2.putText(canvas, "Voice probability history →",
                        (12, wave_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 130), 1)

            if history:
                pts = []
                for i, p in enumerate(history):
                    x = 10 + int(i / len(history) * (W - 20))
                    y = wave_y + wave_h - int(p * wave_h)
                    pts.append((x, y))
                # Draw threshold line
                th_y = wave_y + wave_h - int(VAD_THRESH * wave_h)
                cv2.line(canvas, (10, th_y), (W - 10, th_y), (80, 80, 30), 1)
                # Draw waveform
                for i in range(1, len(pts)):
                    col = (50, 200, 80) if history[i] >= VAD_THRESH else (80, 80, 120)
                    cv2.line(canvas, pts[i-1], pts[i], col, 1)

            # ── Stats ──────────────────────────────────────────────────────────
            sy = 270
            cv2.rectangle(canvas, (10, sy), (380, sy + 90), (20, 20, 35), -1)
            cv2.putText(canvas, f"Total speech:  {t_speech:.1f}s",
                        (20, sy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 200, 100), 1)
            cv2.putText(canvas, f"Total silence: {t_silence:.1f}s",
                        (20, sy + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 200), 1)
            cv2.putText(canvas, f"Last utterance: {utt_dur:.2f}s",
                        (20, sy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 100), 1)

            # ── Flags ──────────────────────────────────────────────────────────
            fx = 400
            cv2.rectangle(canvas, (fx, sy), (W - 10, sy + 90), (30, 15, 15), -1)
            cv2.putText(canvas, "RECENT FLAGS:", (fx + 8, sy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 150, 150), 1)
            if flags:
                for i, fl in enumerate(reversed(flags[-3:])):
                    elapsed = time.time() - fl['time']
                    msg = (f"Long silence: {fl['dur']:.1f}s" if fl['type'] == 'long_silence'
                           else f"Short utterance: {fl['dur']:.2f}s")
                    cv2.putText(canvas, f"{msg} ({elapsed:.0f}s ago)",
                                (fx + 8, sy + 42 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 120, 120), 1)
            else:
                cv2.putText(canvas, "None", (fx + 8, sy + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 120, 80), 1)

            # ── Help ───────────────────────────────────────────────────────────
            cv2.putText(canvas, "Proctoring use: flag if silent >8s (frozen) or utterance <0.5s (yes/no to coach)",
                        (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 130), 1)
            cv2.putText(canvas, "Q=Quit  S=Screenshot",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 100), 1)

            cv2.imshow("Proctoring — Test 6: Silero VAD", canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"vad_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, canvas)
                print(f"📸  Saved {fname}")

    cv2.destroyAllWindows()
    print("\n✅  Done.")

if __name__ == "__main__":
    main()