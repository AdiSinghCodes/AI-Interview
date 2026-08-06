# Testing Analysis Report - Proctoring Test Suite

## **Status: Dependencies Installed ✅**

### Installation Results:
- ✅ OpenCV (cv2)
- ✅ NumPy (downgraded to 1.26.4 for compatibility)
- ✅ Pillow
- ✅ YOLOv8 (ultralytics)
- ✅ MediaPipe
- ✅ InsightFace
- ✅ ONNX Runtime (CPU)
- ✅ SoundDevice
- ✅ Silero VAD
- ⏳ PyAnnote (optional - requires HuggingFace token)

---

## **About These Tests**

**Important:** All test files (test_01 through test_09) are designed to run **INTERACTIVELY** with:
- **Live Webcam Input** — They open your webcam and process video in real-time
- **Real-time Display** — Shows bounding boxes, landmarks, scores, etc. on your screen
- **Keyboard Controls** — Q to quit, S to screenshot, etc.

### **Why They Can't Be Automated**
These tests are meant to be interactive demos where you:
1. Point your webcam at yourself
2. See live AI analysis in real-time
3. Test different scenarios (phone in frame, looking away, etc.)

---

## **Test-by-Test Guide**

### **TEST 1: `test_01_yolov8.py` ✅ (Ready)**

**What It Does:**
- Opens your webcam
- Detects objects: phones, laptops, books, people, keyboards, monitors
- Assigns **severity scores** (phone = 10, book = 4, etc.)
- Shows **Integrity Score** (0-100)

**How to Run:**
```bash
python test_01_yolov8.py
```

**What You'll See:**
- Green/cyan/red boxes around detected objects
- Object names with confidence scores
- Integrity score in top-right
- FPS counter

**Controls:**
- `Q` — Quit
- `S` — Save screenshot
- `M` — Toggle between nano/medium models

**Detection Classes:**
```
person (blue)      → Integrity: 0
tv (green)         → Integrity: -5
laptop (cyan)      → Integrity: -7
mouse (yellow)     → Integrity: -1
keyboard (purple)  → Integrity: -2
cell phone (red)   → Integrity: -10 (most suspicious!)
book (orange)      → Integrity: -4
```

---

### **TEST 2: `test_02_mediapipe_facemesh.py` ✅ (Ready)**

**What It Does:**
- Detects 468 facial landmarks
- Calculates **Eye Aspect Ratio (EAR)** — detects blinking
- Detects **mouth open/closed**
- Estimates **head pose** (yaw/pitch/roll)
- Counts **faces** in view
- Shows **iris position**

**How to Run:**
```bash
python test_02_mediapipe_facemesh.py
```

**What You'll See:**
- 468 green dots on your face (facial landmarks)
- Eye blink detection with EAR value
- Mouth open/closed indicator
- Head rotation angles
- Face count

**Controls:**
- `Q` — Quit
- `S` — Screenshot
- `L` — Toggle landmark dots on/off

**How Blinking Works (EAR - Eye Aspect Ratio):**
- Normal eyes: EAR ≈ 0.2-0.3
- Blinking: EAR < 0.15
- Flags prolonged non-blinking (suspicious)

---

### **TEST 3: `test_03_insightface.py` ✅ (Ready)**

**What It Does:**
- Captures your **baseline face** at start
- Compares every frame with baseline
- Calculates **cosine similarity** (0.0 to 1.0)
- Alerts if **different person** detected (similarity < 0.5)

**How to Run:**
```bash
python test_03_insightface.py
```

**Setup (First Time):**
1. Run the script
2. Press `SPACE` to capture baseline (do this 5 times for best accuracy)
3. Then it monitors every frame

**What You'll See:**
- Face detection box
- Similarity score (0-100%)
- Green = same person, Red = different person
- 512D face embeddings computed per frame

**Controls:**
- `SPACE` — Capture baseline
- `Q` — Quit
- `S` — Screenshot
- `R` — Reset baseline

**Critical Threshold:**
- `> 0.5` = Same person ✅
- `< 0.5` = Different person ❌ (cheating alert)

---

### **TEST 4: `test_04_gaze.py` ✅ (Ready)**

**What It Does:**
- Estimates **gaze direction** (where you're looking)
- Two modes available:
  - **Mode A (default)**: MediaPipe iris tracking (~15° MAE)
  - **Mode B**: L2CS-Net (~9° MAE, requires manual setup)

**How to Run:**
```bash
# Default (MediaPipe)
python test_04_gaze.py

# Advanced (if L2CS weights downloaded)
python test_04_gaze.py --mode l2cs
```

**What You'll See:**
- Gaze vector arrows from your eyes
- Yaw angle (left/right looking)
- Pitch angle (up/down looking)
- Alerts if gaze exceeds thresholds for 3+ seconds

**Thresholds:**
- Yaw > 30° = "Looking too far away"
- Pitch > 25° = "Looking too far down/up"

**Controls:**
- `Q` — Quit
- `S` — Screenshot
- `M` — Toggle mode (if L2CS available)

**Advanced Setup (L2CS-Net):**
```bash
# Clone repo
git clone https://github.com/Ahmednull/L2CS-Net

# Download weights and place in ./weights/l2cs_gaze360.pkl
# Then run: python test_04_gaze.py --mode l2cs
```

---

### **TEST 5: `test_05_headpose.py` ✅ (Ready)**

**What It Does:**
- Measures **head rotation** angles (yaw, pitch, roll)
- Two modes:
  - **Mode A (default)**: MediaPipe + OpenCV solvePnP (instant, accurate)
  - **Mode B**: 6DRepNet (more robust, requires manual setup)

**How to Run:**
```bash
# Default
python test_05_headpose.py

# Advanced
python test_05_headpose.py --mode 6drepnet
```

**What You'll See:**
- 3D coordinate axes on your face
- Rotation angles (degrees)
- Alerts if head turns > thresholds

**Thresholds:**
- Yaw > 30° = "Turned too far"
- Pitch > 25° = "Bent too far"

**Controls:**
- `Q` — Quit
- `S` — Screenshot
- `A` — Toggle 3D axes display

**Head Pose Interpretation:**
```
Yaw = 0°    → Looking straight
Yaw = 30°   → Looking far right
Yaw = -30°  → Looking far left

Pitch = 0°   → Level head
Pitch = 20°  → Looking down
Pitch = -20° → Looking up

Roll = 0°    → Head upright
Roll = 20°   → Head tilted right
Roll = -20°  → Head tilted left
```

---

### **TEST 6: `test_06_silero_vad.py` ✅ (Ready)**

**What It Does:**
- Monitors **audio** from your microphone
- Detects **speech vs silence**
- Calculates **voice probability** (0.0 to 1.0)
- Flags:
  - Prolonged silence (> 8 seconds)
  - Very short utterances (< 0.5s) — someone coaching you?

**How to Run:**
```bash
python test_06_silero_vad.py
```

**What You'll See:**
- Voice probability bar (0-100%)
- Speech detected: YES/NO
- Duration of current speech
- Flags (silence, short utterance)
- Timeline of speech

**Thresholds:**
- Voice Prob > 0.5 = Speaking
- Silence > 8s = "Unusual silence - frozen?"
- Utterance < 0.5s = "Too short - coaching signal?"

**Controls:**
- `Q` — Quit
- `S` — Screenshot

**How It Works:**
- Uses Silero VAD (tiny neural network)
- Processes 512-sample chunks (32ms at 16kHz)
- No GPU needed
- Super fast (1000+ FPS equivalent)

---

### **TEST 7: `test_07_pyannote.py` ✅ (Ready - Needs Setup)**

**What It Does:**
- Detects **how many speakers** are in the audio
- **Speaker diarization** — identifies distinct speakers
- Flags if > 1 speaker detected (someone coaching you!)

**Setup Required:**
1. Install: `pip install pyannote.audio`
2. Sign up free at: https://huggingface.co
3. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
4. Click "Agree and access repository"
5. Get token: https://huggingface.co/settings/tokens
6. Set environment variable:
   ```bash
   # Windows PowerShell:
   $env:HF_TOKEN="hf_your_token_here"
   
   # Windows CMD:
   set HF_TOKEN=hf_your_token_here
   ```

**How to Run:**
```bash
python test_07_pyannote.py
```

**What You'll See:**
- Speaker detection results
- Number of speakers detected
- Timeline showing when each speaker talks
- Alert if 2+ speakers detected

**Thresholds:**
- 1 speaker = OK ✅
- 2+ speakers = CHEATING ❌

---

### **TEST 8: `test_08_browser_events.html` ✅ (Ready)**

**What It Does:**
- Monitors **browser events**
- Detects:
  - Tab switching away from exam
  - Copy/paste attempts
  - Right-click / inspect element
  - Browser focus loss
  - Fullscreen exit

**How to Run:**
1. Open file in browser: `test_08_browser_events.html`
2. Dashboard shows real-time event tracking

**What You'll See:**
- Green dots for normal activity
- Red dots for suspicious activity
- Event log with timestamps
- Integrity score

**Suspicious Events Tracked:**
- `blur` — Switched to another tab
- `copy` — Tried to copy text
- `contextmenu` — Right-clicked (inspect?)
- `visibilitychange` — Lost focus
- `fullscreenchange` — Exited fullscreen

---

### **TEST 9: `test_09_full_pipeline.py` ✅ (Ready)**

**What It Does:**
- Runs ALL models simultaneously:
  - YOLOv8 (objects)
  - MediaPipe (face landmarks)
  - InsightFace (identity verification)
  - Gaze (eye tracking)
  - Head Pose (rotation)

**How to Run:**
```bash
python test_09_full_pipeline.py
```

**What You'll See:**
- All detections on one screen
- Single **Integrity Score** (0-100)
- Multi-threaded processing
- FPS counter

**Scoring Logic:**
```
Base Score = 100

For each detection:
  - Phone detected       → -10
  - Laptop detected      → -7
  - Different person     → -20
  - Eyes closed too long → -5
  - Looking away > 30°   → -8
  - Head turned > 30°    → -5
  - Unusual silence      → -5
  - Multiple speakers    → -15

Final Score = Base - penalties
```

**Controls:**
- `Q` — Quit
- `SPACE` — Capture identity baseline
- `R` — Reset all models
- `S` — Save screenshot + print report

---

## **How to Use These Tests**

### **Option 1: Run Each Test Individually**
```bash
python test_01_yolov8.py          # Object detection
python test_02_mediapipe_facemesh.py  # Face landmarks
python test_03_insightface.py     # Identity verification
python test_04_gaze.py            # Gaze tracking
python test_05_headpose.py        # Head pose
python test_06_silero_vad.py      # Voice activity
python test_07_pyannote.py        # Speaker diarization
```

### **Option 2: Run Full Pipeline**
```bash
python test_09_full_pipeline.py
```

### **Option 3: Test Browser Events**
- Open `test_08_browser_events.html` in your browser
- Try copying text, switching tabs, etc.
- Watch the event log

---

## **Key Findings**

| Test | Status | Notes |
|------|--------|-------|
| Test 1 (YOLOv8) | ✅ Ready | Detects objects with severity scoring |
| Test 2 (MediaPipe Face) | ✅ Ready | 468 landmarks, blink detection, head pose |
| Test 3 (InsightFace) | ✅ Ready | Face identity verification, cosine similarity |
| Test 4 (Gaze) | ✅ Ready | Eye gaze direction tracking |
| Test 5 (Head Pose) | ✅ Ready | Head rotation (yaw/pitch/roll) |
| Test 6 (Silero VAD) | ✅ Ready | Voice activity detection, audio analysis |
| Test 7 (PyAnnote) | ⚠️ Needs Setup | Speaker diarization (requires HF token) |
| Test 8 (Browser) | ✅ Ready | Browser event monitoring (HTML) |
| Test 9 (Full Pipeline) | ✅ Ready | All tests combined with unified score |

---

## **Next Steps**

1. **Choose which test to run first**
2. **Point your webcam at yourself**
3. **Follow the on-screen instructions**
4. **Test various scenarios** (look away, bring phone, etc.)
5. **Document findings**

Would you like to:
- Set up PyAnnote (needs HuggingFace account)?
- Run a specific test?
- Make any modifications to the code?

