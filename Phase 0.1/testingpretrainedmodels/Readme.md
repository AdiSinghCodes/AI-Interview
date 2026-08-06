# Proctoring Model Live Test Suite

Test each pretrained model one by one with your webcam.

## Setup (do this once)

```bash
pip install ultralytics mediapipe insightface onnxruntime opencv-python numpy
pip install silero-vad
# For PyAnnote (needs HuggingFace account):
pip install pyannote.audio
```

## Run each test

```bash
# Test 1: YOLOv8 — Object detection (phones, laptops, people)
python test_01_yolov8.py

# Test 2: MediaPipe Face Mesh — Face landmarks, blink, mouth detection
python test_02_mediapipe_facemesh.py

# Test 3: InsightFace — Face identity verification
python test_03_insightface.py

# Test 4: L2CS-Net — Gaze estimation (requires manual weight download)
python test_04_l2cs_gaze.py

# Test 5: 6DRepNet — Head pose estimation (requires manual weight download)
python test_05_6drepnet.py

# Test 6: Silero VAD — Voice activity detection (audio)
python test_06_silero_vad.py

# Test 7: PyAnnote — Speaker diarization (needs HuggingFace token)
python test_07_pyannote.py

# Test 8: Browser Events — Tab switch / copy detection (open HTML in browser)
# Just open: test_08_browser_events.html in your browser

# Run ALL visual models together (the full pipeline)
python test_09_full_pipeline.py
```

## Controls (same for all scripts)
- `Q` — Quit
- `S` — Save screenshot
- `R` — Reset/recalibrate (where applicable)

## Notes
- Tests 4 & 5 need manual weight downloads (instructions inside each file)
- Test 7 needs a HuggingFace token (free account)
- All other tests auto-download weights on first run