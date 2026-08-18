# Eye Contact Detection Model

A real-time eye contact detection system using MediaPipe Face Mesh and OpenCV. This model detects where the candidate is looking and tracks eye gaze direction.

## Features

✅ **Real-time Eye Gaze Detection** - Detects if eyes are looking at camera, left, right, up, or down
✅ **Gaze Direction Metrics** - Horizontal and vertical gaze ratios (0-1)
✅ **Head Pose Estimation** - Pitch, yaw, and roll angles
✅ **Iris Tracking** - Tracks iris center position
✅ **Face Detection** - Detects and tracks face in frame
✅ **Visual Feedback** - Real-time visualization with gaze direction on screen

## Project Structure

```
eye_contact_detection/
├── eye_contact_model.py    # Eye detection model using MediaPipe
├── demo.py                 # Real-time demo script
└── requirements.txt        # Python dependencies
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Required packages:**
- opencv-python (4.8.1.78)
- mediapipe (0.10.5)
- numpy (1.24.3)

## Usage

### Run the Demo

```bash
cd eye_contact_detection
python demo.py
```

**Controls:**
- Press `q` to quit the demo

### What You'll See

The demo displays in real-time:

1. **Left Eye Gaze**
   - Direction: CENTER, LEFT, RIGHT, UP, or DOWN
   - Horizontal ratio: 0 (far left) → 0.5 (center) → 1 (far right)
   - Vertical ratio: 0 (up) → 0.5 (center) → 1 (down)

2. **Right Eye Gaze** (same metrics as left eye)

3. **Head Pose**
   - Yaw: Head rotation left/right
   - Pitch: Head rotation up/down
   - Roll: Head tilt

4. **Visual Markers**
   - Green circle at iris position
   - Green crosshair at screen center

5. **Statistics on Exit**
   - Total frames processed
   - Breakdown of gaze directions

## Testing & Observation

This is a **demo/testing phase**, so we're observing:

1. **Eye Drift Patterns** - How much eyes move left/right/up/down during normal viewing
2. **Gaze Direction Accuracy** - Test by looking in different directions
3. **Head Pose Angles** - Observe how head movement affects detection
4. **Performance** - Frame rate and detection reliability

## Next Steps

After testing:
1. Analyze the gaze data to determine appropriate thresholds
2. Define what constitutes "cheating" behavior (e.g., consistent left gaze for >2 seconds)
3. Implement warning system based on observed patterns
4. Move to next detection component (face orientation)

## Model Details

### MediaPipe Face Mesh
- Detects 468 3D face landmarks
- Real-time performance (~20-30 FPS on CPU)
- High accuracy for gaze direction

### Eye Landmarks Used
- Left/Right iris centers (5 points each)
- Left/Right eye corners and lids (16 points each)

### Gaze Direction Classification
- **CENTER**: Horizontal 0.35-0.65, Vertical 0.35-0.65
- **LEFT**: Horizontal < 0.35
- **RIGHT**: Horizontal > 0.65
- **UP**: Vertical < 0.35
- **DOWN**: Vertical > 0.65

## Performance Notes

- **CPU Usage**: ~15-25%
- **Memory**: ~200-300 MB
- **FPS**: 25-30 FPS on average CPU
- **Latency**: ~30-40ms per frame

## Troubleshooting

**Issue**: No face detected
- Solution: Ensure good lighting, face is clearly visible, and looking at camera

**Issue**: Low FPS
- Solution: Reduce frame resolution or close other applications

**Issue**: Camera not opening
- Solution: Check camera ID (default is 0), or try different camera

## Files Overview

### `eye_contact_model.py`
- **EyeContactDetector class**: Main detection logic
- Methods:
  - `process_frame()`: Process single frame
  - `_calculate_gaze_direction()`: Calculate gaze for one eye
  - `_estimate_head_pose()`: Estimate head angles
  - `_get_iris_position()`: Get iris center
  - `_get_face_center()`: Get face center

### `demo.py`
- **EyeContactDemo class**: Real-time visualization
- `run()`: Main demo loop
- `_draw_visualization()`: Draw output on frame
- `cleanup()`: Print statistics and cleanup

## License

Project for interview practice avatar system
