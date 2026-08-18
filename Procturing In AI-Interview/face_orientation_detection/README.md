# Face Orientation Detection Model

A real-time face orientation detection system using OpenCV. Monitors if the candidate's face is properly aligned and oriented towards the camera.

## Features

✅ **Head Orientation Detection** - Detects Yaw, Pitch, and Roll angles
✅ **Face Alignment Check** - Ensures face is centered in frame
✅ **Face Size Monitoring** - Warns if face is too small/far from camera
✅ **Real-time Feedback** - Shows alignment status (ALIGNED, SLIGHTLY_OFF, MISALIGNED)
✅ **Progressive Warnings** - Accumulates warnings for repeated misalignment
✅ **Interview Termination** - Rejects after 5 warnings

## Project Structure

```
face_orientation_detection/
├── face_orientation_model.py    # Face orientation detection model
├── demo.py                      # Real-time demo script
└── README.md                    # Documentation
```

## Installation

Install dependencies (same as Phase 0):
```bash
pip install opencv-python numpy
```

## Usage

### Run the Demo

```bash
cd face_orientation_detection
python demo.py
```

**Controls:**
- Press `q` to quit

## What You'll See

The demo displays in real-time:

### 1. **Alignment Status**
- ✓ **ALIGNED** (Green) - Face properly positioned
- ⚠️ **SLIGHTLY_OFF** (Orange) - Minor adjustment needed
- ❌ **MISALIGNED** (Red) - Significant deviation

### 2. **Orientation Angles**
- **Yaw**: Head left/right rotation (-45° to +45°)
- **Pitch**: Head up/down tilt (-45° to +45°)
- **Roll**: Head tilt side-to-side

### 3. **Face Metrics**
- **Face Size**: % of frame occupied by face
- **Position**: Horizontal and vertical offset from center
- **Reason**: Explanation of any misalignment

### 4. **Warning System**
- Green border → No warnings
- Orange border → Warnings accumulated (1-4)
- Red border + overlay → Interview terminated (5 warnings)

## Detection Thresholds

The model uses these thresholds to determine misalignment:

```python
self.yaw_threshold = 25°           # Head turn left/right
self.pitch_threshold = 25°         # Head tilt up/down
self.face_size_threshold = 15%     # Minimum % of frame
self.face_position_threshold = 20% # Maximum offset from center
self.misalignment_threshold = 60   # Frames (~2 seconds)
```

## Testing & Observation

Test these scenarios:

1. **Look straight at camera** → "ALIGNED" (Green)
2. **Turn head left** → "MISALIGNED" (Red)
3. **Turn head right** → "MISALIGNED" (Red)
4. **Tilt head up/down** → "MISALIGNED" (Red)
5. **Move face away** → "MISALIGNED" (Red)
6. **Intentionally misalign for 2+ seconds** → Warning triggered
7. **After 5 violations** → Interview terminated

## Warning System

Warnings are triggered when:
- Head is turned > 25° from center (Yaw)
- Head is tilted > 25° up/down (Pitch)
- Face is off-center > 20%
- Face occupies < 15% of frame
- Misalignment sustained for 2+ seconds

After each violation that lasts 2+ seconds:
- Warning counter increments
- Visual indicator updates
- At 5 warnings → Interview rejected

## Output on Exit

When you quit (press 'q'):
```
Total frames processed: XXXX
Final Warning Count: X/5

✓ Interview Status: X warnings remaining

Alignment Statistics:
  ALIGNED: XXX frames (XX.X%)
  SLIGHTLY_OFF: XX frames (X.X%)
  MISALIGNED: XX frames (X.X%)
```

## Files Overview

### `face_orientation_model.py`
- **FaceOrientationDetector class**: Main detection logic
- Methods:
  - `process_frame()`: Process single frame
  - `_calculate_orientation()`: Calculate Yaw, Pitch, Roll
  - `_check_alignment()`: Determine alignment status
  - `reset_warnings()`: Reset warning counter

### `demo.py`
- **FaceOrientationDemo class**: Real-time visualization
- `run()`: Main demo loop
- `_draw_visualization()`: Draw orientation info
- `_draw_warnings()`: Draw warning indicators
- `cleanup()`: Print statistics

## Next Steps

After testing:
1. Adjust thresholds if needed for your requirements
2. Integrate with Phase 0 (Eye Contact Detection)
3. Move to Phase 2 (Posture Detection)

## Performance

- **FPS**: 25-30 FPS on average CPU
- **Memory**: ~100-150 MB
- **Latency**: ~30-40ms per frame
