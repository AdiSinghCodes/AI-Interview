# Phase 2: Posture Detection

Detects candidate posture violations using full-body pose estimation. Monitors for slouching, forward head position, leaning, and overall spinal misalignment.

## Features

- **Slouching Detection**: Detects rounded shoulders, dropped shoulders, and curved spine
- **Forward Head Position**: Detects when head is positioned too far forward (common in desk work)
- **Leaning Detection**: Detects when body is tilted left/right (asymmetrical posture)
- **Spinal Alignment**: Monitors overall spine-to-hip alignment for proper upright posture
- **No-Pose Detection**: Warns when candidate's body is completely out of frame
- **Progressive Warning System**: 5-warning system with visual feedback

## How It Works

### Pose Estimation
Uses MediaPipe Pose to detect 33 body landmarks:
- Head (nose, eyes, ears)
- Shoulders and arms
- Spine and hips
- Legs and feet

### Detection Logic

1. **Slouching** (sustained >2 seconds)
   - Spine angle > 35°
   - Shoulder height difference > 10% of frame
   - Triggers warning after ~2 seconds

2. **Forward Head** (sustained >2 seconds)
   - Head center > 15% of frame width away from shoulder center
   - Triggers warning after ~2 seconds

3. **Leaning** (sustained >2 seconds)
   - Shoulder/hip asymmetry > 3%
   - Triggers warning after ~2 seconds

4. **Overall Misalignment** (sustained >4 seconds)
   - Shoulder-hip distance > 30% of frame
   - Triggers warning after ~4 seconds

5. **No-Pose Detection** (sustained >3 seconds) ⭐ NEW
   - Body/face completely out of frame
   - Triggers warning after ~3 seconds

### Warning System

- **0 Warnings**: ✓ GREEN - Proper posture
- **1-2 Warnings**: 🟠 ORANGE - Some slouching detected
- **3-4 Warnings**: 🔴 RED - Multiple posture violations
- **5 Warnings**: ❌ RED - **INTERVIEW TERMINATED**

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py
```

### Key Controls
- **q**: Quit the demo

### Expected Behavior

**Good Posture Test:**
1. Sit upright with shoulders back
2. Keep head naturally aligned with shoulders
3. Keep body centered in frame
4. Status should show ✓ GREEN with 0 warnings

**Bad Posture Test (Slouching):**
1. Round your shoulders forward
2. Drop your chin
3. After ~2 seconds → ⚠ SLOUCHING warning
4. Status shows 🟠 ORANGE

**Forward Head Test:**
1. Extend your head forward significantly
2. After ~2 seconds → ⚠ Forward Head warning
3. Status shows 🟠 ORANGE

**Leaning Test:**
1. Tilt your body to one side
2. After ~2 seconds → ⚠ LEANING warning
3. Status shows 🟠 ORANGE

**Multiple Violations:**
1. Combine slouching + leaning
2. After ~4 seconds → Status shows 🔴 RED with higher warning count
3. After 5 violations → INTERVIEW TERMINATED

## Thresholds

All thresholds can be adjusted in `PostureDetector.__init__()`:

```python
self.slouching_threshold = 60  # 2 seconds
self.forward_head_threshold = 60  # 2 seconds
self.leaning_threshold = 60  # 2 seconds
self.overall_threshold = 120  # 4 seconds
self.no_pose_threshold = 90  # 3 seconds (NEW)
self.max_warnings = 5
```

## Output Statistics

When you quit (press 'q'), the demo prints:
```
Total frames processed: XXXX
Final Warning Count: X/5
Posture Statistics:
  GOOD: XXXX frames (XX.X%)
  SLIGHTLY_SLOUCHED: XXXX frames (XX.X%)
  SLOUCHED: XXXX frames (XX.X%)
  OVERALL_MISALIGNED: XXXX frames (XX.X%)
```

## Integration with Other Phases

Phase 2 works independently but can be combined with:
- **Phase 0**: Eye Contact Detection
- **Phase 1**: Face Orientation Detection
- **Phase 3**: Object/Multi-person Detection (planned)

## Troubleshooting

**Issue: No pose detected**
- Ensure full body is in frame (at least from shoulders down)
- Improve lighting
- Keep camera steady

**Issue: Too many false positives**
- Increase thresholds in `posture_model.py`
- Ensure you're at typical interview distance from camera

**Issue: Warnings not triggering**
- Check threshold values (default: 2 seconds for individual violations)
- Ensure violations are sustained long enough
