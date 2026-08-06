# Phase 3: Object & Multi-Person Detection

Detects suspicious activities in interview frame: unauthorized participants and hands carrying objects (phone, notes, etc.).

## Features

- **Multiple People Detection**: Detects when 2+ people are in frame (cheating attempt)
- **Hand Carrying Object Detection**: Detects when candidate's hand is holding an object like phone or notes
- **Progressive Warning System**: 5-warning system with visual feedback

## How It Works

### Detection Logic

1. **Multiple People Detection** (sustained >3 seconds)
   - Uses face cascades to count people
   - Triggers warning when 2+ faces detected
   - Warning triggers after ~3 seconds sustained

2. **Hand Carrying Object Detection** (sustained >2 seconds) ⭐ **KEY FEATURE**
   - Detects hands using skin color detection (HSV)
   - Detects objects using edge detection + brightness analysis
   - **Only triggers warning when BOTH are detected simultaneously**
   - Individual hands or objects alone = NO WARNING
   - Only hand + object together = WARNING (indicates cheating attempt)
   - Warning triggers after ~2 seconds sustained

### What Doesn't Trigger Warnings

✅ **Hand alone** (without object) - Normal gestures allowed
✅ **Object alone** (without hand) - Just having paper on desk is okay
✅ **One person in frame** - Normal interview setup

### Warning System

- **0 Warnings**: ✓ GREEN - Clean interview
- **1-2 Warnings**: 🟠 ORANGE - Minor violations
- **3-4 Warnings**: 🔴 RED - Multiple violations
- **5 Warnings**: ❌ RED - **INTERVIEW TERMINATED**

## Usage

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the demo
python demo.py
```

### Key Controls
- **q**: Quit the demo

### Expected Behavior

**Normal Interview (1 person, hands/objects separately):**
- Status: ✓ CLEAN FRAME (green border)
- Warnings: 0/5
- Should stay at 0 warnings

**Multiple People Test:**
1. Position yourself normally (1 person)
2. Have another person enter frame
3. After ~3 seconds → ⚠ MULTIPLE PEOPLE DETECTED
4. Status shows 🟠 ORANGE
5. Warning counter increments to 1/5

**Hand Carrying Object Test:** ⭐ **Main Violation**
1. Sit normally with hand not carrying anything
2. Pick up a phone/paper/note in your hand
3. After ~2 seconds → ⚠ HAND CARRYING OBJECT DETECTED
4. Status shows 🟠 ORANGE
5. Warning counter increments

**Hand Alone (NO WARNING):**
1. Raise hand without holding anything
2. Even after 2+ seconds → ✓ CLEAN FRAME (NO WARNING)
3. Warning counter stays same

**Object Alone (NO WARNING):**
1. Place paper/phone on desk beside you
2. Keep hands empty
3. Even after 2+ seconds → ✓ CLEAN FRAME (NO WARNING)
4. Warning counter stays same

**Multiple Violations → Rejection:**
1. Have 2 people in frame + hand carrying object
2. Sustain violations across multiple cycles
3. After 5 total violations → INTERVIEW TERMINATED 🛑

## Thresholds

All thresholds can be adjusted in `ObjectDetector.__init__()`:

```python
self.multiple_people_threshold = 90  # 3 seconds
self.hand_carrying_threshold = 60  # 2 seconds
self.min_people_for_warning = 2  # Warn if 2+ people
self.hand_area_threshold = 500  # Min pixels for hand
self.material_brightness_threshold = 100  # Brightness cutoff
```

## Output Statistics

When you quit (press 'q'), the demo prints:
```
Total frames processed: XXXX
Final Warning Count: X/5
Detection Statistics:
  single_person: XXXX frames (XX.X%)
  multiple_people: XXXX frames (XX.X%)
  hand_with_object: XXXX frames (XX.X%)  [Only hand carrying object]
  clean_frames: XXXX frames (XX.X%)
```

## Integration with Other Phases

Phase 3 works independently but can be combined with:
- **Phase 0**: Eye Contact Detection
- **Phase 1**: Face Orientation Detection
- **Phase 2**: Posture Detection

All phases share the same 5-warning system when integrated.

## Troubleshooting

**Issue: False multiple people detection**
- Adjust `min_people_for_warning` threshold
- Improve lighting for better face detection

**Issue: Hand carrying object detection too sensitive**
- Requires BOTH hand AND object to be in frame simultaneously
- Check that both are detected together in the info line
- Adjust `hand_area_threshold` if needed

**Issue: Hand carrying object not detecting**
- Ensure object is bright enough (papers, phones)
- Increase `material_brightness_threshold` if too sensitive
- Check lighting conditions
- Make sure hand is clearly in contact with object

**Issue: Warnings not triggering**
- Check threshold values (default: 2-3 seconds)
- Ensure violations are sustained long enough
- Verify both hand AND object are detected in same frame
- Check that hand_area is above 500 pixels
