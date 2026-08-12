# Integrated Interview Cheating Detection System

Complete AI-based avatar interview system with progressive cheating detection across 4 synchronized phases.

## 🎯 Overview

This is the **UNIFIED INTERVIEW SYSTEM** combining all 4 detection phases:

| Phase | Detection | Trigger |
|-------|-----------|---------|
| **0** | Eye Contact | Gaze not CENTER for 2+ seconds |
| **1** | Face Orientation | Face misaligned for 4+ seconds OR no-face for 2+ seconds |
| **2** | Posture | Poor posture for 4+ seconds OR no-pose for 3+ seconds |
| **3** | Object Detection | Multiple people (2+) for 3+ seconds OR hand carrying object for 2+ seconds |

## ⚠️ Warning System

**Single Shared Counter: 0-5 warnings**

- **0 Warnings** ✓ GREEN - Interview Active
- **1-2 Warnings** 🟡 YELLOW - Minor Violations
- **3-4 Warnings** 🟠 ORANGE - Multiple Violations  
- **5 Warnings** 🔴 RED - **INTERVIEW TERMINATED**

Each violation type detected in a frame can contribute to warning count. After sustained violation meets threshold, warning increments by 1.

## 🚀 Usage

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Integrated System

```bash
python integrated_demo.py
```

### Controls

- **q** - Quit and view results

## 📊 Real-Time Dashboard

The unified interface displays:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Warnings: 2/5                              │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Phase 0: Eye Contact - CENTER                                     │
│ ✓ Phase 1: Face Orientation - ALIGNED                              │
│ ✗ Phase 2: Posture - SLOUCHED                                       │
│ ✓ Phase 3: Object Detection - People:1                              │
├─────────────────────────────────────────────────────────────────────┤
│ Violations This Frame: 1                                             │
│ Status: ✓ Interview Active | 3 warning(s) remaining                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Color Indicators

- **GREEN** border + status - Clean interview, 0 warnings
- **YELLOW** border - Minor violations, 1-2 warnings
- **ORANGE** border - Multiple violations, 3-4 warnings  
- **RED** border + overlay - Interview terminated, 5/5 warnings

## 🧪 Testing Scenarios

### Test 1: Normal Interview ✓
- Sit centered
- Eyes looking at camera
- Proper posture
- One person
- No objects in hand

**Expected:** 0/5 warnings, GREEN status

### Test 2: Eye Gaze Violation ⚠️
- Look away from camera for 2+ seconds
- After ~2 sec → gaze changes to LEFT/RIGHT/DOWN
- After ~2 sec sustained → Warning 1/5

### Test 3: Face Misalignment ⚠️
- Turn head significantly to side for 4+ seconds
- After ~4 sec → alignment becomes MISALIGNED
- After ~4 sec sustained → Warning increments

### Test 4: Posture Violation ⚠️
- Slouch or lean forward for 4+ seconds
- After ~4 sec → posture becomes SLOUCHED
- After ~4 sec sustained → Warning increments

### Test 5: Object with Hand ⚠️
- Pick up phone/paper and hold it for 2+ seconds
- After ~2 sec → hand+object detected
- After ~2 sec sustained → Warning increments

### Test 6: Multiple People ⚠️
- Have another person enter frame for 3+ seconds
- After ~3 sec → people_count = 2
- After ~3 sec sustained → Warning increments

### Test 7: Rejection (5 Violations) ❌
- Accumulate 5 violations through combination of above
- After 5th warning → RED overlay appears
- Status: "❌ INTERVIEW TERMINATED"

## 📈 Output Statistics

When interview ends, system prints:

```
Session Completed
═══════════════════════════════════════════════════

Total Frames Processed: XXXX
Final Warning Count: X/5

❌ INTERVIEW RESULT: REJECTED
Candidate exhibited suspicious behavior exceeding acceptable threshold

Violations Breakdown:
  eye_contact: XXX (X.X%)
  face_orientation: XXX (X.X%)
  posture: XXX (X.X%)
  object_detection: XXX (X.X%)
  clean_frames: XXX (X.X%)

Clean Frame Percentage: X.X%
═══════════════════════════════════════════════════
```

## 🔧 Customization

### Thresholds by Phase

**Phase 0 (Eye Contact):** [eye_contact_detection/eye_contact_model.py](../eye_contact_detection/eye_contact_model.py)
- `suspicious_threshold = 60` frames (2 sec)

**Phase 1 (Face Orientation):** [face_orientation_detection/face_orientation_model.py](../face_orientation_detection/face_orientation_model.py)
- `misalignment_threshold = 120` frames (4 sec)
- `no_face_threshold = 60` frames (2 sec)

**Phase 2 (Posture):** [posture_detection/posture_model.py](../posture_detection/posture_model.py)
- `overall_threshold = 120` frames (4 sec)
- `no_pose_threshold = 90` frames (3 sec)

**Phase 3 (Object):** [object_detection/object_model.py](../object_detection/object_model.py)
- `multiple_people_threshold = 90` frames (3 sec)
- `hand_carrying_threshold = 60` frames (2 sec)

### Unified Warning Logic

Edit `_update_warnings()` in [integrated_model.py](integrated_model.py) to customize how violations translate to warnings.

## 🛠️ Architecture

### Files

- **integrated_model.py** - IntegratedInterviewSystem class combining all 4 detectors
- **integrated_demo.py** - Real-time visualization and dashboard
- **requirements.txt** - Python dependencies
- **README.md** - This file

### Detector Integration

```python
system = IntegratedInterviewSystem()
result = system.process_frame(frame)

# Access results
result['violations']           # Dict of 4 violation types
result['warning_count']        # Unified warning counter
result['interview_terminated'] # Boolean
result['details']              # Detailed info per phase
```

## 📝 How It Works

1. **Frame Capture** - Read from webcam at 30 FPS
2. **Phase 0** - Eye contact analysis (gaze direction)
3. **Phase 1** - Face orientation (alignment, head pose)
4. **Phase 2** - Posture analysis (body positioning)
5. **Phase 3** - Object detection (multiple people, hand+object)
6. **Warning Update** - Unified counter reflects all violations
7. **Visualization** - Dashboard shows all 4 detections + unified status
8. **Rejection Logic** - Interview terminates at 5/5 warnings

## ⚡ Performance

- **Latency** - ~50-70ms per frame (depending on system)
- **CPU Usage** - Moderate (optimized with Haar Cascades)
- **FPS** - Target 30 FPS with real-time visualization
- **Memory** - ~150-200MB for 4 detectors + video buffer

## 🔍 Troubleshooting

**Issue: Camera not detected**
- Ensure webcam is connected and not in use by other apps
- Check Windows device manager for camera presence
- Try restarting the demo

**Issue: Warnings triggering too frequently**
- Adjust thresholds in individual phase model files
- Increase frame count thresholds for stricter detection
- Check lighting conditions

**Issue: Certain violations not detected**
- Verify individual phases work correctly first
- Check lighting and camera positioning
- Review detection parameters for that phase

**Issue: System too sensitive**
- Increase threshold values in respective detector
- Reduce Canny edge detection sensitivity (Phase 3)
- Adjust color detection ranges (hand/skin detection)

## 📚 References

- **Phase 0** - [Eye Contact Detection](../eye_contact_detection/README.md)
- **Phase 1** - [Face Orientation Detection](../face_orientation_detection/README.md)
- **Phase 2** - [Posture Detection](../posture_detection/README.md)
- **Phase 3** - [Object Detection](../object_detection/README.md)

## 🎓 Application

This system is designed for:
- Online interview proctoring
- Remote exam invigilation
- AI-based HR screening interviews
- Candidate cheating prevention
- Real-time behavioral monitoring

---

**Status**: ✅ Complete and Tested  
**Last Updated**: Phase 3 Integration  
**Phases**: 4/4 Complete
