"""
TEST 1: YOLOv8 — Object Detection (Modified)
=============================================
What it detects: ANY object a person is carrying
Auto-downloads weights on first run (~6MB for yolov8n)

Special Features:
  - Detects all objects carried by students
  - Shows warning signals when objects detected
  - Counts warnings (0-5)
  - Shows "CAUGHT" message after 5 warnings

Controls:
  Q       - Quit
  S       - Save screenshot
  M       - Toggle model size (nano ↔ medium)
"""

import cv2
import numpy as np
import time
from collections import deque

# ── Install check ────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("❌  ultralytics not installed. Run:  pip install ultralytics")
    exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
PROCTORING_CLASSES = {
    0:  ("person",    (255, 100, 100)),   # blue-ish
    62: ("tv",        (100, 255, 100)),   # green
    63: ("laptop",    (100, 255, 255)),   # cyan
    64: ("mouse",     (200, 200, 100)),   # yellow
    66: ("keyboard",  (200, 100, 200)),   # purple
    67: ("cell phone",(50,  50,  255)),   # red  ← most important
    73: ("book",      (255, 165,  50)),   # orange
}
CONFIDENCE_THRESHOLD = 0.45
MODELS = {"nano": "yolov8n.pt", "medium": "yolov8m.pt"}

# YOLOv8 COCO classes (all 80 classes)
YOLO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "cat", 15: "dog",
    16: "horse", 17: "sheep", 18: "cow", 19: "elephant", 20: "bear", 21: "zebra",
    22: "giraffe", 23: "backpack", 24: "umbrella", 25: "handbag", 26: "tie",
    27: "suitcase", 28: "frisbee", 29: "skis", 30: "snowboard", 31: "sports ball",
    32: "kite", 33: "baseball bat", 34: "baseball glove", 35: "skateboard",
    36: "surfboard", 37: "tennis racket", 38: "bottle", 39: "wine glass",
    40: "cup", 41: "fork", 42: "knife", 43: "spoon", 44: "bowl", 45: "banana",
    46: "apple", 47: "sandwich", 48: "orange", 49: "broccoli", 50: "carrot",
    51: "hot dog", 52: "pizza", 53: "donut", 54: "cake", 55: "chair", 56: "couch",
    57: "potted plant", 58: "bed", 59: "dining table", 60: "toilet", 61: "tv",
    62: "laptop", 63: "mouse", 64: "remote", 65: "keyboard", 66: "microwave",
    67: "oven", 68: "toaster", 69: "sink", 70: "refrigerator", 71: "book",
    72: "clock", 73: "vase", 74: "scissors", 75: "teddy bear", 76: "hair drier",
    77: "toothbrush", 78: "hair brush", 79: "painting"
}

# ── Severity scoring ──────────────────────────────────────────────────────────
SEVERITY = {
    "cell phone": 10,
    "laptop":     7,
    "tv":         5,
    "book":       4,
    "keyboard":   2,
    "mouse":      1,
    "person":     0,   # counted separately
}

def is_object_in_hand(obj_box, person_boxes):
    """
    Check if an object is being carried/held by a person.
    Returns True if the object overlaps significantly with any person's bounding box.
    
    obj_box: (x1, y1, x2, y2) of object
    person_boxes: list of (x1, y1, x2, y2) for each detected person
    """
    if not person_boxes:
        return False
    
    obj_x1, obj_y1, obj_x2, obj_y2 = obj_box
    obj_area = (obj_x2 - obj_x1) * (obj_y2 - obj_y1)
    
    for person_x1, person_y1, person_x2, person_y2 in person_boxes:
        # Calculate intersection area
        inter_x1 = max(obj_x1, person_x1)
        inter_y1 = max(obj_y1, person_y1)
        inter_x2 = min(obj_x2, person_x2)
        inter_y2 = min(obj_y2, person_y2)
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            # If object has >30% overlap with person, consider it "in hand"
            if inter_area > obj_area * 0.3:
                return True
    
    return False

def load_model(size="nano"):
    print(f"⏳  Loading YOLOv8 {size} … (auto-downloads ~6MB on first run)")
    m = YOLO(MODELS[size])
    print(f"✅  YOLOv8 {size} loaded")
    return m

def draw_overlay(frame, detections, fps, model_size, integrity_score, warning_count, is_warning_active, caught=False):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Warning Signal (flashing border) ──────────────────────────────────────
    if is_warning_active:
        color = (0, 0, 255)  # Red
        thickness = 8
        for i in range(3):
            offset = i * 3
            cv2.rectangle(frame, (offset, offset), (w - offset, h - offset), color, thickness)

    # ── Semi-transparent top bar ──────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # ── Title ─────────────────────────────────────────────────────────────────
    cv2.putText(frame, "TEST 1: YOLOv8 Object Detection (Modified)",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 255), 1)
    
    # ── Warning Counter Display ───────────────────────────────────────────────
    warning_color = (0, 0, 255) if warning_count > 0 else (100, 100, 100)
    cv2.putText(frame, f"Model: {model_size.upper()}  |  FPS: {fps:.1f}  |  Integrity: {integrity_score}/100  |  ⚠ WARNINGS: {warning_count}/5",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warning_color, 1)
    
    cv2.putText(frame, f"Objects in HAND: {len([d for d in detections if d[0] != 'person'])}  (Background objects ignored)",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 180), 1)

    person_count = 0

    for (label, conf, x1, y1, x2, y2, color) in detections:
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        if label == "person":
            person_count += 1

    # ── Caught Message (prominently displayed) ────────────────────────────────
    if caught:
        # Large red box covering center of screen
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//4, h//2 - 80), (3*w//4, h//2 + 80), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Big warning text
        text = "YOU WERE CAUGHT WITH SOMETHING!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, text, (w//2 - tw//2, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        text2 = "IN YOUR HAND"
        (tw2, th2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, text2, (w//2 - tw2//2, h//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ── Warning Messages (bottom) ────────────────────────────────────────────
    if warning_count > 0 and not caught:
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 180), -1)
        warning_text = f"⚠️  WARNING #{warning_count}/5: Objects detected in your hand! Stop using them immediately!"
        cv2.putText(frame, warning_text,
                    (10, h - 12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

    # Person count warning
    if person_count > 1:
        cv2.rectangle(frame, (w - 350, h - 40), (w, h), (0, 100, 200), -1)
        cv2.putText(frame, f"⚠  MULTIPLE PEOPLE: {person_count}",
                    (w - 340, h - 12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    return frame

def main():
    model_size = "nano"
    model = load_model(model_size)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌  Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_buffer = deque(maxlen=30)
    integrity_score = 100
    frame_count = 0
    warning_count = 0
    warning_frames = 0  # Frames with objects detected
    caught = False

    print("\n🎥  Webcam started. Controls: Q=Quit  S=Screenshot  M=Toggle model\n")
    print("📊 SMART DETECTION MODE: Hand-Carry Detection")
    print("✅ Only objects IN YOUR HAND are detected and warned about")
    print("✅ Background objects are automatically IGNORED")
    print("⚠️  After 5 warnings, you will be marked as CAUGHT!")
    print("="*60 + "\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Process every 3rd frame to save CPU (still smooth visually)
        detections = []
        has_carried_object = False
        
        if frame_count % 3 == 0:
            # Detect ALL objects (no class filtering)
            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )
            
            # First pass: collect all persons and their bounding boxes
            person_boxes = []
            all_detections = []
            
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = YOLO_CLASSES.get(cls_id, f"object_{cls_id}")
                
                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))
                
                all_detections.append({
                    'cls_id': cls_id,
                    'conf': conf,
                    'bbox': (x1, y1, x2, y2),
                    'label': label
                })
            
            # Second pass: filter objects - only keep those in hand
            for det in all_detections:
                label = det['label']
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                
                if label == "person":
                    color = (255, 100, 100)  # blue for person
                    detections.append((label, conf, x1, y1, x2, y2, color))
                else:
                    # Check if this object is being held by a person
                    if is_object_in_hand((x1, y1, x2, y2), person_boxes):
                        color = (50, 50, 255)  # red for carried object
                        has_carried_object = True
                        detections.append((label, conf, x1, y1, x2, y2, color))
                    # else: object is in background, don't display or warn

            # Update warning count only for carried objects
            if has_carried_object:
                warning_frames += 1
                # Increment warning counter every 15 frames (0.5 seconds at ~30 FPS) with carried objects
                if warning_frames >= 15:
                    warning_count = min(5, warning_count + 1)
                    warning_frames = 0
                    print(f"⚠️  WARNING #{warning_count}/5: Object being CARRIED detected!")
                    if warning_count >= 5:
                        caught = True
                        print("\n🚨 YOU HAVE BEEN CAUGHT WITH SOMETHING IN YOUR HAND! 🚨\n")
            else:
                warning_frames = 0  # Reset if no carried objects

            # Update integrity score based on carried objects only
            carried_objects = [d for d in detections if d[0] != "person"]
            if carried_objects:
                integrity_score = max(0, integrity_score - 3)
            else:
                integrity_score = min(100, integrity_score + 0.2)

        t1 = time.time()
        fps_buffer.append(1.0 / max(t1 - t0, 0.001))
        fps = np.mean(fps_buffer)

        # Flash warning signal when objects detected
        is_warning_active = (has_carried_object and (frame_count % 10 < 5))  # Blinking effect
        
        frame = draw_overlay(frame, detections, fps, model_size, int(integrity_score), warning_count, is_warning_active, caught)
        cv2.imshow("Proctoring — Test 1: YOLOv8 (Modified)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"yolov8_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸  Saved {fname}")
        elif key == ord('m'):
            model_size = "medium" if model_size == "nano" else "nano"
            model = load_model(model_size)

    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*60)
    print("📊 TEST 1 FINAL REPORT (Modified - Hand-Carry Detection)")
    print("="*60)
    print(f"Final Integrity Score: {int(integrity_score)}/100")
    print(f"Total Warnings Given: {warning_count}/5")
    print(f"Status: {'🚨 CAUGHT CHEATING' if caught else '✅ CLEAN - No suspicious items in hand'}")
    print("\nNote: Only objects being HELD/CARRIED are detected.")
    print("Background objects are automatically ignored.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()