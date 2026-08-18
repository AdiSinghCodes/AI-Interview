"""
TEST 2: MediaPipe Face Mesh
============================
What it shows:
  • 468 facial landmarks drawn live
  • Eye Aspect Ratio (EAR) — blink detection
  • Mouth open detection
  • Head pose (yaw/pitch/roll) from landmarks
  • Face count (multi-person detection)
  • Iris position approximation

No GPU required. Runs on CPU at 60+ FPS.

Controls:
  Q  - Quit
  S  - Screenshot
  L  - Toggle landmark dots on/off
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
import time
from collections import deque
import math


try:
    import mediapipe as mp
except ImportError:
    print("❌  mediapipe not installed. Run:  pip install mediapipe")
    exit(1)

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Landmark indices ─────────────────────────────────────────────────────────
# Eye landmarks for EAR
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Iris
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Mouth
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291

# ── EAR calculation ──────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # Vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    # Horizontal distance
    h1 = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h1 + 1e-6)

# ── Head pose (simplified) from landmarks ────────────────────────────────────
def estimate_head_pose(landmarks, w, h):
    """
    Approximate yaw/pitch from nose tip vs face center.
    Rough but instant — no extra model needed.
    """
    nose_tip   = landmarks[1]
    chin       = landmarks[152]
    left_eye   = landmarks[33]
    right_eye  = landmarks[263]

    eye_center_x = (left_eye.x + right_eye.x) / 2
    yaw   = (nose_tip.x - eye_center_x) * 200   # degrees approx
    pitch = (nose_tip.y - chin.y) * (-300)       # degrees approx
    return yaw, pitch

def draw_bar(frame, x, y, w, h, value, vmin, vmax, label, color):
    """Horizontal gauge bar."""
    pct = (value - vmin) / (vmax - vmin)
    pct = max(0.0, min(1.0, pct))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * pct), y + h), color, -1)
    cv2.putText(frame, f"{label}: {value:.1f}",
                (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("❌ Failed to open webcam. Please check if another camera index or app is active.")
        return


    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,   # enables iris
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    fps_buffer   = deque(maxlen=30)
    blink_count  = 0
    blink_active = False
    EAR_THRESH   = 0.21
    show_landmarks = True
    
    # ── Warning system ──────────────────────────────────────────────────────
    warning_count = 0
    caught = False
    looking_away_start = None
    looking_away_warned = False
    multi_person_warned = False

    print("\n🎥  Webcam started. Controls: Q=Quit  S=Screenshot  L=Toggle landmarks\n")
    print("ℹ️   Blink slowly to test EAR detection")
    print("ℹ️   Turn your head left/right to test yaw")
    print("ℹ️   Tilt head up/down to test pitch\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Default values
        ear = 0.0
        yaw = 0.0
        pitch = 0.0
        num_faces = 0
        mouth_open = False
        looking_away = False

        # ── Check for multi-person warning ──────────────────────────────────────
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            
            # Multi-person warning (only count once per incident)
            if num_faces > 1 and not multi_person_warned:
                warning_count += 1
                multi_person_warned = True
                if warning_count >= 5:
                    caught = True
                print(f"🚨 WARNING #{warning_count}: Multiple people detected!")
            elif num_faces == 1:
                multi_person_warned = False

            for face_lm in results.multi_face_landmarks:
                lm = face_lm.landmark

                # Draw mesh
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, face_lm,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        frame, face_lm,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    if hasattr(mp_face_mesh, 'FACEMESH_IRISES'):
                        mp_drawing.draw_landmarks(
                            frame, face_lm,
                            mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                        )

                # EAR
                ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
                ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                ear   = (ear_l + ear_r) / 2.0

                # Blink detection
                if ear < EAR_THRESH and not blink_active:
                    blink_active = True
                    blink_count += 1
                elif ear >= EAR_THRESH:
                    blink_active = False

                # Mouth open
                mt = lm[MOUTH_TOP]
                mb = lm[MOUTH_BOTTOM]
                mouth_gap = abs(mb.y - mt.y)
                mouth_open = mouth_gap > 0.04

                # Head pose
                yaw, pitch = estimate_head_pose(lm, w, h)
                looking_away = abs(yaw) > 30 or abs(pitch) > 25
                
                # ── Looking away timer (3 seconds) ──────────────────────────────
                if looking_away:
                    if looking_away_start is None:
                        looking_away_start = time.time()
                    elif (time.time() - looking_away_start) >= 3.0 and not looking_away_warned:
                        warning_count += 1
                        looking_away_warned = True
                        if warning_count >= 5:
                            caught = True
                        print(f"🚨 WARNING #{warning_count}: Looking away for 3+ seconds!")
                else:
                    looking_away_start = None
                    looking_away_warned = False

                break  # only analyze first face for stats

        # ── HUD ──────────────────────────────────────────────────────────────
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "TEST 2: MediaPipe Face Mesh",
                    (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 255), 1)

        face_color = (60, 200, 60) if num_faces == 1 else (50, 50, 255) if num_faces > 1 else (200, 80, 80)
        cv2.putText(frame, f"Faces: {num_faces}  |  Blinks: {blink_count}  |  Landmarks: 468/face",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)

        # Right panel
        px = w - 230
        cv2.rectangle(frame, (px - 10, 60), (w - 5, 310), (20, 20, 30), -1)

        draw_bar(frame, px, 80,  200, 14, ear,   0.0, 0.4,  "EAR (eyes)",  (100, 200, 255))
        draw_bar(frame, px, 120, 200, 14, abs(yaw),   0, 60, "Yaw |°|",   (255, 200, 100))
        draw_bar(frame, px, 160, 200, 14, abs(pitch), 0, 60, "Pitch |°|", (200, 100, 255))

        # Status indicators
        def status_dot(text, cond_true, true_col, false_col, y_pos):
            col = true_col if cond_true else false_col
            cv2.circle(frame, (px + 8, y_pos), 7, col, -1)
            cv2.putText(frame, text, (px + 22, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        status_dot("Blinking",      blink_active,  (50, 50, 200),  (50, 50, 50),  210)
        status_dot("Mouth Open",    mouth_open,    (255, 165, 50), (50, 50, 50),  235)
        status_dot("Looking Away",  looking_away,  (50, 50, 220),  (50, 150, 50), 260)
        status_dot("Multi-person",  num_faces > 1, (50, 50, 220),  (50, 50, 50),  285)

        # FPS
        t1 = time.time()
        fps_buffer.append(1.0 / max(t1 - t0, 0.001))
        cv2.putText(frame, f"FPS: {np.mean(fps_buffer):.1f}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        # ── CAUGHT DETECTION (5+ warnings) ──────────────────────────────────────
        if caught:
            # Large red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "YOU HAVE BEEN CAUGHT!",
                        (w//2 - 250, h//2 - 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, "SUSPICIOUS BEHAVIOR DETECTED",
                        (w//2 - 200, h//2 + 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 100, 255), 2)
            
            # Bottom bar
            cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 220), -1)
            cv2.putText(frame, f"🚨 CAUGHT! Total Warnings: {warning_count}/5",
                        (10, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 3)
        
        # ── Bottom warning bar ──────────────────────────────────────────────────
        elif num_faces > 1:
            # Multi-person warning
            cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 200), -1)
            cv2.putText(frame, f"🚨 MULTIPLE PEOPLE DETECTED ({num_faces}) — Warnings: {warning_count}/5",
                        (10, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 100, 255), 2)
        elif looking_away_warned:
            # Looking away warning (3+ seconds) - TRIGGERED
            cv2.rectangle(frame, (0, h - 48), (w, h), (0, 100, 200), -1)
            cv2.putText(frame, f"⚠️  LOOKING AWAY (3+ sec) — Warnings: {warning_count}/5",
                        (10, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 200, 255), 2)
        elif looking_away and num_faces > 0:
            # Looking away but not yet 3 seconds - TIMER
            cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 180), -1)
            time_looking = 0
            if looking_away_start:
                time_looking = time.time() - looking_away_start
            cv2.putText(frame, f"⚠  LOOKING AWAY ({time_looking:.1f}s/3.0s) — Warnings: {warning_count}/5",
                        (10, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        elif num_faces == 0:
            cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 130), -1)
            cv2.putText(frame, "⚠  NO FACE DETECTED",
                        (10, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 200, 200), 2)
        else:
            # Normal state - all good
            cv2.rectangle(frame, (0, h - 48), (w, h), (0, 100, 0), -1)
            cv2.putText(frame, f"✅ Normal — Warnings: {warning_count}/5",
                        (10, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 100), 2)

        cv2.imshow("Proctoring — Test 2: MediaPipe Face Mesh", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"mediapipe_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸  Saved {fname}")
        elif key == ord('l'):
            show_landmarks = not show_landmarks

    cap.release()
    cv2.destroyAllWindows()
    
    # ── Final Report ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 TEST 2 FINAL REPORT (Modified - With Warning System)")
    print("="*70)
    print(f"Total Blinks Detected: {blink_count}")
    print(f"Total Warnings Given: {warning_count}/5")
    print(f"\nWarnings Monitored:")
    print(f"  • Looking Away (3+ sec): Detected")
    print(f"  • Multi-Person Detection: Detected")
    print(f"\nStatus: {'🚨 CAUGHT CHEATING' if caught else '✅ CLEAN SESSION'}")
    print("="*70)

if __name__ == "__main__":
    main()