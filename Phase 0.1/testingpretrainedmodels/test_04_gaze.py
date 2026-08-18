"""
ADVANCED GAZE MONITORING SYSTEM
================================

UPDATED FEATURES:
✓ Better LEFT/RIGHT detection
✓ Better UP/DOWN detection
✓ Head turn tracking
✓ Warning after 3 seconds
✓ 5 warnings = interview termination
✓ Screenshot evidence saving
✓ FPS display
✓ Real-time gaze tracking

CONTROLS:
------------------------------------------------
Q -> Quit
S -> Screenshot
------------------------------------------------
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'


import cv2
import numpy as np
import time
import math
from collections import deque

# =========================================================
# MEDIAPIPE IMPORT
# =========================================================

try:
    import mediapipe as mp


except ImportError:

    print("❌ mediapipe not installed")
    print("Run:")
    print("pip install mediapipe")

    exit(1)

# =========================================================
# LANDMARKS
# =========================================================

LEFT_IRIS = [474, 475, 476, 477]

RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263

RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33

# =========================================================
# SETTINGS
# =========================================================

# IMPROVED SIDE DETECTION
GAZE_YAW_THRESH = 15.0

GAZE_PITCH_THRESH = 20.0

GAZE_TIME_LIMIT = 7.5


MAX_WARNINGS = 5

# =========================================================
# GAZE ESTIMATION
# =========================================================

def iris_gaze(landmarks, w, h):

    def pt(idx):

        l = landmarks[idx]

        return np.array([
            l.x * w,
            l.y * h
        ])

    # Iris centers
    left_iris_pts = [pt(i) for i in LEFT_IRIS]

    right_iris_pts = [pt(i) for i in RIGHT_IRIS]

    left_iris_c = np.mean(left_iris_pts, axis=0)

    right_iris_c = np.mean(right_iris_pts, axis=0)

    # Eye corners
    ll_inner = pt(LEFT_EYE_INNER)

    ll_outer = pt(LEFT_EYE_OUTER)

    rl_inner = pt(RIGHT_EYE_INNER)

    rl_outer = pt(RIGHT_EYE_OUTER)

    # Horizontal ratio
    def horiz_ratio(iris, inner, outer):

        span = np.linalg.norm(
            outer - inner
        ) + 1e-6

        return np.dot(
            iris - inner,
            outer - inner
        ) / (span * span)

    left_ratio = horiz_ratio(
        left_iris_c,
        ll_inner,
        ll_outer
    )

    right_ratio = horiz_ratio(
        right_iris_c,
        rl_inner,
        rl_outer
    )

    avg_ratio = (
        left_ratio + right_ratio
    ) / 2.0

    # =========================================
    # IMPROVED YAW SCALING
    # =========================================

    yaw = (
        avg_ratio - 0.5
    ) * 220.0

    # =========================================
    # PITCH
    # =========================================

    nose = pt(1)

    forehead = pt(10)

    eye_center_y = (
        left_iris_c[1] +
        right_iris_c[1]
    ) / 2

    face_h = abs(
        nose[1] - forehead[1]
    ) + 1e-6

    vert_ratio = (
        eye_center_y - forehead[1]
    ) / face_h

    pitch = (
        vert_ratio - 0.5
    ) * -80.0

    return float(yaw), float(pitch)

# =========================================================
# DRAW GAZE ARROW
# =========================================================

def draw_gaze_arrow(
    frame,
    center,
    yaw_deg,
    pitch_deg,
    length=80,
    color=(0, 255, 120)
):

    yaw_r = math.radians(yaw_deg)

    pitch_r = math.radians(pitch_deg)

    dx = int(
        math.sin(yaw_r) * length
    )

    dy = int(
        -math.sin(pitch_r) * length
    )

    end = (
        center[0] + dx,
        center[1] + dy
    )

    cv2.arrowedLine(
        frame,
        center,
        end,
        color,
        3,
        tipLength=0.3
    )

# =========================================================
# SAVE EVIDENCE
# =========================================================

def save_evidence(frame, reason):

    filename = (
        f"evidence_{reason}_{int(time.time())}.jpg"
    )

    cv2.imwrite(filename, frame)

    print(f"📸 Evidence saved: {filename}")

# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # MEDIAPIPE FACE MESH
    # =====================================================

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # =====================================================
    # CAMERA
    # =====================================================

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    if not cap.isOpened():

        print("❌ Failed to open webcam. Please check if another app is using the camera.")

        return


    # =====================================================
    # VARIABLES
    # =====================================================

    fps_buffer = deque(maxlen=30)

    off_screen_seconds = 0.0

    warning_count = 0

    violation_active = False

    interview_terminated = False

    t_last = time.time()

    # =====================================================
    # START MESSAGE
    # =====================================================

    print("\n🎥 Advanced Gaze Monitoring Started")

    print(
        "⚠ Looking away for more than 3 seconds "
        "will increase warning count"
    )

    print(
        f"❌ Interview terminates after "
        f"{MAX_WARNINGS} warnings"
    )

    # =====================================================
    # MAIN LOOP
    # =====================================================

    while True:

        t0 = time.time()

        ret, frame = cap.read()

        if not ret:

            print("❌ Failed to read frame")

            break

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        yaw = 0.0

        pitch = 0.0

        face_found = False

        looking_away = False

        # =================================================
        # TERMINATION SCREEN
        # =================================================

        if interview_terminated:

            frame[:] = (0, 0, 120)

            cv2.putText(
                frame,
                "INTERVIEW TERMINATED",
                (220, h // 2 - 20),
                cv2.FONT_HERSHEY_DUPLEX,
                1.3,
                (255, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "You repeatedly looked away from the screen",
                (100, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Advanced Gaze Monitoring",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):

                break

            continue

        # =================================================
        # FACE DETECTION
        # =================================================

        if results.multi_face_landmarks:

            face_found = True

            lm = results.multi_face_landmarks[0].landmark

            # =============================================
            # GAZE ESTIMATION
            # =============================================

            yaw, pitch = iris_gaze(
                lm,
                w,
                h
            )

            # Nose point
            nose = lm[1]

            nose_center = (
                int(nose.x * w),
                int(nose.y * h)
            )

            # Draw arrow
            draw_gaze_arrow(
                frame,
                nose_center,
                yaw,
                pitch
            )

            # =============================================
            # IMPROVED GAZE DETECTION
            # =============================================

            yaw_violation = (
                abs(yaw) > GAZE_YAW_THRESH
            )

            pitch_violation = (
                abs(pitch) > GAZE_PITCH_THRESH
            )

            # =============================================
            # HEAD TURN DETECTION
            # =============================================

            nose_x = nose.x

            head_turn_violation = (
                nose_x < 0.35
                or
                nose_x > 0.65
            )

            # =============================================
            # FINAL LOOKING AWAY
            # =============================================

            looking_away = (
                yaw_violation
                or
                pitch_violation
                or
                head_turn_violation
            )

            # =============================================
            # VISUAL STATUS
            # =============================================

            if looking_away:

                cv2.putText(
                    frame,
                    "LOOKING AWAY",
                    (w // 2 - 120, 150),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

        # =================================================
        # TIME CALCULATION
        # =================================================

        dt = t0 - t_last

        t_last = t0

        # =================================================
        # GAZE TIMER
        # =================================================

        if looking_away and face_found:

            off_screen_seconds += dt

        else:

            off_screen_seconds = 0

            violation_active = False

        # =================================================
        # WARNING INCREASE
        # =================================================

        if (
            off_screen_seconds >= GAZE_TIME_LIMIT
            and
            not violation_active
        ):

            warning_count += 1

            violation_active = True

            print(
                f"⚠ WARNING {warning_count}/{MAX_WARNINGS}"
            )

            print(
                "Reason: Looking away from screen"
            )

            save_evidence(
                frame,
                "gaze_violation"
            )

        # =================================================
        # TERMINATION
        # =================================================

        if warning_count >= MAX_WARNINGS:

            interview_terminated = True

            save_evidence(
                frame,
                "interview_terminated"
            )

            print("\n❌ INTERVIEW TERMINATED")

        # =================================================
        # STATUS
        # =================================================

        status_text = "LOOKING AT SCREEN"

        status_color = (0, 255, 0)

        if looking_away:

            remaining = max(
                0,
                GAZE_TIME_LIMIT - off_screen_seconds
            )

            status_text = (
                f"⚠ LOOKING AWAY "
                f"({remaining:.1f}s before warning)"
            )

            status_color = (0, 165, 255)

        # =================================================
        # HUD
        # =================================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 120),
            (20, 20, 20),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.8,
            frame,
            0.2,
            0,
            frame
        )

        # Title
        cv2.putText(
            frame,
            "ADVANCED GAZE MONITORING SYSTEM",
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1
        )

        # Status
        cv2.putText(
            frame,
            status_text,
            (15, 65),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            status_color,
            2
        )

        # Warnings
        cv2.putText(
            frame,
            f"Warnings: {warning_count}/{MAX_WARNINGS}",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Yaw
        cv2.putText(
            frame,
            f"Yaw: {yaw:+.1f}",
            (w - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Pitch
        cv2.putText(
            frame,
            f"Pitch: {pitch:+.1f}",
            (w - 250, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Timer
        cv2.putText(
            frame,
            f"Off-screen Time: {off_screen_seconds:.1f}s",
            (w - 320, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # =================================================
        # FPS
        # =================================================

        t1 = time.time()

        fps = 1.0 / max(
            t1 - t0,
            0.001
        )

        fps_buffer.append(fps)

        avg_fps = np.mean(fps_buffer)

        cv2.putText(
            frame,
            f"FPS: {avg_fps:.1f}",
            (w - 150, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1
        )

        # =================================================
        # SHOW WINDOW
        # =================================================

        cv2.imshow(
            "Advanced Gaze Monitoring",
            frame
        )

        # =================================================
        # KEYBOARD
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):

            break

        elif key == ord('s'):

            filename = (
                f"screenshot_{int(time.time())}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            print(
                f"📸 Screenshot saved: {filename}"
            )

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()

    print("\n✅ Program Closed")

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()