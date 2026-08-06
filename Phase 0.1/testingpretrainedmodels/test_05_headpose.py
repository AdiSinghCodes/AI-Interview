"""
ADVANCED HEAD POSE MONITORING SYSTEM
====================================

FEATURES:
✓ Real-time head pose estimation
✓ Left/right/up/down head tracking
✓ Warning after 3 seconds
✓ Warning counter system
✓ Interview termination after 5 warnings
✓ Screenshot evidence saving
✓ 3D pose axes visualization
✓ FPS monitoring

CONTROLS:
------------------------------------------------
Q -> Quit
S -> Screenshot
A -> Toggle 3D axes
------------------------------------------------
"""

import cv2
import numpy as np
import math
import time
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
# 3D FACE MODEL
# =========================================================

MODEL_POINTS = np.array([

    (0.0,    0.0,    0.0),      # Nose tip
    (0.0,   -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0,  170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0,  -150.0, -125.0)    # Right mouth corner

], dtype=np.float32)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# =========================================================
# SETTINGS
# =========================================================

YAW_THRESH = 20.0

PITCH_THRESH = 20.0

POSE_TIME_LIMIT = 3.0

MAX_WARNINGS = 5

# =========================================================
# HEAD POSE ESTIMATION
# =========================================================

def get_head_pose(landmarks, w, h):

    image_pts = np.array([

        [
            landmarks[idx].x * w,
            landmarks[idx].y * h
        ]

        for idx in LANDMARK_IDS

    ], dtype=np.float32)

    focal = w

    cam_matrix = np.array([

        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]

    ], dtype=np.float32)

    dist = np.zeros((4, 1), dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(

        MODEL_POINTS,
        image_pts,
        cam_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE

    )

    if not success:

        return 0.0, 0.0, 0.0, None, None, None

    rmat, _ = cv2.Rodrigues(rvec)

    proj = np.hstack((rmat, tvec))

    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)

    pitch, yaw, roll = [

        float(value)

        for value in euler.ravel()[:3]

    ]

    # Normalize pitch
    pitch = (pitch + 360) % 360

    if pitch > 180:

        pitch -= 360

    return yaw, pitch, roll, rvec, tvec, cam_matrix

# =========================================================
# DRAW 3D AXES
# =========================================================

def draw_axes(frame, rvec, tvec, cam_matrix, length=100):

    dist = np.zeros((4, 1), dtype=np.float32)

    axis_pts = np.float32([

        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
        [0, 0, 0]

    ])

    imgpts, _ = cv2.projectPoints(

        axis_pts,
        rvec,
        tvec,
        cam_matrix,
        dist

    )

    imgpts = imgpts.astype(int)

    origin = tuple(imgpts[3].ravel())

    # X axis - RED
    cv2.arrowedLine(
        frame,
        origin,
        tuple(imgpts[0].ravel()),
        (0, 0, 255),
        3,
        tipLength=0.2
    )

    # Y axis - GREEN
    cv2.arrowedLine(
        frame,
        origin,
        tuple(imgpts[1].ravel()),
        (0, 255, 0),
        3,
        tipLength=0.2
    )

    # Z axis - BLUE
    cv2.arrowedLine(
        frame,
        origin,
        tuple(imgpts[2].ravel()),
        (255, 0, 0),
        3,
        tipLength=0.2
    )

# =========================================================
# SAVE EVIDENCE
# =========================================================

def save_evidence(frame, reason):

    filename = f"evidence_{reason}_{int(time.time())}.jpg"

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

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():

        print("❌ Failed to open webcam")

        return

    # =====================================================
    # VARIABLES
    # =====================================================

    fps_buffer = deque(maxlen=30)

    pose_violation_seconds = 0.0

    warning_count = 0

    violation_active = False

    interview_terminated = False

    show_axes = True

    t_last = time.time()

    # =====================================================
    # START MESSAGE
    # =====================================================

    print("\n🎥 Advanced Head Pose Monitoring Started")

    print(
        "⚠ Turning head away for more than "
        "3 seconds increases warning count"
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

        roll = 0.0

        face_found = False

        head_turned = False

        rvec = None

        tvec = None

        cam_matrix = None

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
                "You repeatedly turned your head away",
                (130, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2

            )

            cv2.imshow(
                "Advanced Head Pose Monitoring",
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
            # HEAD POSE
            # =============================================

            yaw, pitch, roll, rvec, tvec, cam_matrix = get_head_pose(

                lm,
                w,
                h

            )

            # =============================================
            # DRAW AXES
            # =============================================

            if show_axes and rvec is not None:

                draw_axes(
                    frame,
                    rvec,
                    tvec,
                    cam_matrix
                )

            # =============================================
            # HEAD TURN DETECTION
            # =============================================

            yaw_violation = abs(yaw) > YAW_THRESH

            pitch_violation = abs(pitch) > PITCH_THRESH

            head_turned = (

                yaw_violation
                or
                pitch_violation

            )

            # =============================================
            # VISUAL WARNING
            # =============================================

            if head_turned:

                cv2.putText(

                    frame,
                    "HEAD TURNED AWAY",
                    (w // 2 - 180, 160),
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
        # TIMER LOGIC
        # =================================================

        if head_turned and face_found:

            pose_violation_seconds += dt

        else:

            pose_violation_seconds = 0

            violation_active = False

        # =================================================
        # WARNING INCREASE
        # =================================================

        if (

            pose_violation_seconds >= POSE_TIME_LIMIT
            and
            not violation_active

        ):

            warning_count += 1

            violation_active = True

            print(
                f"⚠ WARNING {warning_count}/{MAX_WARNINGS}"
            )

            print(
                "Reason: Head turned away from screen"
            )

            save_evidence(
                frame,
                "headpose_violation"
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

        status_text = "HEAD FACING SCREEN"

        status_color = (0, 255, 0)

        if head_turned:

            remaining = max(
                0,
                POSE_TIME_LIMIT - pose_violation_seconds
            )

            status_text = (
                f"⚠ HEAD TURNED "
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
            (w, 130),
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
            "ADVANCED HEAD POSE MONITORING SYSTEM",
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

        # Axes status
        cv2.putText(

            frame,
            f"Axes: {'ON' if show_axes else 'OFF'}",
            (15, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1

        )

        # Yaw
        cv2.putText(

            frame,
            f"Yaw: {yaw:+.1f}",
            (w - 260, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2

        )

        # Pitch
        cv2.putText(

            frame,
            f"Pitch: {pitch:+.1f}",
            (w - 260, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2

        )

        # Roll
        cv2.putText(

            frame,
            f"Roll: {roll:+.1f}",
            (w - 260, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2

        )

        # Timer
        cv2.putText(

            frame,
            f"Violation Time: {pose_violation_seconds:.1f}s",
            (w - 360, 145),
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
            "Advanced Head Pose Monitoring",
            frame
        )

        # =================================================
        # KEYBOARD
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):

            break

        # Screenshot
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

        # Toggle axes
        elif key == ord('a'):

            show_axes = not show_axes

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