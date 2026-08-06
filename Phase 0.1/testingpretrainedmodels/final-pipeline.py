# =========================================================
# FINAL STABLE AI MOCK INTERVIEW MONITORING SYSTEM
# =========================================================

import cv2
import numpy as np
import time
import os

from ultralytics import YOLO
from insightface.app import FaceAnalysis
import mediapipe as mp

from collections import deque
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

MAX_WARNINGS = 10

WARNING_COOLDOWN = 5

PHONE_CONF = 0.55

PERSON_CONF = 0.55

VIOLATION_TIME = 3.0

# ---------------------------------------------------------
# GAZE SETTINGS
# ---------------------------------------------------------

GAZE_YAW_THRESH = 30

GAZE_PITCH_THRESH = 28

# ---------------------------------------------------------
# FACE CENTER SETTINGS
# ---------------------------------------------------------

CENTER_LEFT = 0.35
CENTER_RIGHT = 0.65

CENTER_TOP = 0.30
CENTER_BOTTOM = 0.75

# =========================================================
# EVIDENCE FOLDER
# =========================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

EVIDENCE_DIR = os.path.join(
    "Interview_Evidence",
    timestamp
)

os.makedirs(EVIDENCE_DIR, exist_ok=True)

print(f"\n📂 Evidence Folder Created:")
print(EVIDENCE_DIR)

# =========================================================
# GLOBAL VARIABLES
# =========================================================

warning_count = 0

interview_terminated = False

last_warning_times = {}

authorized_embedding = None

# =========================================================
# MEDIAPIPE LANDMARKS
# =========================================================

LEFT_IRIS = [474, 475, 476, 477]

RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263

RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33

# =========================================================
# SAVE EVIDENCE
# =========================================================

def save_evidence(frame, reason):

    filename = os.path.join(

        EVIDENCE_DIR,
        f"{reason}_{int(time.time())}.jpg"

    )

    cv2.imwrite(filename, frame)

    print(f"📸 Evidence Saved: {filename}")

# =========================================================
# WARNING SYSTEM
# =========================================================

def add_warning(reason, frame):

    global warning_count
    global interview_terminated

    current_time = time.time()

    if reason not in last_warning_times:

        last_warning_times[reason] = 0

    # cooldown
    if (
        current_time - last_warning_times[reason]
        < WARNING_COOLDOWN
    ):
        return

    warning_count += 1

    last_warning_times[reason] = current_time

    print(f"\n⚠ WARNING {warning_count}/{MAX_WARNINGS}")

    print(f"Reason: {reason}")

    save_evidence(frame, reason)

    # TERMINATION
    if warning_count >= MAX_WARNINGS:

        interview_terminated = True

        save_evidence(
            frame,
            "INTERVIEW_TERMINATED"
        )

# =========================================================
# INSIGHTFACE HELPERS
# =========================================================

def normalize_embedding(embedding):

    return embedding / (
        np.linalg.norm(embedding)
        + 1e-6
    )

def cosine_similarity(a, b):

    return np.dot(a, b) / (

        np.linalg.norm(a)
        *
        np.linalg.norm(b)

        + 1e-6
    )

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

    left_iris_pts = [pt(i) for i in LEFT_IRIS]

    right_iris_pts = [pt(i) for i in RIGHT_IRIS]

    left_iris_c = np.mean(left_iris_pts, axis=0)

    right_iris_c = np.mean(right_iris_pts, axis=0)

    ll_inner = pt(LEFT_EYE_INNER)

    ll_outer = pt(LEFT_EYE_OUTER)

    rl_inner = pt(RIGHT_EYE_INNER)

    rl_outer = pt(RIGHT_EYE_OUTER)

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

    yaw = (
        avg_ratio - 0.5
    ) * 220.0

    nose = pt(1)

    forehead = pt(10)

    eye_center_y = (
        left_iris_c[1]
        +
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

    return yaw, pitch

# =========================================================
# LOAD YOLO
# =========================================================

print("\n⏳ Loading YOLOv8...")

yolo_model = YOLO("yolov8n.pt")

print("✅ YOLO Loaded")

# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print("\n⏳ Loading InsightFace...")

face_app = FaceAnalysis(
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("✅ InsightFace Loaded")

# =========================================================
# LOAD MEDIAPIPE
# =========================================================

print("\n⏳ Loading MediaPipe...")

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

)

print("✅ MediaPipe Loaded")

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():

    print("❌ Failed to open webcam")

    exit(1)

# =========================================================
# VARIABLES
# =========================================================

fps_buffer = deque(maxlen=30)

gaze_timer = 0

face_position_timer = 0

t_last = time.time()

# ---------------------------------------------------------
# STABILITY BUFFERS
# ---------------------------------------------------------

gaze_yaw_history = deque(maxlen=10)

gaze_pitch_history = deque(maxlen=10)

face_x_history = deque(maxlen=10)

face_y_history = deque(maxlen=10)

# =========================================================
# START
# =========================================================

print("\n🎥 FINAL AI MOCK INTERVIEW SYSTEM STARTED")

print("\nControls:")
print("SPACE -> Capture Authorized User")
print("Q -> Quit")
print("S -> Screenshot")

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    t0 = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # TERMINATION SCREEN
    # =====================================================

    if interview_terminated:

        frame[:] = (0, 0, 120)

        cv2.putText(

            frame,
            "INTERVIEW TERMINATED",
            (220, h // 2 - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            (255, 255, 255),
            3

        )

        cv2.putText(

            frame,
            "Cheating detected repeatedly",
            (180, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2

        )

        cv2.imshow(
            "AI Mock Interview System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        continue

    # =====================================================
    # YOLO DETECTION
    # =====================================================

    yolo_results = yolo_model(
        frame,
        verbose=False
    )

    person_count = 0

    for result in yolo_results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            name = yolo_model.names[cls]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # PERSON
            if (
                name == "person"
                and
                conf > PERSON_CONF
            ):

                person_count += 1

            # PHONE
            if (
                name == "cell phone"
                and
                conf > PHONE_CONF
            ):

                cv2.rectangle(

                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2

                )

                cv2.putText(

                    frame,
                    "PHONE DETECTED",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2

                )

                add_warning(
                    "PHONE_DETECTED",
                    frame
                )

    # =====================================================
    # MULTIPLE PERSON
    # =====================================================

    if person_count > 1:

        add_warning(
            "MULTIPLE_PEOPLE",
            frame
        )

        cv2.putText(

            frame,
            "MULTIPLE PEOPLE DETECTED",
            (w // 2 - 250, 90),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 0, 255),
            3

        )

    # =====================================================
    # INSIGHTFACE
    # =====================================================

    faces = face_app.get(rgb)

    if len(faces) == 1:

        face = faces[0]

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        current_embedding = normalize_embedding(
            face.embedding
        )

        if authorized_embedding is not None:

            similarity = cosine_similarity(

                authorized_embedding,
                current_embedding

            )

            if similarity >= 0.50:

                color = (0, 255, 0)

                label = "AUTHORIZED"

            else:

                color = (0, 0, 255)

                label = "UNAUTHORIZED"

                add_warning(
                    "UNAUTHORIZED_FACE",
                    frame
                )

            cv2.rectangle(

                frame,
                (x1, y1),
                (x2, y2),
                color,
                2

            )

            cv2.putText(

                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2

            )

    # =====================================================
    # MEDIAPIPE
    # =====================================================

    results = face_mesh.process(rgb)

    dt = t0 - t_last

    t_last = t0

    if results.multi_face_landmarks:

        lm = results.multi_face_landmarks[0].landmark

        # =================================================
        # GAZE DETECTION
        # =================================================

        gaze_yaw, gaze_pitch = iris_gaze(
            lm,
            w,
            h
        )

        gaze_yaw_history.append(gaze_yaw)

        gaze_pitch_history.append(gaze_pitch)

        avg_gaze_yaw = np.mean(
            gaze_yaw_history
        )

        avg_gaze_pitch = np.mean(
            gaze_pitch_history
        )

        looking_away = (

            abs(avg_gaze_yaw)
            > GAZE_YAW_THRESH

            and

            abs(avg_gaze_pitch)
            > 10

        )

        # =================================================
        # FACE CENTER TRACKING
        # =================================================

        nose = lm[1]

        face_x = nose.x
        face_y = nose.y

        face_x_history.append(face_x)
        face_y_history.append(face_y)

        avg_face_x = np.mean(face_x_history)
        avg_face_y = np.mean(face_y_history)

        head_turned = (

            avg_face_x < CENTER_LEFT
            or
            avg_face_x > CENTER_RIGHT
            or
            avg_face_y < CENTER_TOP
            or
            avg_face_y > CENTER_BOTTOM

        )

        # =================================================
        # PRIORITY:
        # HEAD POSITION > GAZE
        # =================================================

        if head_turned:
            looking_away = False

        # =================================================
        # GAZE TIMER
        # =================================================

        if looking_away:

            gaze_timer += dt

            cv2.putText(

                frame,
                "LOOKING AWAY",
                (w // 2 - 140, 140),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 0, 255),
                3

            )

        else:

            gaze_timer = 0

        if gaze_timer >= VIOLATION_TIME:

            add_warning(
                "GAZE_VIOLATION",
                frame
            )

            gaze_timer = 0

        # =================================================
        # FACE POSITION TIMER
        # =================================================

        if head_turned:

            face_position_timer += dt

            cv2.putText(

                frame,
                "FACE OUT OF CENTER",
                (w // 2 - 220, 190),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 0, 255),
                3

            )

        else:

            face_position_timer = 0

        if face_position_timer >= VIOLATION_TIME:

            add_warning(
                "FACE_POSITION_VIOLATION",
                frame
            )

            face_position_timer = 0

        # =================================================
        # DEBUG VALUES
        # =================================================

        cv2.putText(

            frame,
            f"GazeYaw: {avg_gaze_yaw:+.1f}",
            (w - 300, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2

        )

        cv2.putText(

            frame,
            f"FaceX: {avg_face_x:.2f}",
            (w - 300, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2

        )

        cv2.putText(

            frame,
            f"FaceY: {avg_face_y:.2f}",
            (w - 300, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2

        )

    # =====================================================
    # TOP HUD
    # =====================================================

    overlay = frame.copy()

    cv2.rectangle(

        overlay,
        (0, 0),
        (w, 110),
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

    cv2.putText(

        frame,
        "FINAL AI MOCK INTERVIEW MONITORING SYSTEM",
        (15, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255,255,255),
        1

    )

    cv2.putText(

        frame,
        f"Warnings: {warning_count}/{MAX_WARNINGS}",
        (15, 70),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (0,165,255),
        2

    )

    # =====================================================
    # FPS
    # =====================================================

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
        (180,180,180),
        1

    )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "AI Mock Interview System",
        frame
    )

    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    # QUIT
    if key == ord('q'):
        break

    # CAPTURE AUTHORIZED USER
    elif key == ord(' '):

        if len(faces) == 1:

            authorized_embedding = normalize_embedding(
                faces[0].embedding
            )

            print("\n✅ Authorized User Captured")

    # SCREENSHOT
    elif key == ord('s'):

        save_evidence(
            frame,
            "MANUAL_SCREENSHOT"
        )

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("\n✅ System Closed")