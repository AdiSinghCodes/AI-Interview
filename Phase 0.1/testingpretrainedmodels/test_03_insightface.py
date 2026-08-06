"""
ADVANCED INTERVIEW MONITORING SYSTEM
===================================

FEATURES:
✓ Authorized user capture
✓ Face verification
✓ Multiple people detection
✓ Warning counter
✓ Interview termination after 5 warnings
✓ Intruder detection
✓ Screenshot evidence saving
✓ Real-time monitoring

CONTROLS:
------------------------------------------------
SPACE  -> Capture authorized user
R      -> Reset system
S      -> Save screenshot
Q      -> Quit
------------------------------------------------
"""

import cv2
import numpy as np
import time
from collections import deque

# =========================================================
# INSIGHTFACE IMPORT
# =========================================================

try:
    import insightface
    from insightface.app import FaceAnalysis

except ImportError:

    print("❌ insightface not installed.")
    print("Run this command:")
    print("pip install insightface onnxruntime")

    exit(1)

# =========================================================
# SETTINGS
# =========================================================

SIMILARITY_THRESHOLD = 0.50

MAX_WARNINGS = 5

WARNING_COOLDOWN = 3   # seconds

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
    )


def normalize_embedding(embedding):

    return embedding / (
        np.linalg.norm(embedding) + 1e-6
    )


def draw_face(frame, face, color, label):

    bbox = face.bbox.astype(int)

    x1, y1, x2, y2 = bbox

    # Face rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # Label size
    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    # Label background
    cv2.rectangle(
        frame,
        (x1, y1 - th - 10),
        (x1 + tw + 10, y1),
        color,
        -1
    )

    # Label text
    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    # Facial landmarks
    if hasattr(face, 'kps') and face.kps is not None:

        for kp in face.kps:

            cv2.circle(
                frame,
                (int(kp[0]), int(kp[1])),
                3,
                color,
                -1
            )


def save_evidence(frame, reason):

    timestamp = int(time.time())

    filename = f"evidence_{reason}_{timestamp}.jpg"

    cv2.imwrite(filename, frame)

    print(f"📸 Evidence saved: {filename}")


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    print("⏳ Loading InsightFace...")

    # =====================================================
    # LOAD MODEL
    # =====================================================

    try:

        app = FaceAnalysis(
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )

        app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

        print("✅ InsightFace Loaded Successfully")

    except Exception as e:

        print(f"❌ Failed to load model: {e}")

        return

    # =====================================================
    # START CAMERA
    # =====================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("❌ Failed to open webcam")

        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # =====================================================
    # VARIABLES
    # =====================================================

    authorized_embedding = None

    warning_count = 0

    last_warning_time = 0

    interview_terminated = False

    fps_buffer = deque(maxlen=30)

    # =====================================================
    # INSTRUCTIONS
    # =====================================================

    print("\n🎥 SYSTEM STARTED")

    print("SPACE -> Capture authorized user")

    print("Q -> Quit")

    print("R -> Reset")

    print("S -> Screenshot")

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

        # =================================================
        # RGB CONVERSION
        # =================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # =================================================
        # FACE DETECTION
        # =================================================

        faces = app.get(rgb)

        current_time = time.time()

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
                "Multiple persons detected repeatedly",
                (150, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Interview Monitoring System",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):

                break

            continue

        # =================================================
        # DEFAULT STATUS
        # =================================================

        status_text = "WAITING FOR AUTHORIZED USER"

        status_color = (200, 200, 200)

        # =================================================
        # MULTIPLE PEOPLE DETECTION
        # =================================================

        if len(faces) > 1:

            # Warning cooldown
            if current_time - last_warning_time > WARNING_COOLDOWN:

                warning_count += 1

                last_warning_time = current_time

                print(
                    f"⚠ WARNING {warning_count}/{MAX_WARNINGS}: "
                    f"Multiple people detected"
                )

                save_evidence(
                    frame,
                    "multiple_people"
                )

            status_text = (
                f"⚠ WARNING {warning_count}/{MAX_WARNINGS}: "
                f"MULTIPLE PEOPLE DETECTED"
            )

            status_color = (0, 165, 255)

            # Draw all faces
            for face in faces:

                draw_face(
                    frame,
                    face,
                    (0, 0, 255),
                    "MULTIPLE PEOPLE"
                )

            # TERMINATION CONDITION
            if warning_count >= MAX_WARNINGS:

                interview_terminated = True

                save_evidence(
                    frame,
                    "interview_terminated"
                )

                print("\n❌ INTERVIEW TERMINATED")

                print(
                    "Reason: Multiple persons detected repeatedly"
                )

        # =================================================
        # SINGLE FACE DETECTED
        # =================================================

        elif len(faces) == 1:

            face = faces[0]

            # No authorized user yet
            if authorized_embedding is None:

                draw_face(
                    frame,
                    face,
                    (255, 255, 0),
                    "FACE DETECTED"
                )

            else:

                current_embedding = normalize_embedding(
                    face.embedding
                )

                similarity = cosine_similarity(
                    authorized_embedding,
                    current_embedding
                )

                # AUTHORIZED USER
                if similarity >= SIMILARITY_THRESHOLD:

                    status_text = (
                        "✅ VERIFIED AUTHORIZED USER"
                    )

                    status_color = (0, 255, 0)

                    draw_face(
                        frame,
                        face,
                        (0, 255, 0),
                        f"AUTHORIZED ({similarity:.2f})"
                    )

                # UNAUTHORIZED USER
                else:

                    status_text = "⚠ INTRUDER DETECTED"

                    status_color = (0, 0, 255)

                    draw_face(
                        frame,
                        face,
                        (0, 0, 255),
                        f"UNAUTHORIZED ({similarity:.2f})"
                    )

        # =================================================
        # NO FACE DETECTED
        # =================================================

        else:

            if authorized_embedding is not None:

                status_text = (
                    "⚠ AUTHORIZED USER MISSING"
                )

                status_color = (0, 0, 255)

        # =================================================
        # TOP PANEL
        # =================================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 100),
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
            "ADVANCED INTERVIEW MONITORING SYSTEM",
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

        # Warning Count
        cv2.putText(
            frame,
            f"Warnings: {warning_count}/{MAX_WARNINGS}",
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # =================================================
        # FPS DISPLAY
        # =================================================

        t1 = time.time()

        fps = 1.0 / max(t1 - t0, 0.001)

        fps_buffer.append(fps)

        avg_fps = np.mean(fps_buffer)

        cv2.putText(
            frame,
            f"FPS: {avg_fps:.1f} | Faces: {len(faces)}",
            (w - 220, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1
        )

        # =================================================
        # SHOW WINDOW
        # =================================================

        cv2.imshow(
            "Interview Monitoring System",
            frame
        )

        # =================================================
        # KEYBOARD CONTROLS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        # QUIT
        if key == ord('q'):

            break

        # CAPTURE AUTHORIZED USER
        elif key == ord(' '):

            if len(faces) == 0:

                print("❌ No face detected")

            elif len(faces) > 1:

                print("❌ Multiple faces detected")

                print(
                    "Only ONE person should be visible"
                )

            else:

                authorized_embedding = normalize_embedding(
                    faces[0].embedding
                )

                print(
                    "✅ Authorized user captured successfully"
                )

        # RESET SYSTEM
        elif key == ord('r'):

            authorized_embedding = None

            warning_count = 0

            interview_terminated = False

            print("🔄 System reset")

        # MANUAL SCREENSHOT
        elif key == ord('s'):

            filename = (
                f"screenshot_{int(time.time())}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            print(f"📸 Screenshot saved: {filename}")

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