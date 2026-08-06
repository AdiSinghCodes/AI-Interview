"""
TEST 9: Full Visual Pipeline — All Models Together
====================================================
Runs YOLOv8 + MediaPipe + InsightFace + Gaze + Head Pose simultaneously
in separate threads, aggregates into a single integrity score.

This is the closest to a real proctoring system.

Controls:
  Q      - Quit
  SPACE  - Capture identity baseline (for InsightFace)
  R      - Reset all
  S      - Save screenshot + print report
"""

import cv2
import numpy as np
import time
import threading
import math
from collections import deque

# ── Import all models ─────────────────────────────────────────────────────────
print("⏳  Loading models…")

try:
    from ultralytics import YOLO
    yolo = YOLO('yolov8n.pt')
    YOLO_OK = True
    print("  ✅ YOLOv8 ready")
except Exception as e:
    YOLO_OK = False
    print(f"  ⚠  YOLOv8 unavailable: {e}")

try:
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=2,
        refine_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    MP_OK = True
    print("  ✅ MediaPipe ready")
except Exception as e:
    MP_OK = False
    print(f"  ⚠  MediaPipe unavailable: {e}")

try:
    import insightface
    from insightface.app import FaceAnalysis
    iface = FaceAnalysis(providers=['CPUExecutionProvider'],
                         allowed_modules=['detection', 'recognition'])
    iface.prepare(ctx_id=-1, det_size=(320, 320))
    IFACE_OK = True
    print("  ✅ InsightFace ready")
except Exception as e:
    IFACE_OK = False
    print(f"  ⚠  InsightFace unavailable: {e}")

print("✅  All available models loaded\n")

# ── Constants ─────────────────────────────────────────────────────────────────
PROCTORING_CLASSES = {0:"person", 62:"tv", 63:"laptop", 64:"mouse", 66:"keyboard", 67:"cell phone", 73:"book"}
SEVERITY_MAP = {"cell phone":10, "laptop":7, "tv":5, "book":4, "keyboard":2, "mouse":1}
EYE_THRESH   = 0.21
YAW_THRESH   = 30.0
PITCH_THRESH = 25.0
ID_THRESH    = 0.50

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock              = threading.Lock()
        self.objects           = []
        self.num_faces         = 0
        self.ear               = 0.0
        self.gaze_yaw          = 0.0
        self.gaze_pitch        = 0.0
        self.head_yaw          = 0.0
        self.head_pitch        = 0.0
        self.mouth_open        = False
        self.baseline_emb      = None
        self.identity_sim      = None
        self.identity_verdict  = "No baseline"
        self.blink_count       = 0
        self.integrity         = 100.0
        self.violations        = deque(maxlen=50)
        self.gaze_off_dur      = 0.0
        self.head_off_dur      = 0.0
        self.face_absent_dur   = 0.0
        self.last_phone_t      = 0.0
        self.fps               = 0.0
        self.baseline_captures = []
        self.capturing_baseline = False

state = State()

# ── Helper functions ──────────────────────────────────────────────────────────
def ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    h1 = math.dist(pts[0], pts[3]) + 1e-6
    return (v1 + v2) / (2 * h1)

def iris_gaze(lm, w, h):
    def c(idxs): return np.mean([(lm[i].x*w, lm[i].y*h) for i in idxs], axis=0)
    li = c(LEFT_IRIS);  ri = c(RIGHT_IRIS)
    ll = np.array([lm[362].x*w, lm[362].y*h]); lr = np.array([lm[263].x*w, lm[263].y*h])
    rl = np.array([lm[33].x*w,  lm[33].y*h]);  rr = np.array([lm[133].x*w, lm[133].y*h])
    def hr(iris, a, b):
        span = np.linalg.norm(b-a)+1e-6
        return np.dot(iris-a, b-a)/(span**2)
    yaw = ((hr(li,ll,lr)+hr(ri,rl,rr))/2 - 0.5) * 120
    nose = np.array([lm[1].x*w, lm[1].y*h])
    chin = np.array([lm[152].x*w, lm[152].y*h])
    pitch = (nose[1]-chin[1])/(abs(nose[1]-chin[1])+1e-6) * -30
    return float(yaw), float(pitch)

def head_pose_simple(lm, w, h):
    eye_cx = (lm[33].x + lm[263].x) / 2
    yaw    = (lm[1].x - eye_cx) * 200
    pitch  = (lm[1].y - lm[152].y) * -300
    return float(yaw), float(pitch)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

# ── Processing thread ─────────────────────────────────────────────────────────
def processing_loop(cap):
    t_last = time.time()
    blink_active = False
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dt   = t0 - t_last
        t_last = t0

        # ── YOLOv8 (every 3 frames) ───────────────────────────────────────────
        objects = []
        if YOLO_OK and frame_idx % 3 == 0:
            res = yolo(frame, classes=list(PROCTORING_CLASSES.keys()), conf=0.45, verbose=False)
            for box in res[0].boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                objects.append((PROCTORING_CLASSES[cls_id], conf, x1, y1, x2, y2))

        # ── MediaPipe ─────────────────────────────────────────────────────────
        num_faces = 0; ear_val = 0.0; gaze_yaw = 0.0; gaze_pitch = 0.0
        head_yaw  = 0.0; head_pitch = 0.0; mouth_open = False
        if MP_OK:
            mp_res = face_mesh.process(rgb)
            if mp_res.multi_face_landmarks:
                num_faces = len(mp_res.multi_face_landmarks)
                lm = mp_res.multi_face_landmarks[0].landmark
                ear_l = ear(lm, LEFT_EYE, w, h)
                ear_r = ear(lm, RIGHT_EYE, w, h)
                ear_val = (ear_l + ear_r) / 2
                gaze_yaw, gaze_pitch = iris_gaze(lm, w, h)
                head_yaw, head_pitch = head_pose_simple(lm, w, h)
                mouth_open = abs(lm[13].y - lm[14].y) > 0.04

        # ── InsightFace identity (every 30 frames) ────────────────────────────
        id_sim = None; id_verdict = "No baseline"
        if IFACE_OK:
            if state.capturing_baseline:
                faces = iface.get(rgb)
                if faces:
                    with state.lock:
                        state.baseline_captures.append(faces[0].embedding.copy())
                        if len(state.baseline_captures) >= 5:
                            embs = np.array(state.baseline_captures)
                            base = np.mean(embs, axis=0)
                            state.baseline_emb = base / (np.linalg.norm(base) + 1e-6)
                            state.baseline_captures = []
                            state.capturing_baseline = False

            elif state.baseline_emb is not None and frame_idx % 30 == 0:
                faces = iface.get(rgb)
                if faces:
                    emb = faces[0].embedding
                    emb_n = emb / (np.linalg.norm(emb) + 1e-6)
                    id_sim = cosine_sim(state.baseline_emb, emb_n)
                    id_verdict = "✓ VERIFIED" if id_sim >= ID_THRESH else "✗ DIFFERENT PERSON"

        # ── Aggregate & score ─────────────────────────────────────────────────
        with state.lock:
            state.objects    = objects
            state.num_faces  = num_faces
            state.ear        = ear_val
            state.gaze_yaw   = gaze_yaw
            state.gaze_pitch = gaze_pitch
            state.head_yaw   = head_yaw
            state.head_pitch = head_pitch
            state.mouth_open = mouth_open
            state.fps        = 1.0 / max(dt, 0.001)
            if id_sim is not None:
                state.identity_sim     = id_sim
                state.identity_verdict = id_verdict

            # Blink
            if ear_val < EYE_THRESH and not blink_active:
                blink_active = True
                state.blink_count += 1
            elif ear_val >= EYE_THRESH:
                blink_active = False

            # Duration tracking
            if num_faces == 0:
                state.face_absent_dur += dt
            else:
                state.face_absent_dur = max(0, state.face_absent_dur - dt*0.5)

            if (abs(gaze_yaw) > YAW_THRESH or abs(gaze_pitch) > PITCH_THRESH) and num_faces > 0:
                state.gaze_off_dur += dt
            else:
                state.gaze_off_dur = max(0, state.gaze_off_dur - dt*0.5)

            if (abs(head_yaw) > YAW_THRESH or abs(head_pitch) > PITCH_THRESH) and num_faces > 0:
                state.head_off_dur += dt
            else:
                state.head_off_dur = max(0, state.head_off_dur - dt*0.5)

            now = time.time()
            # Violations
            for obj_name, conf, *_ in objects:
                if obj_name == "cell phone" and now - state.last_phone_t > 5:
                    state.integrity = max(0, state.integrity - 10)
                    state.violations.appendleft({'t': now, 'msg': f'Phone detected ({conf:.2f})', 'sev': 10})
                    state.last_phone_t = now
                elif obj_name in SEVERITY_MAP and obj_name != "cell phone":
                    sev = SEVERITY_MAP[obj_name]
                    state.integrity = max(0, state.integrity - sev * 0.1)

            if num_faces > 1:
                state.integrity = max(0, state.integrity - 20)
                state.violations.appendleft({'t': now, 'msg': f'Multiple faces: {num_faces}', 'sev': 20})

            if state.gaze_off_dur >= 3.0:
                state.integrity = max(0, state.integrity - 2)
                state.violations.appendleft({'t': now, 'msg': f'Gaze off {state.gaze_off_dur:.1f}s', 'sev': 2})
                state.gaze_off_dur = 0

            if state.head_off_dur >= 3.0:
                state.integrity = max(0, state.integrity - 2)
                state.violations.appendleft({'t': now, 'msg': f'Head turned {state.head_off_dur:.1f}s', 'sev': 2})
                state.head_off_dur = 0

            if id_verdict == "✗ DIFFERENT PERSON":
                state.integrity = max(0, state.integrity - 30)
                state.violations.appendleft({'t': now, 'msg': f'Identity mismatch! sim={id_sim:.3f}', 'sev': 30})

            # Slow recovery
            state.integrity = min(100, state.integrity + 0.01)

        # Store frame for display
        state._frame = frame.copy()

# ── Display loop ──────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state._frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    worker = threading.Thread(target=processing_loop, args=(cap,), daemon=True)
    worker.start()

    print("🎥  Full pipeline started!")
    print("   SPACE = capture identity baseline")
    print("   R = reset all  |  S = screenshot  |  Q = quit\n")

    BOX_COLORS = {"cell phone":(50,50,220), "laptop":(50,200,50), "tv":(50,200,200),
                  "book":(200,140,50), "keyboard":(180,50,180), "mouse":(180,180,50), "person":(200,200,200)}

    while True:
        frame = state._frame.copy()
        h, w = frame.shape[:2]

        with state.lock:
            objects    = list(state.objects)
            nf         = state.num_faces
            ear_v      = state.ear
            gy, gp     = state.gaze_yaw, state.gaze_pitch
            hy, hp     = state.head_yaw, state.head_pitch
            score      = state.integrity
            viols      = list(state.violations)
            blinks     = state.blink_count
            id_sim     = state.identity_sim
            id_verdict = state.identity_verdict
            fps        = state.fps
            capturing  = state.capturing_baseline
            baseline_set = state.baseline_emb is not None
            gaze_dur   = state.gaze_off_dur
            head_dur   = state.head_off_dur

        # Draw object boxes
        for (name, conf, x1, y1, x2, y2) in objects:
            col = BOX_COLORS.get(name, (200,200,200))
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # ── Score panel (top-right) ────────────────────────────────────────────
        px, py = w - 250, 0
        cv2.rectangle(frame, (px, py), (w, py + 240), (15,15,25), -1)

        score_col = (50,200,50) if score>=85 else (50,200,255) if score>=60 else (50,50,220)
        cv2.putText(frame, f"{int(score)}", (px+30, py+80), cv2.FONT_HERSHEY_DUPLEX, 2.5, score_col, 3)
        cv2.putText(frame, "INTEGRITY", (px+20, py+100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)

        def mini_stat(label, value, y_off, ok_cond):
            col = (50,200,50) if ok_cond else (50,50,220)
            cv2.putText(frame, f"{label}: {value}", (px+8, py+120+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        mini_stat("Faces",  nf,            0,  nf == 1)
        mini_stat("Blinks", blinks,       16,  True)
        mini_stat("Gaze°",  f"{gy:+.0f}", 32,  abs(gy) < YAW_THRESH)
        mini_stat("Head°",  f"{hy:+.0f}", 48,  abs(hy) < YAW_THRESH)
        mini_stat("EAR",    f"{ear_v:.2f}",64, ear_v > EYE_THRESH)

        id_col = (50,200,50) if baseline_set and "VERIFIED" in id_verdict else (50,50,220) if baseline_set else (150,150,150)
        cv2.putText(frame, id_verdict[:20], (px+8, py+210), cv2.FONT_HERSHEY_SIMPLEX, 0.38, id_col, 1)
        if id_sim is not None:
            cv2.putText(frame, f"sim={id_sim:.3f}", (px+8, py+226), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)

        # ── Violations list (bottom-right) ─────────────────────────────────────
        vx, vy = w - 340, h - 160
        cv2.rectangle(frame, (vx, vy), (w, h), (15,10,10), -1)
        cv2.putText(frame, "VIOLATIONS:", (vx+6, vy+16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,100,100), 1)
        for i, v in enumerate(viols[:5]):
            elapsed = time.time() - v['t']
            cv2.putText(frame, f"  {v['msg']} ({elapsed:.0f}s)",
                        (vx+6, vy+32+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180,80,80), 1)

        # ── Top bar ────────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (px, 55), (15,15,25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "TEST 9: Full Proctoring Pipeline",
                    (10,22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220,220,255), 1)
        active = []
        if YOLO_OK: active.append("YOLOv8")
        if MP_OK:   active.append("MediaPipe")
        if IFACE_OK:active.append("InsightFace")
        cv2.putText(frame, f"Active: {', '.join(active)}  |  FPS: {fps:.1f}",
                    (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,200), 1)

        # Baseline capture progress
        if capturing:
            prog = int(min(1.0, len(state.baseline_captures)/5.0) * (w-20))
            cv2.rectangle(frame, (10, h//2-25), (w-10, h//2+25), (20,30,20), -1)
            cv2.rectangle(frame, (10, h//2-25), (10+prog, h//2+25), (50,180,50), -1)
            cv2.putText(frame, "Capturing baseline… look at camera",
                        (20, h//2+8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

        # Alert if no baseline
        if not baseline_set and not capturing:
            cv2.putText(frame, "Press SPACE to capture identity baseline",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,100), 1)

        cv2.imshow("Proctoring — Test 9: Full Pipeline", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            with state.lock:
                state.capturing_baseline = True
                state.baseline_captures = []
            print("📸  Capturing identity baseline…")
        elif key == ord('r'):
            with state.lock:
                state.baseline_emb = None
                state.integrity    = 100.0
                state.violations.clear()
                state.blink_count  = 0
            print("🔄  Reset")
        elif key == ord('s'):
            fname = f"pipeline_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"\n📸  Screenshot saved: {fname}")
            with state.lock:
                print(f"\n📊  REPORT:")
                print(f"   Integrity score: {int(state.integrity)}/100")
                print(f"   Blinks: {state.blink_count}")
                print(f"   Violations: {len(state.violations)}")
                for v in list(state.violations)[:5]:
                    print(f"     - {v['msg']}")

    cap.release()
    cv2.destroyAllWindows()

    with state.lock:
        print(f"\n📊  FINAL REPORT")
        print(f"   Integrity Score: {int(state.integrity)}/100")
        print(f"   Total blinks:    {state.blink_count}")
        print(f"   Total violations: {len(state.violations)}")
    print("✅  Done.")

if __name__ == "__main__":
    main()