try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import base64
from collections import deque
import time

class ProctoringEngine:
    def __init__(self):
        self.yolo_model = None
        self.face_mesh = None
        self.warning_count = 0
        self.max_warnings = 10
        self.violations = []
        
        # Gaze tracking settings
        self.gaze_yaw_history = deque(maxlen=10)
        self.gaze_pitch_history = deque(maxlen=10)
        self.gaze_yaw_thresh = 30
        self.gaze_pitch_thresh = 28
        
        # Head tracking settings
        self.face_x_history = deque(maxlen=10)
        self.face_y_history = deque(maxlen=10)
        self.center_left = 0.35
        self.center_right = 0.65
        self.center_top = 0.30
        self.center_bottom = 0.75
        
        # YOLO classes (only load what's needed)
        self.YOLO_CLASSES = {
            0: "person", 67: "cell phone", 62: "tv", 63: "laptop", 73: "book"
        }
        
        self._load_models()

    def _load_models(self):
        # Lazy loading of Ultralytics YOLO
        if self.yolo_model is None:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")
                print("[Proctoring] YOLOv8 loaded")
            except ImportError:
                print("[Proctoring] ultralytics not installed. YOLO proctoring disabled.")
            except Exception as e:
                print(f"Failed to load YOLO: {e}")
                
        # Lazy loading of MediaPipe FaceMesh
        if self.face_mesh is None:
            try:
                import mediapipe as mp
                try:
                    mp_face_mesh = mp.solutions.face_mesh
                except AttributeError:
                    import mediapipe.python.solutions.face_mesh as mp_face_mesh

                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[Proctoring] MediaPipe FaceMesh loaded")
            except ImportError:
                print("[Proctoring] mediapipe not installed. Gaze tracking disabled.")
            except Exception as e:
                print(f"Failed to load MediaPipe: {e}")
                
    def _is_object_in_hand(self, obj_box, person_boxes):
        if not person_boxes:
            return False
        
        obj_x1, obj_y1, obj_x2, obj_y2 = obj_box
        obj_area = (obj_x2 - obj_x1) * (obj_y2 - obj_y1)
        
        for person_x1, person_y1, person_x2, person_y2 in person_boxes:
            inter_x1 = max(obj_x1, person_x1)
            inter_y1 = max(obj_y1, person_y1)
            inter_x2 = min(obj_x2, person_x2)
            inter_y2 = min(obj_y2, person_y2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                if inter_area > obj_area * 0.3:
                    return True
        return False
        
    def _iris_gaze(self, landmarks, w, h):
        def pt(idx):
            l = landmarks[idx]
            return np.array([l.x * w, l.y * h])
            
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        LEFT_EYE_INNER = 362
        LEFT_EYE_OUTER = 263
        RIGHT_EYE_INNER = 133
        RIGHT_EYE_OUTER = 33

        left_iris_pts = [pt(i) for i in LEFT_IRIS]
        right_iris_pts = [pt(i) for i in RIGHT_IRIS]
        
        left_iris_c = np.mean(left_iris_pts, axis=0)
        right_iris_c = np.mean(right_iris_pts, axis=0)
        
        ll_inner = pt(LEFT_EYE_INNER)
        ll_outer = pt(LEFT_EYE_OUTER)
        rl_inner = pt(RIGHT_EYE_INNER)
        rl_outer = pt(RIGHT_EYE_OUTER)
        
        def horiz_ratio(iris, inner, outer):
            span = np.linalg.norm(outer - inner) + 1e-6
            return np.dot(iris - inner, outer - inner) / (span * span)
            
        left_ratio = horiz_ratio(left_iris_c, ll_inner, ll_outer)
        right_ratio = horiz_ratio(right_iris_c, rl_inner, rl_outer)
        
        avg_ratio = (left_ratio + right_ratio) / 2.0
        yaw = (avg_ratio - 0.5) * 220.0
        
        nose = pt(1)
        forehead = pt(10)
        eye_center_y = (left_iris_c[1] + right_iris_c[1]) / 2
        face_h = abs(nose[1] - forehead[1]) + 1e-6
        vert_ratio = (eye_center_y - forehead[1]) / face_h
        pitch = (vert_ratio - 0.5) * -80.0
        
        return yaw, pitch

    def process_frame(self, frame_data: str) -> dict:
        result = {
            "warning_count": self.warning_count,
            "messages": [],
            "status": "ok"
        }
        
        try:
            if cv2 is None:
                return result
                
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return result
                
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            person_boxes = []
            
            # YOLO Object Detection
            if self.yolo_model is not None:
                try:
                    yolo_results = self.yolo_model(frame, verbose=False, conf=0.45)
                    for box in yolo_results[0].boxes:
                        cls_id = int(box.cls)
                        label = self.yolo_model.names.get(cls_id, f"obj_{cls_id}")
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if label == "person":
                            person_boxes.append((x1, y1, x2, y2))
                        elif label in ["cell phone", "laptop", "book"]:
                            if self._is_object_in_hand((x1, y1, x2, y2), person_boxes):
                                result["messages"].append(f"Prohibited item detected: {label}")
                                self.warning_count += 1
                                self.violations.append(f"Item: {label}")
                except Exception as e:
                    print(f"YOLO processing error: {e}")
                    
            if len(person_boxes) > 1:
                result["messages"].append(f"Multiple people detected: {len(person_boxes)}")
                self.warning_count += 1
                self.violations.append("Multiple people")
            elif len(person_boxes) == 0 and self.yolo_model is not None:
                result["messages"].append("No person detected")
                
            # MediaPipe FaceMesh
            if self.face_mesh is not None:
                try:
                    results = self.face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        lm = results.multi_face_landmarks[0].landmark
                        
                        # Gaze
                        yaw, pitch = self._iris_gaze(lm, w, h)
                        self.gaze_yaw_history.append(yaw)
                        self.gaze_pitch_history.append(pitch)
                        
                        avg_yaw = np.mean(self.gaze_yaw_history)
                        avg_pitch = np.mean(self.gaze_pitch_history)
                        
                        looking_away = (abs(avg_yaw) > self.gaze_yaw_thresh and abs(avg_pitch) > 10)
                        
                        # Head position
                        nose = lm[1]
                        self.face_x_history.append(nose.x)
                        self.face_y_history.append(nose.y)
                        
                        avg_x = np.mean(self.face_x_history)
                        avg_y = np.mean(self.face_y_history)
                        
                        head_turned = (avg_x < self.center_left or avg_x > self.center_right or 
                                       avg_y < self.center_top or avg_y > self.center_bottom)
                                       
                        if head_turned:
                            result["messages"].append("Face out of center")
                        elif looking_away:
                            result["messages"].append("Looking away from screen")
                except Exception as e:
                    print(f"MediaPipe processing error: {e}")
                    
            result["warning_count"] = self.warning_count
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return result
