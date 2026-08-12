"""
Multiple Face Detection Model
Detects if more than one candidate/person is present in the camera frame
Uses Multi-Cascade Ensemble + CLAHE Contrast Enhancement for reliable phone screen & secondary face detection
"""

import cv2
import numpy as np
import time
from typing import Tuple, Dict, List


class MultipleFaceDetector:
    """
    Detects multiple faces in frame and triggers 5-second continuous warnings
    """
    
    def __init__(self):
        """Initialize face classifiers"""
        self.c_default = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.c_alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        self.c_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        # Contrast enhancer for phone screen faces
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
        # Warning tracking
        self.warning_count = 0
        self.max_warnings = 5
        self.multiple_faces_start_time = None
        self.multiple_faces_duration = 0.0
        self.multiple_faces_threshold_sec = 5.0  # 5.0 seconds continuous multiple faces triggers warning
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process frame and detect presence of multiple faces
        """
        current_time = time.time()
        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        
        output = {
            'success': True,
            'face_boxes': [],
            'face_count': 0,
            'has_multiple_faces': False,
            'warning_count': self.warning_count,
            'duration': 0.0,
            'threshold': self.multiple_faces_threshold_sec
        }
        
        raw_boxes = []
        
        # Pass 1: Standard Frontal Cascade on raw gray & enhanced gray
        f1 = self.c_default.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
        for (x, y, fw, fh) in f1:
            raw_boxes.append((x, y, fw, fh))
            
        f2 = self.c_alt2.detectMultiScale(enhanced_gray, scaleFactor=1.06, minNeighbors=3, minSize=(25, 25))
        for (x, y, fw, fh) in f2:
            raw_boxes.append((x, y, fw, fh))
            
        # Pass 2: Profile Cascade for tilted/side faces
        f3 = self.c_profile.detectMultiScale(enhanced_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        for (x, y, fw, fh) in f3:
            raw_boxes.append((x, y, fw, fh))
            
        # Merge overlapping boxes (NMS)
        merged_boxes = self._non_max_suppression(raw_boxes)
        
        face_count = len(merged_boxes)
        output['face_count'] = face_count
        
        # Sort face boxes by area (largest face = primary candidate)
        merged_boxes = sorted(merged_boxes, key=lambda b: b[2] * b[3], reverse=True)
        output['face_boxes'] = merged_boxes
        
        # Has multiple faces?
        has_multiple_faces = (face_count >= 2)
        output['has_multiple_faces'] = has_multiple_faces
        
        # Time-based continuous warning logic (5.0 seconds threshold)
        if has_multiple_faces:
            if self.multiple_faces_start_time is None:
                self.multiple_faces_start_time = current_time
            
            self.multiple_faces_duration = current_time - self.multiple_faces_start_time
            
            if self.multiple_faces_duration >= self.multiple_faces_threshold_sec:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                    print(f"\nWARNING #{self.warning_count}: Multiple faces detected for > {self.multiple_faces_threshold_sec}s!")
                self.multiple_faces_start_time = current_time
                self.multiple_faces_duration = 0.0
        else:
            # Single face or no face - reset duration timer immediately
            self.multiple_faces_start_time = None
            self.multiple_faces_duration = 0.0
            
        output['duration'] = self.multiple_faces_duration
        output['warning_count'] = self.warning_count
        
        return output

    def _non_max_suppression(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge bounding boxes that refer to the same face"""
        if not boxes:
            return []
            
        boxes_arr = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        pick = []
        
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w_overlap = np.maximum(0, xx2 - xx1 + 1)
            h_overlap = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w_overlap * h_overlap) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.35)[0])))
            
        final_boxes = []
        for p in pick:
            box = boxes_arr[p]
            final_boxes.append((int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])))
            
        return final_boxes

    def reset_warnings(self):
        """Reset warning counter"""
        self.warning_count = 0
        self.multiple_faces_start_time = None
        self.multiple_faces_duration = 0.0


