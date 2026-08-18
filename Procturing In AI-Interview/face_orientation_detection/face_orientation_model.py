"""
Face Orientation Detection Model
Detects if candidate's face is properly oriented towards camera
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
from collections import deque


class FaceOrientationDetector:
    """
    Detects face orientation and alignment to camera
    """
    
    def __init__(self):
        """Initialize face detector"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Warning tracking
        self.warning_count = 0
        self.max_warnings = 5
        self.out_of_bounds_start_time = None
        self.out_of_bounds_duration = 0.0
        self.out_of_bounds_threshold_sec = 5.0  # Exactly 5.0 seconds out-of-bounds triggers warning
        
        # Thresholds for face orientation
        self.yaw_threshold = 40  # degrees (head turn left/right)
        self.pitch_threshold = 40  # degrees (head tilt up/down)
        self.face_size_threshold = 0.03  # Minimum % of frame face should occupy
        self.face_position_threshold = 0.35  # Max offset from center allowed
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process frame and detect face orientation & position relative to framing guide
        """
        import time
        current_time = time.time()
        
        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        output = {
            'success': False,
            'face_box': None,
            'face_center': None,
            'frame_center': (w // 2, h // 2),
            'orientation': None,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'face_area_ratio': 0,
            'alignment': 'MISALIGNED',
            'is_misaligned': True,
            'warning_count': self.warning_count,
            'misalignment_reason': "No face detected",
            'no_face_detected': True,
            'out_of_bounds_duration': 0.0,
            'out_of_bounds_threshold': self.out_of_bounds_threshold_sec
        }
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            output['no_face_detected'] = True
            output['is_misaligned'] = True
            output['misalignment_reason'] = "Face out of camera frame"
            self._update_time_warnings(True, current_time)
            output['out_of_bounds_duration'] = self.out_of_bounds_duration
            output['warning_count'] = self.warning_count
            return output
        
        # Get largest face
        x, y, fw, fh = max(faces, key=lambda face: face[2] * face[3])
        output['success'] = True
        output['no_face_detected'] = False
        output['face_box'] = (x, y, fw, fh)
        
        # Face center in absolute coordinates
        face_center_x = x + fw // 2
        face_center_y = y + fh // 2
        output['face_center'] = (face_center_x, face_center_y)
        
        frame_center_x = w // 2
        frame_center_y = h // 2
        
        # Calculate orientation angles
        yaw, pitch, roll = self._calculate_orientation(x, y, fw, fh, w, h)
        output['yaw'] = yaw
        output['pitch'] = pitch
        output['roll'] = roll
        
        # Face area ratio
        face_area_ratio = (fw * fh) / (w * h)
        output['face_area_ratio'] = face_area_ratio
        
        # Check alignment relative to Big Circle framing guide
        alignment_status, reason = self._check_alignment(
            (x, y, fw, fh), face_center_x, face_center_y, face_area_ratio,
            yaw, pitch, roll, frame_center_x, frame_center_y, w, h
        )
        output['alignment'] = alignment_status
        output['misalignment_reason'] = reason
        
        # Is face out of the target zone/shape or misaligned?
        is_misaligned = (alignment_status == 'MISALIGNED')
        output['is_misaligned'] = is_misaligned
        
        # Update 5-second continuous out-of-bounds timer
        self._update_time_warnings(is_misaligned, current_time)
        output['out_of_bounds_duration'] = self.out_of_bounds_duration
        output['warning_count'] = self.warning_count
        
        return output
    
    def _update_time_warnings(self, is_misaligned: bool, current_time: float):
        if is_misaligned:
            if self.out_of_bounds_start_time is None:
                self.out_of_bounds_start_time = current_time
            
            self.out_of_bounds_duration = current_time - self.out_of_bounds_start_time
            
            # If face is out of shape/circle for more than 5.0 seconds, trigger 1 warning!
            if self.out_of_bounds_duration >= self.out_of_bounds_threshold_sec:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                    print(f"\nWARNING #{self.warning_count}: Face out of circle for > {self.out_of_bounds_threshold_sec}s!")
                self.out_of_bounds_start_time = current_time
                self.out_of_bounds_duration = 0.0
        else:
            # Face properly inside circle - reset timer immediately
            self.out_of_bounds_start_time = None
            self.out_of_bounds_duration = 0.0

    def _calculate_orientation(self, x: int, y: int, w: int, h: int,
                              frame_w: int, frame_h: int) -> Tuple[float, float, float]:
        """
        Calculate head orientation angles using face bounding box
        """
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        x_deviation = face_center_x - frame_center_x
        yaw = (x_deviation / (frame_w / 2)) * 45
        
        y_deviation = face_center_y - frame_center_y
        pitch = (y_deviation / (frame_h / 2)) * 45
        
        aspect_ratio = w / h if h > 0 else 1
        roll = (aspect_ratio - 1) * 30
        
        return yaw, pitch, roll
    
    def _check_alignment(self, face_box: Tuple, face_center_x: int, face_center_y: int,
                        face_area_ratio: float, yaw: float, pitch: float, roll: float,
                        frame_center_x: int, frame_center_y: int,
                        frame_w: int, frame_h: int) -> Tuple[str, str]:
        """
        Check if face is inside the Big Centered Circle Target Zone.
        """
        reasons = []
        
        # Big Circle Parameters (Centered in video frame, covering generous head space)
        circle_cx = frame_w // 2
        circle_cy = int(frame_h * 0.50)
        rx = int(frame_w * 0.32)  # Generous width radius (~64% of frame width)
        ry = int(frame_h * 0.38)  # Generous height radius (~76% of frame height)
        
        # Check normalized ellipse equation for face center
        dx = (face_center_x - circle_cx) / rx
        dy = (face_center_y - circle_cy) / ry
        dist_sq = dx*dx + dy*dy
        
        # Check if face box boundaries extend significantly beyond the Big Circle
        x, y, fw, fh = face_box
        top_dy = (y - circle_cy) / ry
        bottom_dy = (y + fh - circle_cy) / ry
        left_dx = (x - circle_cx) / rx
        right_dx = (x + fw - circle_cx) / rx
        
        is_outside = (dist_sq > 1.0) or (left_dx < -1.15) or (right_dx > 1.15) or (top_dy < -1.15) or (bottom_dy > 1.15)
        
        if is_outside:
            reasons.append("Please keep your face inside the circle")
        
        if abs(yaw) > self.yaw_threshold:
            reasons.append(f"Head turned {'left' if yaw < 0 else 'right'}")
        if abs(pitch) > self.pitch_threshold:
            reasons.append(f"Head tilted {'up' if pitch < 0 else 'down'}")
            
        if reasons:
            return 'MISALIGNED', " | ".join(reasons)
        
        return 'ALIGNED', "Face inside circle"
    
    def reset_warnings(self):
        """Reset warning counter"""
        self.warning_count = 0
        self.out_of_bounds_start_time = None
        self.out_of_bounds_duration = 0.0

