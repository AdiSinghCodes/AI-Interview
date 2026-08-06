"""
Eye Contact Detection Model using OpenCV Face Detection
Detects eye gaze direction and determines if candidate is looking at camera
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
from collections import deque


class EyeContactDetector:
    """
    Detects eye contact and gaze direction using OpenCV Haar Cascades
    """
    
    def __init__(self):
        """Initialize face and eye detectors"""
        # Load pre-trained Haar cascade classifiers
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.gaze_history = deque(maxlen=5)
        self.history_size = 5
        
        # Warning tracking
        self.warning_count = 0
        self.suspicious_frame_count = 0
        self.suspicious_threshold = 60  # ~2 seconds at 30 FPS
        self.max_warnings = 5
        self.head_pose_threshold = 15  # degrees
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process frame and detect eye contact
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Dictionary containing detection results
        """
        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        output = {
            'success': False,
            'left_gaze': None,
            'right_gaze': None,
            'head_pose': None,
            'face_center': None,
            'face_box': None,
            'eyes': None
        }
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return output
        
        # Get first face
        x, y, fw, fh = faces[0]
        output['face_box'] = (x, y, fw, fh)
        output['success'] = True
        
        # Get face center
        face_center = (x + fw // 2, y + fh // 2)
        output['face_center'] = face_center
        
        # Detect eyes in face region
        face_roi_gray = gray[y:y+fh, x:x+fw]
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
        
        if len(eyes) >= 1:
            # Convert eye coordinates to absolute frame coordinates
            abs_eyes = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
            output['eyes'] = abs_eyes[:2]  # Get up to 2 eyes
            
            # Calculate gaze for eyes
            if len(eyes) >= 1:
                left_eye = eyes[0]
                left_gaze = self._calculate_eye_gaze(
                    left_eye, face_region=(x, y, fw, fh), 
                    frame_size=(w, h)
                )
                output['left_gaze'] = left_gaze
            
            if len(eyes) >= 2:
                right_eye = eyes[1]
                right_gaze = self._calculate_eye_gaze(
                    right_eye, face_region=(x, y, fw, fh),
                    frame_size=(w, h)
                )
                output['right_gaze'] = right_gaze
            else:
                output['right_gaze'] = output['left_gaze']
        
        # Estimate head pose from face box
        head_pose = self._estimate_head_pose_simple(x, y, fw, fh, w, h)
        output['head_pose'] = head_pose
        
        # Check for suspicious behavior and update warnings
        is_suspicious = self._is_suspicious_behavior(output)
        output['is_suspicious'] = is_suspicious
        output['warning_count'] = self.warning_count
        
        return output
    
    def _calculate_eye_gaze(self, eye_box: Tuple, face_region: Tuple,
                           frame_size: Tuple) -> Dict:
        """
        Calculate gaze direction for an eye
        
        Args:
            eye_box: (ex, ey, ew, eh) relative to face region
            face_region: (fx, fy, fw, fh) absolute face position
            frame_size: (w, h) frame dimensions
        """
        ex, ey, ew, eh = eye_box
        fx, fy, fw, fh = face_region
        frame_w, frame_h = frame_size
        
        # Eye center in absolute frame coordinates
        eye_center_x_abs = fx + ex + ew // 2
        eye_center_y_abs = fy + ey + eh // 2
        
        # Normalize relative to face region (0-1 within face)
        horizontal_ratio = (ex + ew // 2) / fw if fw > 0 else 0.5
        vertical_ratio = (ey + eh // 2) / fh if fh > 0 else 0.5
        
        # Clamp values
        horizontal_ratio = np.clip(horizontal_ratio, 0, 1)
        vertical_ratio = np.clip(vertical_ratio, 0, 1)
        
        direction = self._determine_gaze_direction(horizontal_ratio, vertical_ratio)
        
        return {
            'direction': direction,
            'horizontal_ratio': horizontal_ratio,
            'vertical_ratio': vertical_ratio,
            'confidence': 0.7
        }
    
    def _determine_gaze_direction(self, horizontal: float, vertical: float) -> str:
        """Determine gaze direction from ratios"""
        # Thresholds for different gaze directions
        h_threshold_low = 0.3
        h_threshold_high = 0.7
        v_threshold_low = 0.2
        v_threshold_high = 0.8
        
        if h_threshold_low <= horizontal <= h_threshold_high and \
           v_threshold_low <= vertical <= v_threshold_high:
            return 'CENTER'
        elif horizontal < h_threshold_low:
            return 'LEFT'
        elif horizontal > h_threshold_high:
            return 'RIGHT'
        elif vertical < v_threshold_low:
            return 'UP'
        elif vertical > v_threshold_high:
            return 'DOWN'
        
        return 'CENTER'
    
    def _estimate_head_pose_simple(self, x: int, y: int, w: int, h: int,
                                  frame_w: int, frame_h: int) -> Dict:
        """Estimate head pose from face bounding box"""
        
        # Face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Frame center
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        # Calculate deviation from center
        yaw = (face_center_x - frame_center_x) / (frame_w / 2) * 30 if frame_w > 0 else 0
        pitch = (face_center_y - frame_center_y) / (frame_h / 2) * 30 if frame_h > 0 else 0
        
        # Aspect ratio gives roll hint
        aspect_ratio = w / h if h > 0 else 1
        roll = (aspect_ratio - 1) * 10
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll
        }
    
    def _is_suspicious_behavior(self, result: Dict) -> bool:
        """
        Detect suspicious behavior:
        - Eyes not looking at CENTER
        - Head turned away from camera
        
        Returns: True if suspicious, False if normal
        """
        is_suspicious = False
        
        # Check eye gaze
        if result['left_gaze'] and result['right_gaze']:
            left_direction = result['left_gaze']['direction']
            right_direction = result['right_gaze']['direction']
            
            # If either eye is looking LEFT, RIGHT, UP, or DOWN (not CENTER)
            if left_direction != 'CENTER' or right_direction != 'CENTER':
                is_suspicious = True
        
        # Check head pose
        if result['head_pose']:
            yaw = abs(result['head_pose']['yaw'])
            pitch = abs(result['head_pose']['pitch'])
            
            # If head is turned significantly away from camera
            if yaw > self.head_pose_threshold or pitch > self.head_pose_threshold:
                is_suspicious = True
        
        # Track suspicious frames
        if is_suspicious:
            self.suspicious_frame_count += 1
        else:
            self.suspicious_frame_count = 0
        
        # If suspicious for too long, increment warning
        if self.suspicious_frame_count >= self.suspicious_threshold:
            if self.warning_count < self.max_warnings:
                self.warning_count += 1
            self.suspicious_frame_count = 0  # Reset counter
        
        return is_suspicious
