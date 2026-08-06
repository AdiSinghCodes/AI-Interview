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
        self.misaligned_frame_count = 0
        self.no_face_frame_count = 0  # Track frames with no face detected
        self.misalignment_threshold = 120  # ~4 seconds at 30 FPS
        self.no_face_threshold = 60  # ~2 seconds at 30 FPS (faster warning) - was 90
        self.max_warnings = 5
        
        # Thresholds for face orientation (VERY LENIENT - only catch serious cheating)
        self.yaw_threshold = 45  # degrees (head turn left/right)
        self.pitch_threshold = 45  # degrees (head tilt up/down)
        self.face_size_threshold = 0.03  # Minimum % of frame face should occupy (very low) - was 0.08
        self.face_position_threshold = 0.40  # How far from center is acceptable
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process frame and detect face orientation
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Dictionary containing orientation results
        """
        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        output = {
            'success': False,
            'face_box': None,
            'face_center': None,
            'frame_center': None,
            'orientation': None,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'face_area_ratio': 0,
            'alignment': None,
            'is_misaligned': False,
            'warning_count': self.warning_count,
            'misalignment_reason': None,
            'no_face_detected': False
        }
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # No face detected - increment counter
            self.no_face_frame_count += 1
            output['no_face_detected'] = True
            
            # If no face for too long, trigger warning
            if self.no_face_frame_count >= self.no_face_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.no_face_frame_count = 0  # Reset counter
            
            output['warning_count'] = self.warning_count
            return output
        
        # Face detected - reset no_face counter
        self.no_face_frame_count = 0
        
        # Get largest face (closest to camera)
        x, y, fw, fh = max(faces, key=lambda face: face[2] * face[3])
        output['success'] = True
        output['face_box'] = (x, y, fw, fh)
        
        # Face center in absolute coordinates
        face_center_x = x + fw // 2
        face_center_y = y + fh // 2
        output['face_center'] = (face_center_x, face_center_y)
        
        # Frame center
        frame_center_x = w // 2
        frame_center_y = h // 2
        output['frame_center'] = (frame_center_x, frame_center_y)
        
        # Calculate orientation angles
        yaw, pitch, roll = self._calculate_orientation(x, y, fw, fh, w, h)
        output['yaw'] = yaw
        output['pitch'] = pitch
        output['roll'] = roll
        
        # Calculate face area ratio
        face_area = fw * fh
        frame_area = w * h
        face_area_ratio = face_area / frame_area
        output['face_area_ratio'] = face_area_ratio
        
        # Check alignment
        alignment_status, reason = self._check_alignment(
            face_center_x, face_center_y, face_area_ratio,
            yaw, pitch, roll, frame_center_x, frame_center_y, w, h
        )
        output['alignment'] = alignment_status
        output['misalignment_reason'] = reason
        
        # Determine if misaligned
        is_misaligned = alignment_status == 'MISALIGNED'  # Only actual MISALIGNED counts
        output['is_misaligned'] = is_misaligned
        
        # Track misalignment and update warnings
        if is_misaligned:
            self.misaligned_frame_count += 1
        else:
            self.misaligned_frame_count = 0  # Reset if not MISALIGNED
        
        # If misaligned for too long, increment warning
        if self.misaligned_frame_count >= self.misalignment_threshold:
            if self.warning_count < self.max_warnings:
                self.warning_count += 1
            self.misaligned_frame_count = 0  # Reset counter
        
        output['warning_count'] = self.warning_count
        
        return output
    
    def _calculate_orientation(self, x: int, y: int, w: int, h: int,
                              frame_w: int, frame_h: int) -> Tuple[float, float, float]:
        """
        Calculate head orientation angles using face bounding box
        
        Returns: (yaw, pitch, roll) in degrees
        """
        # Face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Frame center
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        # Calculate deviation from center
        # YAW: Left/Right rotation
        x_deviation = face_center_x - frame_center_x
        yaw = (x_deviation / (frame_w / 2)) * 45  # Scale to -45 to +45 degrees
        
        # PITCH: Up/Down tilt
        y_deviation = face_center_y - frame_center_y
        pitch = (y_deviation / (frame_h / 2)) * 45  # Scale to -45 to +45 degrees
        
        # ROLL: Head tilt (using aspect ratio)
        aspect_ratio = w / h if h > 0 else 1
        roll = (aspect_ratio - 1) * 30  # Scale to approximate degrees
        
        return yaw, pitch, roll
    
    def _check_alignment(self, face_center_x: int, face_center_y: int,
                        face_area_ratio: float, yaw: float, pitch: float, roll: float,
                        frame_center_x: int, frame_center_y: int,
                        frame_w: int, frame_h: int) -> Tuple[str, str]:
        """
        Check if face is properly aligned to camera
        
        Returns: (alignment_status, reason)
        - alignment_status: 'ALIGNED', 'SLIGHTLY_OFF', 'MISALIGNED'
        - reason: Explanation of misalignment
        
        LENIENT MODE: Only warn for significant deviations
        """
        reasons = []
        severity = 0  # 0=aligned, 1=slightly off, 2=misaligned
        
        # Check head rotation (Yaw) - ONLY severe deviations matter
        if abs(yaw) > self.yaw_threshold:  # > 35 degrees
            reasons.append(f"Head turned {'left' if yaw < 0 else 'right'} ({abs(yaw):.1f}°)")
            severity = 2  # Only serious turn counts
        elif abs(yaw) > self.yaw_threshold * 0.7:  # > 24.5 degrees
            reasons.append(f"Head slightly turned {'left' if yaw < 0 else 'right'}")
            severity = max(severity, 1)
        
        # Check head tilt (Pitch) - ONLY severe tilts matter
        if abs(pitch) > self.pitch_threshold:  # > 35 degrees
            reasons.append(f"Head tilted {'up' if pitch < 0 else 'down'} ({abs(pitch):.1f}°)")
            severity = 2  # Only serious tilt counts
        elif abs(pitch) > self.pitch_threshold * 0.7:  # > 24.5 degrees
            reasons.append(f"Head slightly tilted {'up' if pitch < 0 else 'down'}")
            severity = max(severity, 1)
        
        # Check face position (horizontally centered) - MORE LENIENT
        x_offset = abs(face_center_x - frame_center_x) / frame_w
        if x_offset > self.face_position_threshold:  # > 30%
            reasons.append(f"Face off-center horizontally ({x_offset:.2%})")
            severity = 2
        elif x_offset > self.face_position_threshold * 0.6:  # > 18%
            reasons.append(f"Face slightly off-center")
            severity = max(severity, 1)
        
        # Check face position (vertically centered) - MORE LENIENT
        y_offset = abs(face_center_y - frame_center_y) / frame_h
        if y_offset > self.face_position_threshold:  # > 30%
            reasons.append(f"Face off-center vertically ({y_offset:.2%})")
            severity = 2
        elif y_offset > self.face_position_threshold * 0.6:  # > 18%
            # Don't count vertical offset as heavily as horizontal
            pass
        
        # Check face size in frame - MUCH MORE LENIENT
        if face_area_ratio < self.face_size_threshold:  # < 10%
            reasons.append(f"Face too small ({face_area_ratio:.2%} of frame)")
            severity = 2
        
        # Determine status - bias towards ALIGNED
        if severity == 0:
            status = 'ALIGNED'
        elif severity == 1:
            status = 'SLIGHTLY_OFF'
        else:
            status = 'MISALIGNED'
        
        reason = " | ".join(reasons) if reasons else "Properly aligned"
        
        return status, reason
    
    def reset_warnings(self):
        """Reset warning counter"""
        self.warning_count = 0
        self.misaligned_frame_count = 0
