import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class PostureDetector:
    """
    Detects candidate posture violations using face and body cascade detection.
    Monitors for slouching, forward head position, leaning, and general misalignment.
    """
    
    def __init__(self):
        """Initialize cascade detectors and thresholds"""
        # Load cascade classifiers
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml'
        )
        self.upper_body_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_upperbody.xml'
        )
        
        # Warning tracking
        self.warning_count = 0
        self.max_warnings = 5
        
        # Frame counters for different violation types
        self.slouching_frame_count = 0
        self.forward_head_frame_count = 0
        self.leaning_frame_count = 0
        self.overall_misaligned_frame_count = 0
        self.no_pose_frame_count = 0  # NEW: Track frames when posture not detected
        
        # Thresholds
        self.slouching_threshold = 60  # ~2 seconds at 30 FPS
        self.forward_head_threshold = 60
        self.leaning_threshold = 60
        self.overall_threshold = 120  # ~4 seconds
        self.no_pose_threshold = 90  # ~3 seconds at 30 FPS (NEW)
        self.max_warnings = 5
        
        # Detection parameters
        self.forward_head_distance_threshold = 0.15  # Normalized distance
        self.slouching_angle_threshold = 40  # degrees
        self.leaning_angle_threshold = 0.20  # normalized
        self.vertical_deviation_threshold = 0.15  # normalized
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a frame and detect posture violations using cascade detectors
        
        Returns:
            dict: Detection results with keys:
                - success: bool
                - posture_status: str
                - warning_count: int
                - violations: dict
                - face_bbox: (x, y, w, h) or None
                - body_bbox: (x, y, w, h) or None
                - no_face_detected: bool
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces and bodies
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        bodies = self.upper_body_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        output = {
            'success': False,
            'posture_status': 'UNKNOWN',
            'warning_count': self.warning_count,
            'violations': {
                'slouching': False,
                'forward_head': False,
                'leaning': False,
                'overall_misaligned': False
            },
            'face_bbox': None,
            'body_bbox': None,
            'no_face_detected': False,
            'frame_dimensions': (h, w)
        }
        
        # Check if face is detected
        if len(faces) == 0:
            output['no_face_detected'] = True
            # Handle no-pose warning (3 seconds threshold)
            self.no_pose_frame_count += 1
            if self.no_pose_frame_count >= self.no_pose_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.no_pose_frame_count = 0
            output['warning_count'] = self.warning_count
            return output
        
        # Use the largest face detected
        face = max(faces, key=lambda f: f[2] * f[3])
        output['face_bbox'] = face
        
        # Reset no-pose counter since face is detected
        self.no_pose_frame_count = 0
        
        # Use the largest body detected (if any)
        if len(bodies) > 0:
            body = max(bodies, key=lambda b: b[2] * b[3])
            output['body_bbox'] = body
        
        output['success'] = True
        
        # Analyze posture
        violations = self._analyze_posture(frame, face, output['body_bbox'], w, h)
        output['violations'] = violations
        
        # Determine overall posture status
        posture_status = self._determine_posture_status(violations)
        output['posture_status'] = posture_status
        
        # Update warning count based on sustained violations
        output['warning_count'] = self._update_warnings(violations)
        
        return output
    
    def _analyze_posture(self, frame: np.ndarray, face_bbox: Tuple, 
                        body_bbox: Optional[Tuple], frame_w: int, frame_h: int) -> Dict:
        """Analyze posture from face and body bounding boxes"""
        violations = {
            'slouching': False,
            'forward_head': False,
            'leaning': False,
            'overall_misaligned': False
        }
        
        try:
            fx, fy, fw, fh = face_bbox
            
            # Calculate face center
            face_center_x = fx + fw / 2
            face_center_y = fy + fh / 2
            
            # Normalized face center (0-1)
            norm_face_x = face_center_x / frame_w
            norm_face_y = face_center_y / frame_h
            
            # Frame center (0.5, 0.5 in normalized coordinates)
            frame_center_x = 0.5
            frame_center_y = 0.5
            
            # 1. Forward Head Detection
            # Head position significantly forward if too far left/right from center
            horizontal_deviation = abs(norm_face_x - frame_center_x)
            if horizontal_deviation > self.forward_head_distance_threshold:
                violations['forward_head'] = True
            
            # 2. Leaning Detection - check if face is off-center horizontally
            # and if there's a body, check if it's also off-center
            if body_bbox is not None:
                bx, by, bw, bh = body_bbox
                body_center_x = bx + bw / 2
                norm_body_x = body_center_x / frame_w
                
                # If face and body centers differ significantly, person is leaning
                body_face_diff = abs(norm_face_x - norm_body_x)
                if body_face_diff > self.leaning_angle_threshold:
                    violations['leaning'] = True
            
            # 3. Slouching Detection
            # Face being too low relative to body suggests slouching
            if body_bbox is not None:
                bx, by, bw, bh = body_bbox
                # Body top is at 'by', face is at 'fy'
                # If face drops too low below body, it's slouching
                body_top_norm = by / frame_h
                face_top_norm = fy / frame_h
                
                # Slouching: face is lower than expected position relative to body
                # Calculate vertical deviation from center
                body_center_y = by + bh / 2
                norm_body_y = body_center_y / frame_h
                vertical_diff = abs(norm_face_y - norm_body_y)
                
                if vertical_diff > self.vertical_deviation_threshold:
                    violations['slouching'] = True
            
            # 4. Overall Misalignment
            # Face occupies too much or too little of frame indicates poor positioning
            face_size_ratio = (fw * fh) / (frame_w * frame_h)
            if face_size_ratio < 0.01 or face_size_ratio > 0.25:  # Face too small or too large
                violations['overall_misaligned'] = True
            
            # Also check if face is too far from center
            total_deviation = np.sqrt(
                (norm_face_x - frame_center_x) ** 2 + 
                (norm_face_y - frame_center_y) ** 2
            )
            if total_deviation > 0.4:  # Face too far off-center
                violations['overall_misaligned'] = True
            
        except (TypeError, ValueError, IndexError):
            pass
        
        return violations
    
    def _determine_posture_status(self, violations: Dict) -> str:
        """Determine overall posture status from violations"""
        violation_count = sum(1 for v in violations.values() if v)
        
        if violation_count == 0:
            return 'GOOD'
        elif violation_count == 1:
            if violations['slouching']:
                return 'SLIGHTLY_SLOUCHED'
            else:
                return 'SLIGHTLY_OFF'
        elif violation_count <= 2:
            return 'SLOUCHED'
        else:
            return 'OVERALL_MISALIGNED'
    
    def _update_warnings(self, violations: Dict) -> int:
        """Update warning count based on sustained posture violations"""
        
        # Track slouching
        if violations['slouching']:
            self.slouching_frame_count += 1
            if self.slouching_frame_count >= self.slouching_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.slouching_frame_count = 0
        else:
            self.slouching_frame_count = 0
        
        # Track forward head
        if violations['forward_head']:
            self.forward_head_frame_count += 1
            if self.forward_head_frame_count >= self.forward_head_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.forward_head_frame_count = 0
        else:
            self.forward_head_frame_count = 0
        
        # Track leaning
        if violations['leaning']:
            self.leaning_frame_count += 1
            if self.leaning_frame_count >= self.leaning_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.leaning_frame_count = 0
        else:
            self.leaning_frame_count = 0
        
        # Track overall misalignment
        if violations['overall_misaligned']:
            self.overall_misaligned_frame_count += 1
            if self.overall_misaligned_frame_count >= self.overall_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.overall_misaligned_frame_count = 0
        else:
            self.overall_misaligned_frame_count = 0
        
        return self.warning_count
    
    def reset(self):
        """Reset detector state"""
        self.warning_count = 0
        self.slouching_frame_count = 0
        self.forward_head_frame_count = 0
        self.leaning_frame_count = 0
        self.overall_misaligned_frame_count = 0
        self.no_pose_frame_count = 0
