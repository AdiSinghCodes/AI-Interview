import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path to import detectors
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / 'eye_contact_detection'))
sys.path.insert(0, str(parent_dir / 'face_orientation_detection'))
sys.path.insert(0, str(parent_dir / 'posture_detection'))
sys.path.insert(0, str(parent_dir / 'object_detection'))

from eye_contact_model import EyeContactDetector
from face_orientation_model import FaceOrientationDetector
from posture_model import PostureDetector
from object_model import ObjectDetector


class IntegratedInterviewSystem:
    """
    Unified interview cheating detection system combining all 4 phases:
    - Phase 0: Eye Contact Detection
    - Phase 1: Face Orientation Detection
    - Phase 2: Posture Detection
    - Phase 3: Object Detection
    
    Single shared warning system: 0-5 warnings
    """
    
    def __init__(self):
        """Initialize all 4 detectors"""
        print("Initializing Integrated Interview System...")
        print("Loading Phase 0: Eye Contact Detection...")
        self.eye_contact_detector = EyeContactDetector()
        
        print("Loading Phase 1: Face Orientation Detection...")
        self.face_orientation_detector = FaceOrientationDetector()
        
        print("Loading Phase 2: Posture Detection...")
        self.posture_detector = PostureDetector()
        
        print("Loading Phase 3: Object Detection...")
        self.object_detector = ObjectDetector()
        
        # Shared warning system
        self.warning_count = 0
        self.max_warnings = 5
        
        # Statistics tracking
        self.violation_stats = {
            'eye_contact': 0,
            'face_orientation': 0,
            'posture': 0,
            'object_detection': 0,
            'clean_frames': 0
        }
        
        self.frame_count = 0
        print("✓ All detectors loaded successfully!")
        print()
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process frame through all 4 detection phases
        
        Returns:
            dict: Unified detection results with all violations
        """
        self.frame_count += 1
        
        output = {
            'frame_count': self.frame_count,
            'warning_count': self.warning_count,
            'interview_terminated': self.warning_count >= self.max_warnings,
            'violations': {
                'eye_contact': False,
                'face_orientation': False,
                'posture': False,
                'object_detection': False
            },
            'details': {}
        }
        
        # PHASE 0: Eye Contact Detection
        eye_result = self.eye_contact_detector.process_frame(frame)
        if eye_result['gaze_direction'] != 'CENTER':
            output['violations']['eye_contact'] = True
            self.violation_stats['eye_contact'] += 1
        output['details']['eye_contact'] = {
            'gaze_direction': eye_result['gaze_direction'],
            'horizontal_ratio': eye_result['horizontal_ratio'],
            'vertical_ratio': eye_result['vertical_ratio']
        }
        
        # PHASE 1: Face Orientation Detection
        face_result = self.face_orientation_detector.process_frame(frame)
        if face_result['alignment_status'] != 'ALIGNED':
            output['violations']['face_orientation'] = True
            self.violation_stats['face_orientation'] += 1
        elif face_result['no_face_detected']:
            output['violations']['face_orientation'] = True
            self.violation_stats['face_orientation'] += 1
        output['details']['face_orientation'] = {
            'alignment': face_result['alignment_status'],
            'yaw': face_result['yaw'],
            'pitch': face_result['pitch'],
            'no_face': face_result['no_face_detected']
        }
        
        # PHASE 2: Posture Detection
        posture_result = self.posture_detector.process_frame(frame)
        if posture_result['posture_status'] not in ['GOOD', 'SLIGHTLY_SLOUCHED']:
            output['violations']['posture'] = True
            self.violation_stats['posture'] += 1
        elif posture_result['no_pose_detected']:
            output['violations']['posture'] = True
            self.violation_stats['posture'] += 1
        output['details']['posture'] = {
            'status': posture_result['posture_status'],
            'no_pose': posture_result['no_pose_detected']
        }
        
        # PHASE 3: Object Detection
        object_result = self.object_detector.process_frame(frame)
        if object_result['hand_carrying_object']:
            output['violations']['object_detection'] = True
            self.violation_stats['object_detection'] += 1
        elif object_result['people_count'] >= 2:
            output['violations']['object_detection'] = True
            self.violation_stats['object_detection'] += 1
        output['details']['object_detection'] = {
            'people_count': object_result['people_count'],
            'hand_carrying_object': object_result['hand_carrying_object'],
            'hand_detected': object_result['hand_detected'],
            'material_detected': object_result['material_detected']
        }
        
        # Update unified warning count
        self._update_warnings(output['violations'])
        output['warning_count'] = self.warning_count
        output['interview_terminated'] = self.warning_count >= self.max_warnings
        
        # Track clean frames
        if not any(output['violations'].values()):
            self.violation_stats['clean_frames'] += 1
        
        return output
    
    def _update_warnings(self, violations: Dict) -> None:
        """
        Update unified warning count based on violations
        
        Logic: Each violation type can trigger ONE warning per cycle
        If multiple violations in same frame, each contributes independently
        """
        violation_count_this_frame = sum(1 for v in violations.values() if v)
        
        if violation_count_this_frame > 0:
            # For demo purposes: add 1 warning per frame with violations
            # In production, could track sustained violations like individual phases
            if self.warning_count < self.max_warnings:
                # Could add multiple per frame, but typically 1 warning per violation type
                self.warning_count += min(violation_count_this_frame, 1)
    
    def get_statistics(self) -> Dict:
        """Get session statistics"""
        return {
            'total_frames': self.frame_count,
            'final_warnings': self.warning_count,
            'interview_result': 'REJECTED' if self.warning_count >= self.max_warnings else 'APPROVED',
            'violations_breakdown': self.violation_stats,
            'clean_frame_percentage': (self.violation_stats['clean_frames'] / max(self.frame_count, 1)) * 100
        }
    
    def reset(self):
        """Reset all detectors"""
        self.warning_count = 0
        self.frame_count = 0
        self.violation_stats = {
            'eye_contact': 0,
            'face_orientation': 0,
            'posture': 0,
            'object_detection': 0,
            'clean_frames': 0
        }
        self.eye_contact_detector.reset()
        self.face_orientation_detector.reset()
        self.posture_detector.reset()
        self.object_detector.reset()
