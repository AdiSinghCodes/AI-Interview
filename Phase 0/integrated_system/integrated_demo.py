import cv2
import numpy as np
from integrated_model import IntegratedInterviewSystem
from typing import Tuple


class IntegratedInterviewDemo:
    """Real-time unified interview monitoring system"""
    
    def __init__(self):
        self.system = IntegratedInterviewSystem()
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def run(self):
        """Main demo loop"""
        print("=" * 70)
        print("INTEGRATED INTERVIEW CHEATING DETECTION SYSTEM")
        print("=" * 70)
        print()
        print("Monitoring 4 Phases Simultaneously:")
        print("  Phase 0: Eye Contact Detection")
        print("  Phase 1: Face Orientation Detection")
        print("  Phase 2: Posture Detection")
        print("  Phase 3: Object Detection")
        print()
        print("Single Warning Counter: 0/5")
        print("Press 'q' to quit and see results")
        print("=" * 70)
        print()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Process through integrated system
                result = self.system.process_frame(frame)
                
                # Draw unified visualization
                frame = self._draw_unified_dashboard(frame, result, w, h)
                
                # Display
                cv2.imshow('Integrated Interview System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def _draw_unified_dashboard(self, frame: np.ndarray, result: dict,
                                frame_w: int, frame_h: int) -> np.ndarray:
        """Draw unified dashboard with all 4 detections"""
        
        # Color scheme based on warnings
        warning_count = result['warning_count']
        if warning_count >= 5:
            main_color = (0, 0, 255)  # Red
            status_color = (0, 0, 255)
        elif warning_count >= 3:
            main_color = (0, 165, 255)  # Orange
            status_color = (0, 165, 255)
        elif warning_count >= 1:
            main_color = (0, 255, 255)  # Yellow
            status_color = (0, 255, 255)
        else:
            main_color = (0, 255, 0)  # Green
            status_color = (0, 255, 0)
        
        # ===== TOP SECTION: Frame Counter & Warning Counter =====
        cv2.putText(frame, f"Frame: {result['frame_count']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Large warning counter (top right)
        warning_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        warning_x = frame_w - text_size[0] - 20
        cv2.rectangle(frame, (warning_x - 10, 10), (frame_w - 5, 60), main_color, -1)
        cv2.putText(frame, warning_text, (warning_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # ===== LEFT COLUMN: Phase 0 & Phase 1 =====
        y_offset = 80
        line_height = 35
        
        # Phase 0: Eye Contact
        phase0_status = "✓" if not result['violations']['eye_contact'] else "✗"
        phase0_color = (0, 255, 0) if not result['violations']['eye_contact'] else (0, 0, 255)
        gaze = result['details']['eye_contact']['gaze_direction']
        cv2.putText(frame, f"{phase0_status} Phase 0: Eye Contact - {gaze}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase0_color, 2)
        y_offset += line_height
        
        # Phase 1: Face Orientation
        phase1_status = "✓" if not result['violations']['face_orientation'] else "✗"
        phase1_color = (0, 255, 0) if not result['violations']['face_orientation'] else (0, 0, 255)
        alignment = result['details']['face_orientation']['alignment']
        no_face = result['details']['face_orientation']['no_face']
        face_status = "NO FACE" if no_face else alignment
        cv2.putText(frame, f"{phase1_status} Phase 1: Face Orientation - {face_status}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase1_color, 2)
        y_offset += line_height
        
        # Phase 2: Posture
        phase2_status = "✓" if not result['violations']['posture'] else "✗"
        phase2_color = (0, 255, 0) if not result['violations']['posture'] else (0, 0, 255)
        posture = result['details']['posture']['status']
        no_pose = result['details']['posture']['no_pose']
        posture_status = "NO POSE" if no_pose else posture
        cv2.putText(frame, f"{phase2_status} Phase 2: Posture - {posture_status}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase2_color, 2)
        y_offset += line_height
        
        # Phase 3: Object Detection
        phase3_status = "✓" if not result['violations']['object_detection'] else "✗"
        phase3_color = (0, 255, 0) if not result['violations']['object_detection'] else (0, 0, 255)
        people = result['details']['object_detection']['people_count']
        hand_with_obj = result['details']['object_detection']['hand_carrying_object']
        obj_status = f"People:{people}"
        if hand_with_obj:
            obj_status += " | HAND+OBJECT"
        cv2.putText(frame, f"{phase3_status} Phase 3: Object Detection - {obj_status}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase3_color, 2)
        y_offset += line_height
        
        # ===== VIOLATION SUMMARY =====
        y_offset += 10
        cv2.line(frame, (10, y_offset), (frame_w - 10, y_offset), (200, 200, 200), 1)
        y_offset += 20
        
        violations_found = sum(1 for v in result['violations'].values() if v)
        if violations_found == 0:
            cv2.putText(frame, "Status: ✓ CLEAN FRAME", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            violation_text = f"Violations This Frame: {violations_found}"
            cv2.putText(frame, violation_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # ===== BOTTOM: Interview Status =====
        status_y = frame_h - 60
        if warning_count >= 5:
            # REJECTION overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, status_y - 80), (frame_w, frame_h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            cv2.putText(frame, "❌ INTERVIEW TERMINATED", (50, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, "Suspicious Activity Detected - Candidate Rejected",
                       (50, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            warnings_remaining = 5 - warning_count
            status_text = f"✓ Interview Active | {warnings_remaining} warning(s) remaining"
            cv2.putText(frame, status_text, (20, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # ===== BORDER based on warning level =====
        border_thickness = 3
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), main_color, border_thickness)
        
        return frame
    
    def cleanup(self):
        """Clean up and print statistics"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        stats = self.system.get_statistics()
        
        print()
        print("=" * 70)
        print("SESSION COMPLETED")
        print("=" * 70)
        print()
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Final Warning Count: {stats['final_warnings']}/5")
        print()
        
        if stats['interview_result'] == 'REJECTED':
            print("❌ INTERVIEW RESULT: REJECTED")
            print("Candidate exhibited suspicious behavior exceeding acceptable threshold")
        else:
            print(f"✓ INTERVIEW RESULT: APPROVED")
            print(f"Candidate maintained acceptable behavior throughout interview")
        
        print()
        print("Violations Breakdown:")
        for violation_type, count in stats['violations_breakdown'].items():
            if count > 0 or violation_type == 'clean_frames':
                percentage = (count / stats['total_frames'] * 100) if stats['total_frames'] > 0 else 0
                print(f"  {violation_type}: {count} ({percentage:.1f}%)")
        
        print()
        print(f"Clean Frame Percentage: {stats['clean_frame_percentage']:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    demo = IntegratedInterviewDemo()
    demo.run()
