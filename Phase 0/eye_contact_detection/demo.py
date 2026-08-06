"""
Real-time Eye Contact Detection Demo
Tests eye gaze tracking and visualizes output
"""

import cv2
import sys
import numpy as np
from eye_contact_model import EyeContactDetector


class EyeContactDemo:
    """Demo for eye contact detection"""
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize demo
        
        Args:
            camera_id: Webcam ID (default 0)
        """
        self.detector = EyeContactDetector()
        self.cap = cv2.VideoCapture(camera_id)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_count = 0
        self.gaze_directions = []
        
    def run(self):
        """Run the demo"""
        print("Eye Contact Detection Demo Started")
        print("Press 'q' to quit")
        print("=" * 60)
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror the frame
            h, w, c = frame.shape
            
            # Process frame
            result = self.detector.process_frame(frame)
            
            if result['success']:
                # Draw visualization
                frame = self._draw_visualization(frame, result, w, h)
                
                # Track gaze direction
                if result['left_gaze']:
                    gaze_dir = result['left_gaze']['direction']
                    self.gaze_directions.append(gaze_dir)
                
                # Draw warnings on frame
                frame = self._draw_warnings(frame, result, w, h)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display info
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Eye Contact Detection', frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def _draw_visualization(self, frame: np.ndarray, result: dict, 
                           frame_w: int, frame_h: int) -> np.ndarray:
        """Draw visualization on frame"""
        
        # Draw face box
        if result['face_box']:
            x, y, w, h = result['face_box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw eye boxes
        if result['eyes']:
            for (ex, ey, ew, eh) in result['eyes']:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        # Draw gaze information
        y_offset = 30
        
        # Left eye gaze
        if result['left_gaze']:
            left_gaze = result['left_gaze']
            cv2.putText(frame, f"Left Eye: {left_gaze['direction']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"  Horizontal: {left_gaze['horizontal_ratio']:.2f}", 
                       (10, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"  Vertical: {left_gaze['vertical_ratio']:.2f}", 
                       (10, y_offset + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Right eye gaze
        if result['right_gaze']:
            right_gaze = result['right_gaze']
            cv2.putText(frame, f"Right Eye: {right_gaze['direction']}", (10, y_offset + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"  Horizontal: {right_gaze['horizontal_ratio']:.2f}", 
                       (10, y_offset + 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"  Vertical: {right_gaze['vertical_ratio']:.2f}", 
                       (10, y_offset + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Head pose
        if result['head_pose']:
            head_pose = result['head_pose']
            cv2.putText(frame, f"Head Pose - Yaw: {head_pose['yaw']:.1f}°", 
                       (10, y_offset + 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Pitch: {head_pose['pitch']:.1f}° | Roll: {head_pose['roll']:.1f}°", 
                       (10, y_offset + 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw center reticle
        center_x, center_y = frame_w // 2, frame_h // 2
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
        cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), 
                      cv2.MARKER_CROSS, 15, 2)
        
        return frame
    
    def _draw_warnings(self, frame: np.ndarray, result: dict, 
                      frame_w: int, frame_h: int) -> np.ndarray:
        """Draw warning indicators on frame"""
        
        warning_count = result['warning_count']
        is_suspicious = result['is_suspicious']
        
        # Determine color based on warning level
        if warning_count >= 5:
            # REJECTED - Red with thick border
            color = (0, 0, 255)  # Red
            thickness = 4
            status_text = "⛔ REJECTED - TOO MANY WARNINGS"
            status_color = (0, 0, 255)
        elif warning_count >= 4:
            # Critical - Red
            color = (0, 0, 255)
            thickness = 3
            status_text = f"⚠️  WARNING {warning_count}/5 - FINAL WARNING"
            status_color = (0, 0, 255)
        elif warning_count >= 3:
            # High - Orange
            color = (0, 165, 255)
            thickness = 2
            status_text = f"⚠️  WARNING {warning_count}/5 - SERIOUS"
            status_color = (0, 165, 255)
        elif warning_count >= 1:
            # Moderate - Orange
            color = (0, 165, 255)
            thickness = 2
            status_text = f"⚠️  WARNING {warning_count}/5"
            status_color = (0, 165, 255)
        else:
            # No warning - Green
            color = (0, 255, 0)
            thickness = 1
            status_text = "✓ LOOKING AT CAMERA"
            status_color = (0, 255, 0)
        
        # Draw border around entire frame
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), color, thickness)
        
        # Draw warning status at top
        cv2.rectangle(frame, (0, 0), (frame_w, 60), color, -1)
        cv2.putText(frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw warning counter as large box
        warning_box_y = 80
        counter_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        counter_x = frame_w - text_size[0] - 20
        counter_y = warning_box_y + text_size[1] + 10
        
        # Background for counter
        cv2.rectangle(frame, (counter_x - 10, warning_box_y), 
                     (frame_w - 10, counter_y + 10), status_color, -1)
        cv2.putText(frame, counter_text, (counter_x, counter_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # If at max warnings, show rejection message
        if warning_count >= 5:
            # Draw rejection overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_h // 2 - 100), (frame_w, frame_h // 2 + 100), 
                         (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Draw rejection text
            rejection_text = "INTERVIEW TERMINATED"
            text_size = cv2.getTextSize(rejection_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (frame_w - text_size[0]) // 2
            text_y = frame_h // 2 + 20
            cv2.putText(frame, rejection_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            reason_text = "Too many violations detected"
            reason_size = cv2.getTextSize(reason_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            reason_x = (frame_w - reason_size[0]) // 2
            reason_y = text_y + 40
            cv2.putText(frame, reason_text, (reason_x, reason_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw suspicious indicator if currently suspicious
        if is_suspicious and warning_count < 5:
            cv2.putText(frame, "⚠️ LOOKING AWAY FROM CAMERA", (10, frame_h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return frame
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Final Warning Count: {self.detector.warning_count}/5")
        
        if self.detector.warning_count >= 5:
            print("\n❌ INTERVIEW REJECTED - Too many violations")
        else:
            print(f"\n✓ Interview Status: {5 - self.detector.warning_count} warnings remaining")
        
        if self.gaze_directions:
            print(f"\nGaze Direction Summary:")
            for direction in set(self.gaze_directions):
                count = self.gaze_directions.count(direction)
                percentage = (count / len(self.gaze_directions)) * 100
                print(f"  {direction}: {count} frames ({percentage:.1f}%)")


if __name__ == "__main__":
    demo = EyeContactDemo()
    demo.run()
