"""
Real-time Face Orientation Detection Demo
Tests face alignment and orientation towards camera
"""

import cv2
import sys
import numpy as np
from typing import Tuple
from face_orientation_model import FaceOrientationDetector


class FaceOrientationDemo:
    """Demo for face orientation detection"""
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize demo
        
        Args:
            camera_id: Webcam ID (default 0)
        """
        self.detector = FaceOrientationDetector()
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera. Please check if another camera index or app is active.")
            sys.exit(1)

        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_count = 0
        self.alignment_stats = {'ALIGNED': 0, 'SLIGHTLY_OFF': 0, 'MISALIGNED': 0}
    
    def run(self):
        """Run the demo"""
        print("Face Orientation Detection Demo Started")
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
                
                # Track alignment
                alignment = result['alignment']
                if alignment in self.alignment_stats:
                    self.alignment_stats[alignment] += 1
                
                # Draw warnings
                frame = self._draw_warnings(frame, result, w, h)
            else:
                # No face detected - show warning with proper styling
                if result['no_face_detected']:
                    frame = self._draw_no_face_warning(frame, result, w, h)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame counter
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Orientation Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def _draw_visualization(self, frame: np.ndarray, result: dict,
                           frame_w: int, frame_h: int) -> np.ndarray:
        """Draw orientation visualization and Big Centered Circle framing target shape on frame"""
        
        # 1. Draw Big Centered Circle Target Guide (Generous Head Boundary)
        cx, cy = frame_w // 2, int(frame_h * 0.50)
        rx, ry = int(frame_w * 0.32), int(frame_h * 0.38)  # Big generous circle
        
        is_misaligned = result.get('is_misaligned', True)
        guide_color = (0, 255, 0) if not is_misaligned else (0, 165, 255)
        guide_thickness = 3
        
        # Draw Big Oval/Circle
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, guide_color, guide_thickness)
        
        # Reticle Crosshair in center of Big Circle
        cv2.drawMarker(frame, (cx, cy), guide_color, cv2.MARKER_CROSS, 20, 2)
        
        guide_label = "KEEP FACE INSIDE CIRCLE" if not is_misaligned else "PLEASE KEEP YOUR FACE INSIDE THE CIRCLE"
        cv2.putText(frame, guide_label, (cx - 160, cy - ry - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)
        
        if not result['face_box']:
            return frame
        
        x, y, fw, fh = result['face_box']
        
        # Draw face bounding box
        color = self._get_alignment_color(result['alignment'])
        thickness = 2 if result['alignment'] == 'ALIGNED' else 3
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        
        # Draw face center point
        face_cx, face_cy = result['face_center']
        cv2.circle(frame, (face_cx, face_cy), 5, color, -1)
        
        # Display orientation info (Clean ASCII text without emoji artifacts)
        y_offset = 75
        cv2.putText(frame, f"Alignment: {result['alignment']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f"Yaw: {result['yaw']:.1f} | Pitch: {result['pitch']:.1f}",
                   (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        
        cv2.putText(frame, f"Roll: {result['roll']:.1f} | Face Size: {result['face_area_ratio']:.2%}",
                   (10, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        
        # Display reason for misalignment
        if result['misalignment_reason'] and is_misaligned:
            reason = result['misalignment_reason']
            cv2.putText(frame, f"Issue: {reason}", (10, y_offset + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        return frame
    
    def _draw_warnings(self, frame: np.ndarray, result: dict,
                      frame_w: int, frame_h: int) -> np.ndarray:
        """Draw warning indicators on frame"""
        
        warning_count = result['warning_count']
        is_misaligned = result['is_misaligned']
        out_duration = result.get('out_of_bounds_duration', 0.0)
        out_threshold = result.get('out_of_bounds_threshold', 5.0)
        
        # Determine color and status based on warning level
        if warning_count >= 5:
            color = (0, 0, 255)  # Red
            thickness = 4
            status_text = "REJECTED - TOO MANY WARNINGS"
            status_color = (0, 0, 255)
        elif warning_count >= 4:
            color = (0, 0, 255)  # Red
            thickness = 3
            status_text = f"WARNING {warning_count}/5 - FINAL WARNING"
            status_color = (0, 0, 255)
        elif warning_count >= 1:
            color = (0, 165, 255)  # Orange
            thickness = 2
            status_text = f"WARNING {warning_count}/5"
            status_color = (0, 165, 255)
        else:
            if is_misaligned:
                color = (0, 165, 255)
                thickness = 2
                status_text = f"PLEASE KEEP YOUR FACE INSIDE THE CIRCLE ({out_duration:.1f}s / {out_threshold:.1f}s)"
                status_color = (0, 165, 255)
            else:
                color = (0, 255, 0)  # Green
                thickness = 1
                status_text = "FACE PROPERLY INSIDE CIRCLE"
                status_color = (0, 255, 0)
        
        # Draw border around frame
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), color, thickness)
        
        # Draw status bar at top
        cv2.rectangle(frame, (0, 0), (frame_w, 55), color, -1)
        cv2.putText(frame, status_text, (10, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        # Draw warning counter
        counter_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        counter_x = frame_w - text_size[0] - 15
        counter_y = 38
        
        cv2.rectangle(frame, (counter_x - 10, 5), (frame_w - 5, 50), status_color, -1)
        cv2.putText(frame, counter_text, (counter_x, counter_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Live out-of-bounds timer & progress bar at bottom
        if is_misaligned and warning_count < 5:
            timer_text = f"PLEASE KEEP FACE INSIDE CIRCLE: {out_duration:.1f}s / {out_threshold:.1f}s (Warning at {out_threshold:.1f}s)"
            cv2.rectangle(frame, (0, frame_h - 45), (frame_w, frame_h), (0, 0, 0), -1)
            
            progress_pct = min(1.0, out_duration / out_threshold)
            cv2.rectangle(frame, (0, frame_h - 8), (int(frame_w * progress_pct), frame_h), (0, 165, 255), -1)
            
            cv2.putText(frame, timer_text, (15, frame_h - 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
        
        # If at max warnings, show rejection overlay
        if warning_count >= 5:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_h // 2 - 100), (frame_w, frame_h // 2 + 100),
                         (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            rejection_text = "INTERVIEW TERMINATED"
            text_size = cv2.getTextSize(rejection_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
            text_x = (frame_w - text_size[0]) // 2
            text_y = frame_h // 2 + 20
            cv2.putText(frame, rejection_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
            
            reason_text = "Face not kept inside target circle"
            reason_size = cv2.getTextSize(reason_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
            reason_x = (frame_w - reason_size[0]) // 2
            reason_y = text_y + 40
            cv2.putText(frame, reason_text, (reason_x, reason_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        
        return frame


    
    def _get_alignment_color(self, alignment: str) -> Tuple[int, int, int]:
        """Get color based on alignment status"""
        if alignment == 'ALIGNED':
            return (0, 255, 0)  # Green
        elif alignment == 'SLIGHTLY_OFF':
            return (0, 165, 255)  # Orange
        else:  # MISALIGNED
            return (0, 0, 255)  # Red
    
    def _draw_no_face_warning(self, frame: np.ndarray, result: dict,
                             frame_w: int, frame_h: int) -> np.ndarray:
        """Draw warning when no face is detected"""
        
        warning_count = result['warning_count']
        
        # Determine color based on warning level
        if warning_count >= 5:
            color = (0, 0, 255)  # Red
            status_text = "⛔ REJECTED - TOO MANY WARNINGS"
        elif warning_count >= 4:
            color = (0, 0, 255)  # Red
            status_text = f"⚠️  WARNING {warning_count}/5 - FACE MISSING"
        else:
            color = (0, 0, 255)  # Red
            status_text = f"⚠️  NO FACE DETECTED - WARNING {warning_count}/5"
        
        # Draw border around entire frame
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), color, 3)
        
        # Draw status bar at top
        cv2.rectangle(frame, (0, 0), (frame_w, 60), color, -1)
        cv2.putText(frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw warning counter
        counter_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        counter_x = frame_w - text_size[0] - 15
        counter_y = 45
        
        cv2.rectangle(frame, (counter_x - 10, 5), (frame_w - 5, 60), color, -1)
        cv2.putText(frame, counter_text, (counter_x, counter_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Draw center message
        center_text = "Please keep your face in frame"
        text_size = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (frame_w - text_size[0]) // 2
        text_y = frame_h // 2
        
        cv2.putText(frame, center_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # If at max warnings, show rejection overlay
        if warning_count >= 5:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_h // 2 - 100), (frame_w, frame_h // 2 + 100),
                         (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            rejection_text = "INTERVIEW TERMINATED"
            text_size = cv2.getTextSize(rejection_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (frame_w - text_size[0]) // 2
            text_y = frame_h // 2 + 20
            cv2.putText(frame, rejection_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return frame
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Final Warning Count: {self.detector.warning_count}/5")
        
        if self.detector.warning_count >= 5:
            print("\n❌ INTERVIEW REJECTED - Face not properly oriented")
        else:
            print(f"\n✓ Interview Status: {5 - self.detector.warning_count} warnings remaining")
        
        print(f"\nAlignment Statistics:")
        total = sum(self.alignment_stats.values())
        if total > 0:
            for status, count in self.alignment_stats.items():
                percentage = (count / total) * 100
                print(f"  {status}: {count} frames ({percentage:.1f}%)")


if __name__ == "__main__":
    demo = FaceOrientationDemo()
    demo.run()
