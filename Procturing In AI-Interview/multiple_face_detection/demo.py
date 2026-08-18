"""
Real-time Multiple Face Detection Demo
Detects extra unauthorized people in camera frame and triggers 5s continuous warnings
"""

import cv2
import sys
import numpy as np
from multiple_face_model import MultipleFaceDetector


class MultipleFaceDemo:
    """Demo for multiple face detection"""
    
    def __init__(self, camera_id: int = 0):
        """Initialize demo"""
        self.detector = MultipleFaceDetector()
        
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera. Please check if another camera app is active.")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_count = 0
        
    def run(self):
        """Run the demo"""
        print("Multiple Face Detection Demo Started")
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
            
            # Draw visualizations & bounding boxes
            frame = self._draw_visualization(frame, result, w, h)
            
            # Draw warnings and timer UI
            frame = self._draw_warnings(frame, result, w, h)
            
            # Frame counter
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Multiple Face Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
        
    def _draw_visualization(self, frame: np.ndarray, result: dict, frame_w: int, frame_h: int) -> np.ndarray:
        """Draw Big Centered Circle guide, face bounding boxes, and candidate labels"""
        
        # 1. Draw Big Centered Target Circle
        cx, cy = frame_w // 2, int(frame_h * 0.50)
        rx, ry = int(frame_w * 0.32), int(frame_h * 0.38)
        
        has_multiple = result.get('has_multiple_faces', False)
        guide_color = (0, 255, 0) if not has_multiple else (0, 165, 255)
        
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, guide_color, 2)
        cv2.drawMarker(frame, (cx, cy), guide_color, cv2.MARKER_CROSS, 20, 2)
        
        boxes = result.get('face_boxes', [])
        
        if not boxes:
            cv2.putText(frame, "NO FACE DETECTED", (frame_w // 2 - 90, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            return frame
        
        # Primary candidate (largest face)
        px, py, pw, ph = boxes[0]
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
        cv2.putText(frame, "PRIMARY CANDIDATE", (px, py - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Extra faces (unauthorized people / phone screen faces)
        for idx, (ex, ey, ew, eh) in enumerate(boxes[1:], start=2):
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
            cv2.putText(frame, f"UNAUTHORIZED PERSON #{idx}", (ex, max(15, ey - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            
        return frame


    def _draw_warnings(self, frame: np.ndarray, result: dict, frame_w: int, frame_h: int) -> np.ndarray:
        """Draw warning UI, status text, and 5s timer progress bar"""
        warning_count = result['warning_count']
        has_multiple = result['has_multiple_faces']
        face_count = result['face_count']
        duration = result.get('duration', 0.0)
        threshold = result.get('threshold', 5.0)
        
        # Determine status bar color and message
        if warning_count >= 5:
            color = (0, 0, 255)
            status_text = "REJECTED - MULTIPLE FACES DETECTED TOO MANY TIMES"
            status_color = (0, 0, 255)
        elif warning_count >= 4:
            color = (0, 0, 255)
            status_text = f"WARNING {warning_count}/5 - MULTIPLE FACES DETECTED!"
            status_color = (0, 0, 255)
        elif warning_count >= 1:
            color = (0, 165, 255)
            status_text = f"WARNING {warning_count}/5 - MULTIPLE FACES ({face_count} FACES)"
            status_color = (0, 165, 255)
        else:
            if has_multiple:
                color = (0, 165, 255)
                status_text = f"MULTIPLE FACES DETECTED ({face_count} FACES IN FRAME)"
                status_color = (0, 165, 255)
            else:
                color = (0, 255, 0)
                status_text = f"SINGLE CANDIDATE DETECTED ({face_count} FACE)"
                status_color = (0, 255, 0)
                
        # Outer border
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), color, 3 if has_multiple else 1)
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (frame_w, 55), color, -1)
        cv2.putText(frame, status_text, (10, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        # Warning counter box
        counter_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        counter_x = frame_w - text_size[0] - 15
        counter_y = 38
        
        cv2.rectangle(frame, (counter_x - 10, 5), (frame_w - 5, 50), status_color, -1)
        cv2.putText(frame, counter_text, (counter_x, counter_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Bottom live timer & progress bar if multiple faces present
        if has_multiple and warning_count < 5:
            timer_text = f"PLEASE ONLY SINGLE CANDIDATE IN FRAME: {duration:.1f}s / {threshold:.1f}s (Warning at {threshold:.1f}s)"
            cv2.rectangle(frame, (0, frame_h - 45), (frame_w, frame_h), (0, 0, 0), -1)
            
            progress_pct = min(1.0, duration / threshold)
            cv2.rectangle(frame, (0, frame_h - 8), (int(frame_w * progress_pct), frame_h), (0, 0, 255), -1)
            
            cv2.putText(frame, timer_text, (10, frame_h - 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 255), 2)
            
        # Rejection overlay
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
            
            reason_text = "Multiple people detected in interview frame"
            reason_size = cv2.getTextSize(reason_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
            reason_x = (frame_w - reason_size[0]) // 2
            reason_y = text_y + 40
            cv2.putText(frame, reason_text, (reason_x, reason_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            
        return frame

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Final Warning Count: {self.detector.warning_count}/5")
        if self.detector.warning_count >= 5:
            print("❌ INTERVIEW REJECTED - Multiple people detected")
        else:
            print(f"✓ Status: {5 - self.detector.warning_count} warnings remaining")


if __name__ == "__main__":
    demo = MultipleFaceDemo()
    demo.run()
