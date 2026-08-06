import cv2
import numpy as np
from posture_model import PostureDetector
from typing import Tuple, Dict


class PostureDemo:
    """Real-time posture detection demo with visualization"""
    
    def __init__(self):
        self.detector = PostureDetector()
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Statistics tracking
        self.posture_stats = {
            'GOOD': 0,
            'SLIGHTLY_SLOUCHED': 0,
            'SLIGHTLY_OFF': 0,
            'SLOUCHED': 0,
            'OVERALL_MISALIGNED': 0,
            'UNKNOWN': 0
        }
        self.frame_count = 0
    
    def run(self):
        """Main demo loop"""
        print("Posture Detection Demo Started")
        print("Press 'q' to quit")
        print("=" * 60)
        print()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Mirror frame for better UX
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Process frame
                result = self.detector.process_frame(frame)
                self.frame_count += 1
                
                if result['success']:
                    # Draw visualization
                    frame = self._draw_visualization(frame, result, w, h)
                    
                    # Track statistics
                    status = result['posture_status']
                    if status in self.posture_stats:
                        self.posture_stats[status] += 1
                    
                    # Draw warnings
                    frame = self._draw_warnings(frame, result, w, h)
                else:
                    # No pose detected - show warning
                    if result['no_face_detected']:
                        frame = self._draw_no_pose_warning(frame, result, w, h)
                
                # Display frame
                cv2.imshow('Posture Detection', frame)
                
                # Handle key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def _draw_visualization(self, frame: np.ndarray, result: Dict,
                           frame_w: int, frame_h: int) -> np.ndarray:
        """Draw face/body boxes and analysis"""
        
        # Draw face bounding box
        if result['face_bbox'] is not None:
            fx, fy, fw, fh = result['face_bbox']
            status = result['posture_status']
            color = self._get_status_color(status)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.putText(frame, "Face", (fx, fy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw body bounding box if detected
        if result['body_bbox'] is not None:
            bx, by, bw, bh = result['body_bbox']
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            cv2.putText(frame, "Body", (bx, by - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw posture status and violations
        status = result['posture_status']
        violations = result['violations']
        color = self._get_status_color(status)
        
        # Status text
        status_text = f"Posture: {status}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Violation details
        y_offset = 60
        if violations['forward_head']:
            cv2.putText(frame, "⚠ Forward Head", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_offset += 25
        if violations['slouching']:
            cv2.putText(frame, "⚠ Slouching", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_offset += 25
        if violations['leaning']:
            cv2.putText(frame, "⚠ Leaning", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_offset += 25
        
        return frame
    
    def _draw_warnings(self, frame: np.ndarray, result: Dict,
                      frame_w: int, frame_h: int) -> np.ndarray:
        """Draw warning indicators based on warning count"""
        
        warning_count = result['warning_count']
        posture_status = result['posture_status']
        
        # Determine border color based on warnings
        if warning_count >= 5:
            border_color = (0, 0, 255)  # Red
            bg_color = (0, 0, 255)
            text_color = (255, 255, 255)
            status_msg = "❌ INTERVIEW TERMINATED"
        elif warning_count >= 4:
            border_color = (0, 0, 255)  # Red
            bg_color = (0, 0, 255)
            text_color = (255, 255, 255)
            status_msg = f"🔴 WARNING {warning_count}/5"
        elif warning_count >= 2:
            border_color = (0, 165, 255)  # Orange
            bg_color = (0, 165, 255)
            text_color = (255, 255, 255)
            status_msg = f"🟠 WARNING {warning_count}/5"
        else:
            border_color = (0, 255, 0)  # Green
            bg_color = None
            text_color = (0, 255, 0)
            status_msg = f"✓ WARNING {warning_count}/5"
        
        # Draw border
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), border_color, 3)
        
        # Draw warning counter at top-right
        counter_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        counter_x = frame_w - text_size[0] - 15
        counter_y = 50
        
        if bg_color:
            cv2.rectangle(frame, (counter_x - 10, counter_y - 35),
                         (frame_w - 5, counter_y + 10), bg_color, -1)
            cv2.putText(frame, counter_text, (counter_x, counter_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        else:
            cv2.putText(frame, counter_text, (counter_x, counter_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)
        
        # Draw status message at bottom
        msg_size = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        msg_x = (frame_w - msg_size[0]) // 2
        msg_y = frame_h - 20
        cv2.putText(frame, status_msg, (msg_x, msg_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, border_color, 2)
        
        # Draw rejection overlay if terminated
        if warning_count >= 5:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_h // 2 - 80),
                         (frame_w, frame_h // 2 + 80), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            rejection_text = "INTERVIEW TERMINATED"
            text_size = cv2.getTextSize(rejection_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (frame_w - text_size[0]) // 2
            text_y = frame_h // 2 + 20
            cv2.putText(frame, rejection_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return frame
    
    def _draw_no_pose_warning(self, frame: np.ndarray, result: Dict,
                             frame_w: int, frame_h: int) -> np.ndarray:
        """Draw warning when posture is not detected"""
        
        warning_count = result['warning_count']
        
        # Draw border
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), (0, 0, 255), 3)
        
        # Draw status bar
        cv2.rectangle(frame, (0, 0), (frame_w, 60), (0, 0, 255), -1)
        cv2.putText(frame, "⚠ POSTURE NOT DETECTED", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw warning counter
        counter_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        counter_x = frame_w - text_size[0] - 15
        cv2.rectangle(frame, (counter_x - 10, 5), (frame_w - 5, 60), (0, 0, 255), -1)
        cv2.putText(frame, counter_text, (counter_x, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Draw center message
        center_text = "Position your full body in frame (keep body visible)"
        text_size = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (frame_w - text_size[0]) // 2
        text_y = frame_h // 2 - 20
        cv2.putText(frame, center_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        # Draw timer warning
        timer_text = "Warning in 3 seconds if body not detected"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
        text_x = (frame_w - text_size[0]) // 2
        text_y = frame_h // 2 + 20
        cv2.putText(frame, timer_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # If at max warnings, show rejection overlay
        if warning_count >= 5:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_h // 2 - 80),
                         (frame_w, frame_h // 2 + 80), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            rejection_text = "INTERVIEW TERMINATED"
            text_size = cv2.getTextSize(rejection_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (frame_w - text_size[0]) // 2
            text_y = frame_h // 2 + 20
            cv2.putText(frame, rejection_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return frame
    
    def _get_status_color(self, status: str) -> Tuple[int, int, int]:
        """Get color based on posture status"""
        if status == 'GOOD':
            return (0, 255, 0)  # Green
        elif status == 'SLIGHTLY_SLOUCHED' or status == 'SLIGHTLY_OFF':
            return (0, 165, 255)  # Orange
        else:  # SLOUCHED, OVERALL_MISALIGNED, etc.
            return (0, 0, 255)  # Red
    
    def cleanup(self):
        """Clean up resources and print statistics"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        print()
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Final Warning Count: {self.detector.warning_count}/5")
        print()
        
        if self.detector.warning_count >= 5:
            print("❌ INTERVIEW REJECTED - Poor posture detected")
        else:
            print(f"✓ Interview Status: {5 - self.detector.warning_count} warnings remaining")
        
        print()
        print("Posture Statistics:")
        for status, count in self.posture_stats.items():
            if count > 0:
                percentage = (count / self.frame_count * 100) if self.frame_count > 0 else 0
                print(f"  {status}: {count} frames ({percentage:.1f}%)")


if __name__ == "__main__":
    demo = PostureDemo()
    demo.run()


if __name__ == "__main__":
    demo = PostureDemo()
    demo.run()
