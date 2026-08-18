import cv2
import numpy as np
from object_model import ObjectDetector
from typing import Tuple, Dict


class ObjectDetectionDemo:
    """Real-time object and multi-person detection demo"""
    
    def __init__(self):
        self.detector = ObjectDetector()
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Statistics tracking
        self.detection_stats = {
            'single_person': 0,
            'multiple_people': 0,
            'hand_with_object': 0,  # CHANGED: Hand carrying object only
            'clean_frames': 0
        }
        self.frame_count = 0
    
    def run(self):
        """Main demo loop"""
        print("Object & Multi-Person Detection Demo Started")
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
                
                # Track statistics
                if result['people_count'] > 1:
                    self.detection_stats['multiple_people'] += 1
                elif result['people_count'] == 1:
                    self.detection_stats['single_person'] += 1
                
                # CHANGED: Only count when hand is carrying object (both detected)
                if result['hand_carrying_object']:
                    self.detection_stats['hand_with_object'] += 1
                
                if not any(result['violations'].values()):
                    self.detection_stats['clean_frames'] += 1
                
                # Draw visualization
                frame = self._draw_visualization(frame, result, w, h)
                
                # Draw warnings
                frame = self._draw_warnings(frame, result, w, h)
                
                # Display frame
                cv2.imshow('Object Detection', frame)
                
                # Handle key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def _draw_visualization(self, frame: np.ndarray, result: Dict,
                           frame_w: int, frame_h: int) -> np.ndarray:
        """Draw detection visualization"""
        
        # Detection info at top
        info_text = f"People: {result['people_count']} | Hand: {'Yes' if result['hand_detected'] else 'No'} | Object: {'Yes' if result['material_detected'] else 'No'}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Violation status (CHANGED: Only show violations we actually warn about)
        y_offset = 60
        if result['violations']['multiple_people']:
            cv2.putText(frame, "⚠ MULTIPLE PEOPLE DETECTED", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30
        
        # CHANGED: Show warning only when hand is CARRYING object
        if result['hand_carrying_object']:
            cv2.putText(frame, "⚠ HAND CARRYING OBJECT DETECTED", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            y_offset += 30
        
        # Status text
        if not any(result['violations'].values()):
            cv2.putText(frame, "✓ CLEAN FRAME", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def _draw_warnings(self, frame: np.ndarray, result: Dict,
                      frame_w: int, frame_h: int) -> np.ndarray:
        """Draw warning indicators"""
        
        warning_count = result['warning_count']
        
        # Determine border color
        if warning_count >= 5:
            border_color = (0, 0, 255)  # Red
            status_msg = "❌ INTERVIEW TERMINATED"
        elif warning_count >= 4:
            border_color = (0, 0, 255)  # Red
            status_msg = f"🔴 WARNING {warning_count}/5"
        elif warning_count >= 2:
            border_color = (0, 165, 255)  # Orange
            status_msg = f"🟠 WARNING {warning_count}/5"
        else:
            border_color = (0, 255, 0)  # Green
            status_msg = f"✓ WARNING {warning_count}/5"
        
        # Draw border
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), border_color, 3)
        
        # Draw warning counter
        counter_text = f"Warnings: {warning_count}/5"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        counter_x = frame_w - text_size[0] - 15
        counter_y = 50
        
        cv2.rectangle(frame, (counter_x - 10, counter_y - 35),
                     (frame_w - 5, counter_y + 10), border_color, -1)
        cv2.putText(frame, counter_text, (counter_x, counter_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
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
            print("❌ INTERVIEW REJECTED - Suspicious activity detected")
        else:
            print(f"✓ Interview Status: {5 - self.detector.warning_count} warnings remaining")
        
        print()
        print("Detection Statistics:")
        for stat, count in self.detection_stats.items():
            if count > 0:
                percentage = (count / self.frame_count * 100) if self.frame_count > 0 else 0
                print(f"  {stat}: {count} frames ({percentage:.1f}%)")


if __name__ == "__main__":
    demo = ObjectDetectionDemo()
    demo.run()
