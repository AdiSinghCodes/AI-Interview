import time
import cv2
import numpy as np
from integrated_model import IntegratedInterviewSystem
from typing import Tuple

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class IntegratedInterviewDemo:
    """Real-time unified interview monitoring system"""

    def __init__(self):
        self.system = IntegratedInterviewSystem()
        self.cap = cv2.VideoCapture(0)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.audio_stream = None
        self.active_voice_message = None
        self.voice_message_time = 0.0
        self.voice_message_display_sec = 4.0

        # How long to hold the "INTERVIEW TERMINATED" banner on screen
        # before the session auto-stops.
        self.termination_hold_sec = 3.0
        self.terminated_at = None

        # How long a live caption stays on screen after it's transcribed
        # (transcription only completes once the candidate pauses, so this
        # is intentionally a bit generous).
        self.caption_display_sec = 6.0

        if AUDIO_AVAILABLE:
            try:
                self.audio_stream = sd.InputStream(
                    samplerate=self.system.voice_detector.SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=self.system.voice_detector.CHUNK_SIZE,
                    callback=self._audio_callback,
                )
            except Exception as e:
                print(f"Warning: could not open microphone ({e}). Running without voice clarity.")
                self.audio_stream = None
        else:
            print("Warning: sounddevice not installed. Running without voice clarity.")
            print("         Install with: pip install sounddevice")

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0]
        result = self.system.process_audio_chunk(chunk)
        if result['message']:
            self.active_voice_message = result['message']
            self.voice_message_time = time.time()

    def run(self):
        """Main demo loop"""
        print("=" * 70)
        print("INTEGRATED INTERVIEW CHEATING DETECTION SYSTEM")
        print("=" * 70)
        print()
        print("Monitoring 5 Phases Simultaneously:")
        print("  Phase 0: Eye Contact Detection")
        print("  Phase 1: Face Orientation Detection")
        print("  Phase 2: Posture Detection")
        print("  Phase 3: Object Detection")
        print("  Phase 4: Voice Clarity Detection" + ("" if self.audio_stream else " (disabled - no mic)"))
        print()
        print(f"Single Warning Counter: 0/{self.system.max_warnings}")
        print("Press 'q' to quit and see results")
        print("=" * 70)
        print()

        try:
            if self.audio_stream:
                self.audio_stream.start()

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

                # Auto-stop the session once the rejection banner has been
                # visible long enough for the candidate to see it.
                if result['interview_terminated']:
                    if self.terminated_at is None:
                        self.terminated_at = time.time()
                    elif time.time() - self.terminated_at >= self.termination_hold_sec:
                        break

        finally:
            self.cleanup()

    @staticmethod
    def _wrap_text(text: str, max_width_px: int, font, scale: float, thickness: int) -> list:
        """Greedy word-wrap so caption text fits within the frame width."""
        words = text.split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width_px:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    def _draw_unified_dashboard(self, frame: np.ndarray, result: dict,
                                frame_w: int, frame_h: int) -> np.ndarray:
        """Draw unified dashboard with all 5 phases"""

        # Color scheme based on warnings
        warning_count = result['warning_count']
        max_warnings = self.system.max_warnings
        if warning_count >= max_warnings:
            main_color = (0, 0, 255)  # Red
        elif warning_count >= max_warnings * 3 // 5:
            main_color = (0, 165, 255)  # Orange
        elif warning_count >= 1:
            main_color = (0, 255, 255)  # Yellow
        else:
            main_color = (0, 255, 0)  # Green

        # ===== TOP SECTION: Frame Counter & Warning Counter =====
        cv2.putText(frame, f"Frame: {result['frame_count']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Large warning counter (top right)
        warning_text = f"Warnings: {warning_count}/{max_warnings}"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        warning_x = frame_w - text_size[0] - 20
        cv2.rectangle(frame, (warning_x - 10, 10), (frame_w - 5, 60), main_color, -1)
        cv2.putText(frame, warning_text, (warning_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # ===== LEFT COLUMN: Phase 0-3 =====
        y_offset = 80
        line_height = 35

        def status_tag(is_violation: bool) -> Tuple[str, Tuple[int, int, int]]:
            return ("OK", (0, 255, 0)) if not is_violation else ("X ", (0, 0, 255))

        # Phase 0: Eye Contact
        tag, color = status_tag(result['violations']['eye_contact'])
        gaze = result['details']['eye_contact']['gaze_direction']
        cv2.putText(frame, f"[{tag}] Phase 0: Eye Contact - {gaze}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height

        # Phase 1: Face Orientation
        tag, color = status_tag(result['violations']['face_orientation'])
        alignment = result['details']['face_orientation']['alignment']
        no_face = result['details']['face_orientation']['no_face']
        face_status = "NO FACE" if no_face else alignment
        cv2.putText(frame, f"[{tag}] Phase 1: Face Orientation - {face_status}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height

        # Phase 2: Posture
        tag, color = status_tag(result['violations']['posture'])
        posture = result['details']['posture']['status']
        no_pose = result['details']['posture']['no_pose']
        posture_status = "NO POSE" if no_pose else posture
        cv2.putText(frame, f"[{tag}] Phase 2: Posture - {posture_status}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height

        # Phase 3: Object Detection
        tag, color = status_tag(result['violations']['object_detection'])
        people = result['details']['object_detection']['people_count']
        hand_with_obj = result['details']['object_detection']['hand_carrying_object']
        obj_status = f"People:{people}"
        if hand_with_obj:
            obj_status += " | HAND+OBJECT"
        cv2.putText(frame, f"[{tag}] Phase 3: Object Detection - {obj_status}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height

        # Phase 4: Voice Clarity (informational - not a cheating violation)
        voice = result['details']['voice_clarity']
        voice_clarity = voice['clarity']
        voice_colors = {
            'CLEAR': (0, 255, 0),
            'LOW_VOLUME': (0, 165, 255),
            'SILENT': (150, 150, 150),
        }
        voice_color = voice_colors.get(voice_clarity, (200, 200, 200))
        voice_label = "SPEAKING" if voice['is_speaking'] else "SILENT"
        cv2.putText(frame, f"[..] Phase 4: Voice Clarity - {voice_label} ({voice_clarity})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)
        y_offset += line_height

        # ===== VIOLATION SUMMARY =====
        y_offset += 10
        cv2.line(frame, (10, y_offset), (frame_w - 10, y_offset), (200, 200, 200), 1)
        y_offset += 20

        violations_found = sum(1 for v in result['violations'].values() if v)
        if violations_found == 0:
            cv2.putText(frame, "Status: CLEAN FRAME", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            violation_text = f"Violations This Frame: {violations_found}"
            cv2.putText(frame, violation_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ===== LIVE CAPTIONS =====
        captions = result['details'].get('captions', {})
        caption_text = captions.get('text', '')
        caption_time = captions.get('time', 0.0)
        if caption_text and (time.time() - caption_time < self.caption_display_sec):
            lines = self._wrap_text(caption_text, frame_w - 30, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            line_h = 32
            banner_h = line_h * len(lines) + 16
            banner_y = frame_h - 140 - banner_h
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, banner_y), (frame_w, banner_y + banner_h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (15, banner_y + 26 + i * line_h),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ===== VOICE COACHING BANNER =====
        if (self.active_voice_message and
                time.time() - self.voice_message_time < self.voice_message_display_sec):
            banner_y = frame_h - 100
            cv2.rectangle(frame, (0, banner_y), (frame_w, banner_y + 40), (0, 90, 200), -1)
            cv2.putText(frame, self.active_voice_message, (15, banner_y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        # ===== BOTTOM: Interview Status =====
        status_y = frame_h - 60
        if warning_count >= max_warnings:
            # REJECTION overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, status_y - 80), (frame_w, frame_h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            cv2.putText(frame, "INTERVIEW TERMINATED", (50, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, "Suspicious Activity Detected - Candidate Rejected",
                       (50, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            warnings_remaining = max_warnings - warning_count
            status_text = f"Interview Active | {warnings_remaining} warning(s) remaining"
            cv2.putText(frame, status_text, (20, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ===== BORDER based on warning level =====
        border_thickness = 3
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), main_color, border_thickness)

        return frame

    def cleanup(self):
        """Clean up and print statistics"""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass

        self.cap.release()
        cv2.destroyAllWindows()

        stats = self.system.get_statistics()

        print()
        print("=" * 70)
        print("SESSION COMPLETED")
        print("=" * 70)
        print()
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Final Warning Count: {stats['final_warnings']}/{self.system.max_warnings}")
        print()

        if stats['interview_result'] == 'REJECTED':
            print("INTERVIEW RESULT: REJECTED")
            print("Candidate exhibited suspicious behavior exceeding acceptable threshold")
        else:
            print("INTERVIEW RESULT: APPROVED")
            print("Candidate maintained acceptable behavior throughout interview")

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
