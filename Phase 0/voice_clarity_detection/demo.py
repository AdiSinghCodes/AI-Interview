"""
Real-time Voice Clarity Detection Demo
Listens to the microphone and shows a live coaching banner ("speak louder" /
"can't hear you") whenever the candidate's speech is too quiet or missing.
"""

import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv not installed. Run: pip install opencv-python")
    sys.exit(1)

from voice_clarity_model import VoiceClarityDetector


class VoiceClarityDemo:
    """Demo for real-time voice clarity coaching"""

    def __init__(self):
        self.detector = VoiceClarityDetector()
        self.latest_result = {
            'voice_prob': 0.0,
            'is_speaking': False,
            'rms_dbfs': -100.0,
            'clarity': 'SILENT',
            'message': None,
        }
        self.active_message = None
        self.message_shown_at = 0.0
        self.message_display_sec = 4.0
        self.message_log = []

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0]
        result = self.detector.process_chunk(chunk)
        self.latest_result = result
        if result['message']:
            self.active_message = result['message']
            self.message_shown_at = time.time()
            self.message_log.append(result['message'])

    def run(self):
        print("Voice Clarity Detection Demo Started")
        print(f"Mode: {'Silero VAD' if self.detector.vad_available else 'Volume fallback'}")
        print("Speak normally, then try speaking very quietly to trigger coaching.")
        print("Press 'q' to quit")
        print("=" * 60)

        stream = sd.InputStream(
            samplerate=VoiceClarityDetector.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=VoiceClarityDetector.CHUNK_SIZE,
            callback=self._audio_callback,
        )

        W, H = 700, 320
        with stream:
            while True:
                canvas = np.full((H, W, 3), (25, 25, 30), dtype=np.uint8)
                frame = self._draw(canvas)

                cv2.imshow("Voice Clarity Detection", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        self.cleanup()

    def _draw(self, canvas: np.ndarray) -> np.ndarray:
        result = self.latest_result
        H, W = canvas.shape[:2]

        cv2.putText(canvas, "Voice Clarity Detection", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (230, 230, 255), 1)

        status_text = "SPEAKING" if result['is_speaking'] else "SILENT"
        status_color = (60, 200, 90) if result['is_speaking'] else (120, 120, 120)
        cv2.putText(canvas, status_text, (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 2)

        cv2.putText(canvas, f"Voice probability: {result['voice_prob']:.2f}",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Loudness: {result['rms_dbfs']:.1f} dBFS",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        clarity = result['clarity']
        clarity_color = {
            'CLEAR': (60, 200, 90),
            'LOW_VOLUME': (0, 165, 255),
            'SILENT': (120, 120, 120),
        }.get(clarity, (200, 200, 200))
        cv2.putText(canvas, f"Clarity: {clarity}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, clarity_color, 1)

        # Coaching banner shown for a few seconds after being triggered
        if self.active_message and time.time() - self.message_shown_at < self.message_display_sec:
            cv2.rectangle(canvas, (0, H - 70), (W, H), (0, 90, 200), -1)
            cv2.putText(canvas, self.active_message, (15, H - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        return canvas

    def cleanup(self):
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print(f"Coaching messages triggered: {len(self.message_log)}")
        for msg in self.message_log[-10:]:
            print(f"  - {msg}")
        print("\nDone.")


if __name__ == "__main__":
    demo = VoiceClarityDemo()
    demo.run()
