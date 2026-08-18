import cv2
import numpy as np
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path to import detectors
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / 'eye_contact_detection'))
sys.path.insert(0, str(parent_dir / 'face_orientation_detection'))
sys.path.insert(0, str(parent_dir / 'posture_detection'))
sys.path.insert(0, str(parent_dir / 'object_detection'))
sys.path.insert(0, str(parent_dir / 'voice_clarity_detection'))

from eye_contact_model import EyeContactDetector
from face_orientation_model import FaceOrientationDetector
from posture_model import PostureDetector
from object_model import ObjectDetector
from voice_clarity_model import VoiceClarityDetector
from caption_model import CaptionGenerator


class IntegratedInterviewSystem:
    """
    Unified interview cheating detection system combining all 4 phases:
    - Phase 0: Eye Contact Detection
    - Phase 1: Face Orientation Detection
    - Phase 2: Posture Detection
    - Phase 3: Object Detection
    
    Single shared warning system: 0-25 warnings
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

        print("Loading Phase 4: Voice Clarity Detection...")
        self.voice_detector = VoiceClarityDetector()
        self.voice_lock = threading.Lock()
        self.voice_state = {
            'voice_prob': 0.0,
            'is_speaking': False,
            'rms_dbfs': -100.0,
            'clarity': 'SILENT',
            'message': None,
        }

        print("Loading Live Captioning (faster-whisper)...")
        try:
            self.caption_generator = CaptionGenerator(sample_rate=self.voice_detector.SAMPLE_RATE)
        except Exception as e:
            print(f"Warning: could not load captioning model ({e}). Captions disabled.")
            self.caption_generator = None

        # Speech chunks are buffered into one utterance and only handed to
        # Whisper once the candidate pauses (or talks for too long without
        # pausing), since transcribing 32ms slivers one at a time would be
        # both slow and inaccurate.
        self._utterance_chunks = []
        self._utterance_silence_chunks = 0
        self._UTTERANCE_SILENCE_HOLD_CHUNKS = 20   # ~0.64s of silence ends an utterance
        self._UTTERANCE_MAX_CHUNKS = 470           # ~15s cap on one utterance

        # Shared warning system
        self.warning_count = 0
        self.max_warnings = 25

        # Accumulated violation duration (seconds) per type, so a single
        # noisy frame (blink, brief head turn, misdetection) can't burn a
        # warning by itself. Tracked in wall-clock time rather than frame
        # counts because actual processing FPS varies with hardware/load,
        # and frame-count thresholds would fire at inconsistent real times.
        self.violation_streaks = {
            'eye_contact': 0.0,
            'face_orientation': 0.0,
            'posture': 0.0,
            'object_detection': 0.0
        }
        self.violation_thresholds = {
            'eye_contact': 2.0,        # seconds
            'face_orientation': 2.0,   # seconds
            'posture': 2.0,            # seconds
            'object_detection': 3.0    # seconds
        }
        self._last_warning_update_time = None

        # Statistics tracking
        self.violation_stats = {
            'eye_contact': 0,
            'face_orientation': 0,
            'posture': 0,
            'object_detection': 0,
            'clean_frames': 0
        }

        self.frame_count = 0
        print("All detectors loaded successfully!")
        print()

    def process_audio_chunk(self, chunk: np.ndarray) -> Dict:
        """
        Feed one microphone chunk (mono float32, [-1, 1]) to the voice clarity
        detector. Meant to be called from an audio callback thread, separate
        from the per-frame video loop. Thread-safe against get_voice_state().
        """
        result = self.voice_detector.process_chunk(chunk)
        with self.voice_lock:
            self.voice_state = result

        if self.caption_generator is not None:
            if result['is_speaking']:
                self._utterance_chunks.append(chunk.copy())
                self._utterance_silence_chunks = 0
            elif self._utterance_chunks:
                # Keep buffering through brief pauses instead of dropping
                # them: splicing out every non-speech chunk leaves Whisper
                # with discontinuous, chopped audio that it can't reliably
                # transcribe (verified - it silently returns empty text).
                self._utterance_chunks.append(chunk.copy())
                self._utterance_silence_chunks += 1
                if self._utterance_silence_chunks >= self._UTTERANCE_SILENCE_HOLD_CHUNKS:
                    self._flush_utterance()

            if len(self._utterance_chunks) >= self._UTTERANCE_MAX_CHUNKS:
                self._flush_utterance()

        return result

    def _flush_utterance(self) -> None:
        """Hand the buffered speech segment to the background transcriber."""
        audio = np.concatenate(self._utterance_chunks)
        duration_sec = len(audio) / self.voice_detector.SAMPLE_RATE
        print(f"[Captions] Transcribing {duration_sec:.1f}s of speech...")
        self.caption_generator.submit_utterance(audio)
        self._utterance_chunks = []
        self._utterance_silence_chunks = 0

    def get_voice_state(self) -> Dict:
        """Latest voice clarity snapshot, safe to call from the video thread."""
        with self.voice_lock:
            return dict(self.voice_state)

    def get_caption(self) -> Dict:
        """Latest live-caption snapshot, safe to call from the video thread."""
        if self.caption_generator is None:
            return {'text': '', 'time': 0.0, 'available': False}
        caption = self.caption_generator.get_latest_caption()
        caption['available'] = True
        return caption

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
        gaze = eye_result['left_gaze'] or eye_result['right_gaze']
        gaze_direction = gaze['direction'] if gaze else 'CENTER'
        horizontal_ratio = gaze['horizontal_ratio'] if gaze else 0.5
        vertical_ratio = gaze['vertical_ratio'] if gaze else 0.5
        if gaze_direction != 'CENTER':
            output['violations']['eye_contact'] = True
            self.violation_stats['eye_contact'] += 1
        output['details']['eye_contact'] = {
            'gaze_direction': gaze_direction,
            'horizontal_ratio': horizontal_ratio,
            'vertical_ratio': vertical_ratio
        }

        # PHASE 1: Face Orientation Detection
        face_result = self.face_orientation_detector.process_frame(frame)
        if face_result['alignment'] != 'ALIGNED':
            output['violations']['face_orientation'] = True
            self.violation_stats['face_orientation'] += 1
        elif face_result['no_face_detected']:
            output['violations']['face_orientation'] = True
            self.violation_stats['face_orientation'] += 1
        output['details']['face_orientation'] = {
            'alignment': face_result['alignment'],
            'yaw': face_result['yaw'],
            'pitch': face_result['pitch'],
            'no_face': face_result['no_face_detected']
        }
        
        # PHASE 2: Posture Detection
        posture_result = self.posture_detector.process_frame(frame)
        if posture_result['posture_status'] not in ['GOOD', 'SLIGHTLY_SLOUCHED']:
            output['violations']['posture'] = True
            self.violation_stats['posture'] += 1
        elif posture_result['no_face_detected']:
            output['violations']['posture'] = True
            self.violation_stats['posture'] += 1
        output['details']['posture'] = {
            'status': posture_result['posture_status'],
            'no_pose': posture_result['no_face_detected']
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

        # PHASE 4: Voice Clarity (audio processed on its own thread; this just
        # reads the latest snapshot - not a cheating violation, coaching only)
        output['details']['voice_clarity'] = self.get_voice_state()
        output['details']['captions'] = self.get_caption()

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
        Update unified warning count based on sustained violations.

        Logic: each violation type accumulates wall-clock seconds while
        violating and decays by the same elapsed time (instead of resetting
        outright) on a clean frame, so one noisy/missed frame doesn't erase
        real sustained violation - and the timing holds regardless of actual
        processing FPS. Once a type's accumulated duration crosses its
        threshold it contributes ONE warning to the shared counter and that
        accumulator resets.
        """
        now = time.time()
        if self._last_warning_update_time is None:
            dt = 0.0
        else:
            # Clamp so a long pause (e.g. window minimized) can't be
            # counted as one giant burst of violation or decay.
            dt = min(now - self._last_warning_update_time, 0.5)
        self._last_warning_update_time = now

        for v_type, is_violating in violations.items():
            if is_violating:
                self.violation_streaks[v_type] += dt
            else:
                self.violation_streaks[v_type] = max(0.0, self.violation_streaks[v_type] - dt)

            if self.violation_streaks[v_type] >= self.violation_thresholds[v_type]:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.violation_streaks[v_type] = 0.0
    
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
        self.violation_streaks = {k: 0.0 for k in self.violation_streaks}
        self._last_warning_update_time = None
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
        with self.voice_lock:
            self.voice_detector.reset()
            self.voice_state = {
                'voice_prob': 0.0,
                'is_speaking': False,
                'rms_dbfs': -100.0,
                'clarity': 'SILENT',
                'message': None,
            }
        self._utterance_chunks = []
        self._utterance_silence_chunks = 0
        if self.caption_generator is not None:
            self.caption_generator.reset()
