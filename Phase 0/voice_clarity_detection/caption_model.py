"""
Live Speech Captioning
Transcribes finished speech utterances (bounded by voice-activity silence)
in a background thread using faster-whisper, so inference latency never
blocks audio capture or the video processing loop.
"""

import time
import threading
import queue
from typing import Dict

import numpy as np

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False


class CaptionGenerator:
    """Background speech-to-text captioner for live subtitles."""

    def __init__(self, sample_rate: int, model_size: str = "base.en",
                 device: str = "cpu", compute_type: str = "int8"):
        if not _WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper is not installed")

        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._lock = threading.Lock()
        self._latest_text = ""
        self._latest_time = 0.0

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit_utterance(self, audio: np.ndarray) -> None:
        """Queue a finished speech segment (mono float32 @ sample_rate) for transcription."""
        if audio.size > 0:
            self._queue.put(audio)

    def get_latest_caption(self) -> Dict:
        """Latest transcribed text and the time it was produced (0.0 if none yet)."""
        with self._lock:
            return {'text': self._latest_text, 'time': self._latest_time}

    def reset(self) -> None:
        with self._lock:
            self._latest_text = ""
            self._latest_time = 0.0
        with self._queue.mutex:
            self._queue.queue.clear()

    def _worker_loop(self) -> None:
        while True:
            audio = self._queue.get()
            try:
                # vad_filter re-checks for actual speech inside the segment
                # and drops silence-only audio, which otherwise makes
                # Whisper hallucinate filler words like "you" or "thanks".
                segments, _ = self.model.transcribe(
                    audio, language='en', vad_filter=True, beam_size=1
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
            except Exception as e:
                print(f"[Captions] transcription failed: {e}")
                text = ""
            if text:
                print(f"[Captions] \"{text}\"")
                with self._lock:
                    self._latest_text = text
                    self._latest_time = time.time()
            else:
                print("[Captions] (no speech recognized in that segment)")
