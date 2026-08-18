"""
Voice Clarity Detection Model
Combines Silero VAD (speech vs silence) with RMS loudness analysis to catch
speech that is too quiet to be understood, and produces a coaching message
asking the candidate to speak up (or check their mic if nothing is heard).
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Optional

try:
    from silero_vad import load_silero_vad
    import torch
    _SILERO_AVAILABLE = True
except ImportError:
    _SILERO_AVAILABLE = False


class VoiceClarityDetector:
    """
    Real-time voice clarity detector.

    Feed it mono float32 audio chunks (values in [-1, 1]) captured at
    SAMPLE_RATE and it reports whether the candidate is speaking, how loud
    that speech is, and whether they should be coached to speak louder or
    check their microphone.
    """

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512            # ~32ms at 16kHz, required by Silero VAD

    VAD_THRESHOLD = 0.5
    QUIET_DBFS = -32.0          # speech quieter than this is hard to hear
    LOW_VOLUME_HOLD_SEC = 1.5   # must stay quiet this long before nagging
    MESSAGE_COOLDOWN_SEC = 5.0  # don't repeat a coaching message too often
    LONG_SILENCE_SEC = 8.0      # no speech at all for this long -> check mic

    LOW_VOLUME_MESSAGE = "Your voice is not clear. Please speak a little louder."
    NO_AUDIO_MESSAGE = "We can't hear you. Please check your microphone."

    def __init__(self):
        self.model = None
        self.vad_available = _SILERO_AVAILABLE
        if self.vad_available:
            try:
                self.model = load_silero_vad()
            except Exception:
                self.vad_available = False

        self.voice_prob = 0.0
        self.is_speaking = False
        self.prob_history = deque(maxlen=200)
        self.rms_history = deque(maxlen=50)

        self._quiet_speech_start: Optional[float] = None
        self._last_message_time = 0.0
        self._last_speech_time = time.time()

    def reset(self):
        """Reset all tracking state (keeps the loaded VAD model)."""
        model, vad_available = self.model, self.vad_available
        self.__init__()
        self.model, self.vad_available = model, vad_available

    def process_chunk(self, chunk: np.ndarray, now: Optional[float] = None) -> Dict:
        """
        Process one audio chunk.

        Args:
            chunk: mono float32 numpy array, values in [-1, 1], length CHUNK_SIZE
            now: optional timestamp override, mainly for testing

        Returns:
            dict with:
              voice_prob  - 0..1 probability the chunk contains speech
              is_speaking - bool
              rms_dbfs    - loudness of the chunk in dBFS
              clarity     - 'CLEAR' | 'LOW_VOLUME' | 'SILENT'
              message     - coaching string, or None if nothing to say right now
        """
        now = now if now is not None else time.time()

        prob = self._voice_probability(chunk)
        rms_dbfs = self._rms_dbfs(chunk)

        self.voice_prob = prob
        self.prob_history.append(prob)
        self.is_speaking = prob >= self.VAD_THRESHOLD

        message = None
        clarity = 'SILENT'

        if self.is_speaking:
            self._last_speech_time = now
            self.rms_history.append(rms_dbfs)

            if rms_dbfs < self.QUIET_DBFS:
                clarity = 'LOW_VOLUME'
                if self._quiet_speech_start is None:
                    self._quiet_speech_start = now
                quiet_dur = now - self._quiet_speech_start
                if (quiet_dur >= self.LOW_VOLUME_HOLD_SEC and
                        now - self._last_message_time >= self.MESSAGE_COOLDOWN_SEC):
                    message = self.LOW_VOLUME_MESSAGE
                    self._last_message_time = now
            else:
                clarity = 'CLEAR'
                self._quiet_speech_start = None
        else:
            self._quiet_speech_start = None
            silence_dur = now - self._last_speech_time
            if (silence_dur >= self.LONG_SILENCE_SEC and
                    now - self._last_message_time >= self.MESSAGE_COOLDOWN_SEC):
                message = self.NO_AUDIO_MESSAGE
                self._last_message_time = now

        return {
            'voice_prob': prob,
            'is_speaking': self.is_speaking,
            'rms_dbfs': rms_dbfs,
            'clarity': clarity,
            'message': message,
        }

    def _voice_probability(self, chunk: np.ndarray) -> float:
        """Speech probability via Silero VAD, falling back to RMS volume."""
        if self.vad_available and self.model is not None:
            try:
                tensor = torch.FloatTensor(chunk).unsqueeze(0)
                return float(self.model(tensor, self.SAMPLE_RATE).item())
            except Exception:
                pass
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        return float(min(1.0, rms * 32.0))

    @staticmethod
    def _rms_dbfs(chunk: np.ndarray) -> float:
        """RMS loudness of the chunk in dBFS (0 dBFS = full scale)."""
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) + 1e-9)
        return float(20 * np.log10(rms))
