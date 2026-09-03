"""
AudioStream — turns the socket's arbitrary PCM byte chunks into the fixed
512-sample frames Silero needs, and accumulates the whole utterance for
STT.

Wire format (both clients produce it): **16 kHz, mono, float32
little-endian PCM**, sent as WebSocket binary frames of any length.
"""

from __future__ import annotations

import numpy as np

FRAME_SAMPLES = 512
_DTYPE = np.dtype("<f4")  # float32 little-endian


class AudioStream:
    def __init__(self) -> None:
        self._byte_carry = b""                         # < 4 bytes, spans a chunk
        self._sample_carry = np.empty(0, dtype=np.float32)  # < 512 samples
        self._utterance: list[np.ndarray] = []

    def push(self, pcm_bytes: bytes) -> list[np.ndarray]:
        """Append a raw chunk; return any newly-complete 512-sample frames."""
        data = self._byte_carry + pcm_bytes
        usable = len(data) - (len(data) % 4)
        self._byte_carry = data[usable:]
        samples = np.frombuffer(data[:usable], dtype=_DTYPE)

        buf = np.concatenate([self._sample_carry, samples])
        n_frames = buf.shape[0] // FRAME_SAMPLES
        frames = [
            buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].copy()
            for i in range(n_frames)
        ]
        self._sample_carry = buf[n_frames * FRAME_SAMPLES :].copy()
        self._utterance.extend(frames)
        return frames

    def take_utterance(self) -> np.ndarray:
        """All audio buffered since the last take (incl. trailing silence).
        Clears the utterance buffer; keeps no partial-frame tail."""
        parts = list(self._utterance)
        if self._sample_carry.size:
            parts.append(self._sample_carry.copy())
        self._utterance.clear()
        self._sample_carry = np.empty(0, dtype=np.float32)
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def peek_utterance(self) -> np.ndarray:
        """The utterance-so-far without clearing — for incremental STT."""
        parts = list(self._utterance)
        if self._sample_carry.size:
            parts.append(self._sample_carry)
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def utterance_samples(self) -> int:
        return sum(f.shape[0] for f in self._utterance) + int(self._sample_carry.shape[0])

    def reset(self) -> None:
        self._byte_carry = b""
        self._sample_carry = np.empty(0, dtype=np.float32)
        self._utterance.clear()
