"""
Endpointing — decides WHEN the candidate has stopped talking. The largest
single lever on perceived responsiveness (plan p7).

Silero VAD v5, CPU. It wants exactly 512-sample float32 frames at 16 kHz
(256 at 8 kHz) — `AudioStream` produces those. The model is stateful
(LSTM); `reset()` between utterances.

Config (`vad` block in models.yaml): `speech_threshold`,
`endpoint_silent_frames` (~250 ms of trailing silence = stopped).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from engine.settings import get_models

_FRAME_SAMPLES = 512   # Silero v5 fixed frame length at 16 kHz


class VADProvider(ABC):
    @abstractmethod
    def is_speech(self, frame: np.ndarray) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...


class SileroVAD(VADProvider):
    def __init__(self) -> None:
        import torch  # local — keeps torch off the import path for text-only use

        self._torch = torch
        cfg = get_models().vad
        self._threshold = cfg.speech_threshold
        self._sample_rate = cfg.sample_rate
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.shape[0] != _FRAME_SAMPLES:
            raise ValueError(
                f"Silero expects {_FRAME_SAMPLES}-sample frames, got {frame.shape[0]}"
            )
        tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        prob = self._model(tensor, self._sample_rate).item()
        return prob >= self._threshold

    def reset(self) -> None:
        self._model.reset_states()


class StreamingEndpointer:
    """Fires once when a candidate utterance ends: speech has started, then
    `silent_frames_for_stop` consecutive silent frames follow.

    Takes the speech test as a callable so tests can drive it without loading
    torch. `min_speech_frames` stops a lone noise blip from creating an empty
    turn.
    """

    def __init__(
        self,
        speech_fn: Callable[[np.ndarray], bool],
        *,
        silent_frames_for_stop: int | None = None,
        min_speech_frames: int = 3,
    ):
        cfg = get_models().vad
        self._speech_fn = speech_fn
        self._silent_for_stop = (
            silent_frames_for_stop
            if silent_frames_for_stop is not None
            else cfg.endpoint_silent_frames
        )
        self._min_speech = min_speech_frames
        self.reset()

    def reset(self) -> None:
        self._speech_frames = 0
        self._trailing_silence = 0
        self._fired = False

    @property
    def speech_started(self) -> bool:
        return self._speech_frames >= self._min_speech

    def feed(self, frame: np.ndarray) -> bool:
        """Returns True exactly once, on the frame that completes the endpoint."""
        if self._fired:
            return False
        if self._speech_fn(frame):
            self._speech_frames += 1
            self._trailing_silence = 0
            return False
        if self.speech_started:
            self._trailing_silence += 1
            if self._trailing_silence >= self._silent_for_stop:
                self._fired = True
                return True
        return False
