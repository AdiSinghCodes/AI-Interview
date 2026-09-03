"""
Speech-to-text — faster-whisper, base.en, CPU, int8.

CPU is deliberate (plan p4): the GPU path needs cublas64_12.dll (absent on
this machine) and CPU keeps the full 6 GB for the interviewer LLM. Swap =
a new stt_parakeet.py implementing STTProvider + one line in models.yaml.

Incremental mode: `feed_partial()` is run every ~1.5 s while the candidate
is still talking, so at the endpoint `finalize()` mostly just returns the
last partial instead of transcribing the whole utterance from scratch —
`vad_filter=False` (Silero gates upstream), `beam_size=1` (greedy) keep it
fast either way.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from engine.settings import get_models


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str: ...

    def warmup(self) -> None:
        """Front-load lazy model init. Safe to call more than once."""

    # Incremental API — default to one-shot so fakes don't have to implement it.
    def begin_utterance(self) -> None: ...

    def feed_partial(self, audio: np.ndarray) -> str:
        return self.transcribe(audio)

    def finalize(self, audio: np.ndarray) -> str:
        return self.transcribe(audio)


class WhisperSTT(STTProvider):
    _PARTIAL_FRESH_S = 2.5        # a partial newer than this can stand in for the final
    _PARTIAL_COVERAGE = 0.6       # ...if it also covered at least this much of the audio
                                 # (the uncovered tail is mostly the ~250 ms of trailing silence)

    def __init__(self) -> None:
        self._cfg = get_models().stt
        self._model = None
        self._reset_partial()

    # -- lifecycle -----------------------------------------------------
    def warmup(self) -> None:
        self._ensure_model()

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._cfg.model, device=self._cfg.device, compute_type=self._cfg.compute_type
            )
            actual = str(getattr(self._model.model, "device", self._cfg.device))
            if actual != self._cfg.device:
                raise RuntimeError(
                    f"STT requested device={self._cfg.device!r} but the model "
                    f"reports {actual!r} — refusing a silent fallback"
                )
        return self._model

    # -- core ----------------------------------------------------------
    def _run(self, audio: np.ndarray) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size == 0 or float(np.abs(audio).max()) < 1e-4:
            return ""
        segments, _ = self._ensure_model().transcribe(
            audio, language="en", vad_filter=False, beam_size=1
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        return self._run(audio)

    # -- incremental -------------------------------------------------
    def _reset_partial(self) -> None:
        self._partial_text = ""
        self._partial_at = 0.0
        self._partial_samples = 0

    def begin_utterance(self) -> None:
        self._reset_partial()

    def feed_partial(self, audio: np.ndarray) -> str:
        text = self._run(audio)
        if text:
            self._partial_text = text
            self._partial_at = time.perf_counter()
            self._partial_samples = int(audio.size)
        return self._partial_text

    def finalize(self, audio: np.ndarray) -> str:
        if audio.size < 16_000 * 0.35:   # < ~350 ms — a VAD blip, not an answer
            self._reset_partial()
            return ""
        fresh = (
            self._partial_text
            and (time.perf_counter() - self._partial_at) < self._PARTIAL_FRESH_S
            and self._partial_samples >= audio.size * self._PARTIAL_COVERAGE
        )
        result = self._partial_text if fresh else self._run(audio)
        self._reset_partial()
        return result
