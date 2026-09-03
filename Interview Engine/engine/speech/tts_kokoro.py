"""
Text-to-speech — Kokoro, voice af_heart. Streaming-first: the clause
chunker feeds this fragment-by-fragment so audio starts before the LLM
finishes (plan p5). Output is 24 kHz mono float32.

Kokoro on CPU is compute-heavy and pure-enough-Python that synthesising
on a thread would hold the GIL and starve the async LLM stream + the
socket. So synthesis runs in a **single persistent worker process** with
the model loaded once; the event loop stays free while it works.

`SilenceTTS` is the drop-in stand-in — sized silent buffers, no model. It
runs when `ENGINE_TTS_ENABLED=false` (CI, or a box without Kokoro).

Viseme output (`tts.emit_visemes`) is deferred to build-order pass 5.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from engine.settings import get_models

_SR = 24_000

# --- worker process globals (one KPipeline, loaded once per worker) ---------
_w_pipeline = None
_w_voice = "af_heart"


def _worker_init(lang_code: str, voice: str) -> None:
    global _w_pipeline, _w_voice
    from kokoro import KPipeline

    _w_pipeline = KPipeline(lang_code=lang_code)
    _w_voice = voice


def _worker_synth(text: str) -> np.ndarray:
    parts = [
        np.asarray(item[2] if isinstance(item, tuple) else getattr(item, "audio", item), dtype=np.float32)
        for item in _w_pipeline(text, voice=_w_voice)
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class TTSProvider(ABC):
    sample_rate: int = _SR

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield 24 kHz float32 audio chunks for one clause."""

    def warmup(self) -> None:
        """Front-load lazy model init. Safe to call more than once."""


class SilenceTTS(TTSProvider):
    """Silent audio sized to the clause — keeps the pipeline runnable and
    the latency trace meaningful without loading Kokoro."""

    _SECONDS_PER_WORD = 0.34

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        words = max(1, len(text.split()))
        yield np.zeros(int(self.sample_rate * words * self._SECONDS_PER_WORD), dtype=np.float32)


class KokoroTTS(TTSProvider):
    _STREAM_CHUNK = _SR // 2   # 0.5 s of audio per yielded frame

    def __init__(self) -> None:
        self._cfg = get_models().tts
        self._voice = self._cfg.voice
        self._pool: ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=1,
                initializer=_worker_init,
                initargs=(self._cfg.lang_code, self._voice),
            )
        return self._pool

    def warmup(self) -> None:
        self._ensure_pool().submit(_worker_synth, "Warming up.").result()

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        pool = self._ensure_pool()
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(pool, _worker_synth, text)
        for i in range(0, len(audio), self._STREAM_CHUNK):
            yield audio[i : i + self._STREAM_CHUNK]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
