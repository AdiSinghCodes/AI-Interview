"""
Speech I/O. Each model sits behind an ABC with the model name in the
filename (`stt_whisper.py`), so a swap is a new file + one line in
config/models.yaml — the interface and every caller stay put.
"""

from .audio_stream import FRAME_SAMPLES, AudioStream
from .clause_chunker import ClauseChunker
from .tts_kokoro import KokoroTTS, SilenceTTS, TTSProvider
from .vad_silero import SileroVAD, StreamingEndpointer, VADProvider

__all__ = [
    "AudioStream",
    "FRAME_SAMPLES",
    "ClauseChunker",
    "KokoroTTS",
    "SilenceTTS",
    "TTSProvider",
    "SileroVAD",
    "StreamingEndpointer",
    "VADProvider",
]
