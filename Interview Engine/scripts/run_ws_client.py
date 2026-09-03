"""
NTRVSTA Phase 2 — build-order step 2-3 client (no browser).

Streams audio to the live WebSocket, prints the transcript + the
interviewer's reply + the latency trace, and plays back / saves the
interviewer's spoken audio.

    uvicorn engine.api.main:app                      # start the server first

    python scripts/run_ws_client.py sample.wav       # a WAV file (any rate)
    python scripts/run_ws_client.py sample.wav --play # + hear the reply
    python scripts/run_ws_client.py --mic            # live mic, spoken both ways

Sends 16 kHz mono float32 PCM (512-sample frames); receives 24 kHz mono
float32 PCM back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SEND_SR = 16_000
RECV_SR = 24_000
FRAME = 512
FRAME_SECONDS = FRAME / SEND_SR


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SEND_SR:
        import torch
        import torchaudio.functional as AF

        audio = AF.resample(torch.from_numpy(audio), sr, SEND_SR).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def _frames(audio: np.ndarray):
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        yield audio[i : i + FRAME]


class _Playback:
    """Collects received audio; plays it live with --play, always saves reply.wav."""

    def __init__(self, play: bool):
        self._play = play
        self._buf: list[np.ndarray] = []
        self._stream = None
        if play:
            import sounddevice as sd

            self._stream = sd.OutputStream(samplerate=RECV_SR, channels=1, dtype="float32")
            self._stream.start()

    def feed(self, pcm: bytes) -> None:
        arr = np.frombuffer(pcm, dtype="<f4")
        self._buf.append(arr)
        if self._stream is not None:
            self._stream.write(arr)

    def finish(self, path: Path) -> float:
        if not self._buf:
            return 0.0
        audio = np.concatenate(self._buf)
        self._buf.clear()
        import soundfile as sf

        sf.write(str(path), audio, RECV_SR)
        return len(audio) / RECV_SR

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


async def _receiver(ws, stop: asyncio.Event, one_turn: bool, play: _Playback) -> None:
    audio_frames = 0
    try:
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                play.feed(raw)
                audio_frames += 1
                continue
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "opening":
                print(f"\ninterviewer > {msg['text']}")
            elif kind == "transcript":
                print(f"\ncandidate  > {msg['text'] or '(nothing heard)'}")
            elif kind == "speaking_start":
                audio_frames = 0
            elif kind == "speaking_end":
                secs = play.finish(Path("reply.wav"))
                print(f"    (spoke {secs:.1f}s over {audio_frames} audio frame(s) -> reply.wav)")
            elif kind == "interviewer":
                print(f"interviewer > {msg['text']}")
                t = msg.get("trace", {})
                print(
                    f"    [{msg['kind']}] "
                    + "  ".join(
                        f"{s} {t[s]:.0f}ms"
                        for s in ("stt_final", "ttft", "first_tts_chunk", "playback_start", "response_ready")
                        if s in t
                    )
                )
                if one_turn or msg.get("interview_over"):
                    stop.set()
            elif kind == "end":
                print(f"\n-- {msg.get('reason')} --")
                stop.set()
    except Exception as e:  # noqa: BLE001 - surface why the client stopped listening
        print(f"[receiver stopped: {type(e).__name__}: {e}]")
    finally:
        stop.set()


async def _stream_wav(ws, path: Path, realtime: bool) -> None:
    audio = _load_wav(path)
    print(f"streaming {path.name}: {len(audio) / SEND_SR:.1f}s of audio")
    for frame in _frames(audio):
        await ws.send(frame.tobytes())
        if realtime:
            await asyncio.sleep(FRAME_SECONDS)
    for _ in range(20):  # trailing silence so the endpointer fires
        await ws.send(np.zeros(FRAME, dtype=np.float32).tobytes())
        if realtime:
            await asyncio.sleep(FRAME_SECONDS)


async def _stream_mic(ws, stop: asyncio.Event) -> None:
    import sounddevice as sd

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def cb(indata, frames, time_info, status):
        loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

    print("mic open — speak, pause to end a turn, Ctrl+C to quit")
    with sd.InputStream(
        samplerate=SEND_SR, channels=1, dtype="float32", blocksize=FRAME, callback=cb
    ):
        while not stop.is_set():
            chunk = await queue.get()
            await ws.send(np.ascontiguousarray(chunk, dtype=np.float32).tobytes())


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", type=Path, help="WAV file to stream")
    ap.add_argument("--mic", action="store_true", help="stream the live microphone instead")
    ap.add_argument("--play", action="store_true", help="play the interviewer's audio aloud")
    ap.add_argument("--url", default="ws://localhost:8000/v1/rounds/demo/live")
    ap.add_argument("--fast", action="store_true", help="don't pace WAV playback to realtime")
    args = ap.parse_args()

    if not args.mic and args.wav is None:
        ap.error("give a WAV path or --mic")

    import websockets

    play = _Playback(play=args.play or args.mic)
    async with websockets.connect(args.url, max_size=None, ping_timeout=None) as ws:
        stop = asyncio.Event()
        recv = asyncio.create_task(_receiver(ws, stop, not args.mic, play))
        await ws.send(np.zeros(FRAME, dtype=np.float32).tobytes())  # wake the round
        await asyncio.sleep(0.3)
        try:
            if args.mic:
                sender = asyncio.create_task(_stream_mic(ws, stop))
                await stop.wait()
                sender.cancel()
            else:
                await _stream_wav(ws, args.wav, realtime=not args.fast)
                print("(audio sent — waiting for the interviewer's reply...)")
                await asyncio.wait_for(stop.wait(), timeout=180)
        except (KeyboardInterrupt, asyncio.TimeoutError) as e:
            print(f"[stopped: {type(e).__name__ or 'interrupt'}]")
        finally:
            recv.cancel()
            play.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
