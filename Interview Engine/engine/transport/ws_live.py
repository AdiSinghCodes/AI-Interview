"""
The live interview WebSocket. One connection per round. Thin adapter:
decode audio frames off the socket, endpoint + transcribe them, feed the
transcript to the orchestrator, stream its spoken reply back. All turn
logic stays in `engine.orchestrator.session_orchestrator` — this file
never grows a state machine of its own.

Wire protocol
  up (client -> server):
    - binary  : 16 kHz mono float32 LE PCM, any length
    - text    : {"type": "end_round"}
  down (server -> client):
    - binary  : 24 kHz mono float32 LE PCM — the interviewer's audio
    - {"type": "opening",       "text": ...}
    - {"type": "partial",       "text": ...}          (interim transcript)
    - {"type": "transcript",    "text": ...}          (final, per utterance)
    - {"type": "speaking_start","sr": 24000}
    - {"type": "speaking_end"}
    - {"type": "interviewer",   "text","kind","state","trace","cache_hit","interview_over"}
    - {"type": "end", "reason": ...}

Proctor frames never share this socket (plan §16).
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from engine.api import deps, session_store
from engine.orchestrator import SessionOrchestrator
from engine.orchestrator.speech_gate import SpeechGate
from engine.orchestrator.turn_state import TurnState
from engine.plans import get_plan
from engine.schemas.latency_trace import Stage, TurnTrace
from engine.speech.audio_stream import AudioStream
from engine.speech.vad_silero import StreamingEndpointer

router = APIRouter()

_ECHO_TAIL_MS = 150.0
_SEND_SR = 16_000
_PARTIAL_EVERY_SAMPLES = int(_SEND_SR * 1.6)   # run an interim transcription ~every 1.6 s
_AUDIO_FRAME = 4096                             # bytes per outbound audio frame group


@router.websocket("/v1/rounds/{round_id}/live")
async def live_round(websocket: WebSocket, round_id: str) -> None:
    await websocket.accept()
    provider = deps.get_inference_provider()
    vad = await asyncio.to_thread(deps.get_vad)
    stt = await asyncio.to_thread(deps.get_stt)
    tts = await asyncio.to_thread(deps.get_tts)
    await asyncio.to_thread(stt.warmup)

    rnd = session_store.get_round(round_id)
    plan = rnd.plan if rnd else get_plan("technical")
    prefetch = rnd.prefetch if rnd else None
    if prefetch is not None and not prefetch.ready:
        # a background task (started by POST /v1/sessions) is already warming it
        await prefetch.wait_ready()

    orch = SessionOrchestrator(round_id, plan, provider, prefetch=prefetch)
    stream = AudioStream()
    endpointer = StreamingEndpointer(vad.is_speech)
    gate = SpeechGate(tail_ms=_ECHO_TAIL_MS)
    partial_task: asyncio.Task | None = None
    last_partial_at = 0

    async def send_audio(pcm: bytes) -> None:
        for i in range(0, len(pcm), _AUDIO_FRAME):
            await websocket.send_bytes(pcm[i : i + _AUDIO_FRAME])

    def arm_listening() -> None:
        nonlocal last_partial_at
        stream.reset()
        endpointer.reset()
        stt.begin_utterance()
        last_partial_at = 0

    # opening line
    gate.enter_speaking()
    opening = await orch.start()
    await websocket.send_json({"type": "opening", "text": opening})
    await websocket.send_json({"type": "speaking_start", "sr": tts.sample_rate})
    if orch.opening_audio:
        await send_audio(orch.opening_audio)
    else:
        async for chunk in tts.synthesize(opening):
            await websocket.send_bytes(chunk.tobytes())
    await websocket.send_json({"type": "speaking_end"})
    gate.leave_speaking()
    arm_listening()

    try:
        while True:
            message = await websocket.receive()
            await asyncio.sleep(0)
            if message["type"] == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                if json.loads(text).get("type") == "end_round":
                    break
                continue

            data = message.get("bytes")
            if data is None:
                continue

            if orch.state != TurnState.LISTENING or not gate.accepting_audio():
                stream.reset()
                endpointer.reset()
                continue

            hit = any(endpointer.feed(frame) for frame in stream.push(data))

            if partial_task is not None and partial_task.done():
                try:
                    ptext = partial_task.result()
                    if ptext:
                        await websocket.send_json({"type": "partial", "text": ptext})
                except Exception:  # noqa: BLE001 - a failed partial is not fatal
                    pass
                partial_task = None

            if hit:
                if partial_task is not None:
                    partial_task.cancel()
                    partial_task = None
                if not await _resolve_turn(websocket, orch, stream, endpointer, gate, vad, stt, send_audio):
                    break
                arm_listening()
            elif (
                partial_task is None
                and endpointer.speech_started
                and stream.utterance_samples() - last_partial_at >= _PARTIAL_EVERY_SAMPLES
            ):
                last_partial_at = stream.utterance_samples()
                partial_task = asyncio.create_task(
                    asyncio.to_thread(stt.feed_partial, stream.peek_utterance())
                )
    except (WebSocketDisconnect, RuntimeError):
        pass
    except (KeyError, json.JSONDecodeError):
        await _safe_close(websocket, code=1003, reason="bad frame")
        return
    finally:
        if partial_task is not None:
            partial_task.cancel()
        if orch.state != TurnState.ENDED:
            orch.end()
        if rnd is not None:
            rnd.transcript = [
                {"index": t.index, "role": t.role.value, "kind": t.kind.value, "text": t.text}
                for t in orch.transcript.turns
            ]
        await _safe_close(websocket)


async def _resolve_turn(websocket, orch, stream, endpointer, gate, vad, stt, send_audio) -> bool:
    tts = deps.get_tts()
    trace = TurnTrace(turn_index=orch.transcript.next_index + 1)
    trace.mark(Stage.ANSWER_RECEIVED)

    audio = stream.take_utterance()
    vad.reset()

    text = (await asyncio.to_thread(stt.finalize, audio)).strip()
    trace.mark(Stage.STT_FINAL)
    await websocket.send_json({"type": "transcript", "text": text})

    if not text:
        return True  # silence misfire — stay in LISTENING

    gate.enter_speaking()
    spoke = False

    async def on_clause(clause: str) -> None:
        nonlocal spoke
        if not spoke:
            spoke = True
            await websocket.send_json({"type": "speaking_start", "sr": tts.sample_rate})
        async for chunk in tts.synthesize(clause):
            if Stage.FIRST_TTS_CHUNK.value not in trace.marks:
                trace.mark(Stage.FIRST_TTS_CHUNK)
            await websocket.send_bytes(chunk.tobytes())
            if Stage.PLAYBACK_START.value not in trace.marks:
                trace.mark(Stage.PLAYBACK_START)

    result = await orch.submit_answer(text, trace=trace, on_clause=on_clause)

    if result.prefetched_audio:
        await websocket.send_json({"type": "speaking_start", "sr": tts.sample_rate})
        await send_audio(result.prefetched_audio)
        spoke = True

    if spoke:
        await websocket.send_json({"type": "speaking_end"})
    await websocket.send_json(
        {
            "type": "interviewer",
            "text": result.interviewer_text,
            "kind": result.kind.value,
            "state": result.state.value,
            "cache_hit": result.cache_hit,
            "trace": {k: round(v, 1) for k, v in result.trace_marks.items()},
            "interview_over": result.interview_over,
        }
    )
    gate.leave_speaking()

    if result.interview_over:
        await websocket.send_json({"type": "end", "reason": "interview_complete"})
        return False
    return True


async def _safe_close(websocket: WebSocket, *, code: int = 1000, reason: str = "") -> None:
    if websocket.application_state == WebSocketState.DISCONNECTED:
        return
    try:
        await websocket.close(code=code, reason=reason)
    except (RuntimeError, WebSocketDisconnect):
        pass
