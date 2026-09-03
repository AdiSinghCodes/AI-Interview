"""
FastAPI app assembly. Nothing here holds interview logic — routes are thin
and delegate to agents / the orchestrator.

Live now: /health, POST /v1/preflight, WS /v1/rounds/{id}/live, and
GET /testclient (the minimal browser test page). The session / round /
report REST routes are 501 stubs until their build-order pass.

On startup the speech models (Silero, whisper, Kokoro) are warmed in a
background thread so the first live connection doesn't pay a ~60 s cold
load while uvicorn's WS keepalive times out. Skipped under pytest.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from engine import __version__
from engine.api import deps
from engine.api.routes_preflight import router as preflight_router
from engine.api.routes_reports import router as reports_router
from engine.api.routes_rounds import router as rounds_router
from engine.api.routes_sessions import router as sessions_router
from engine.transport.ws_live import router as ws_router

_TESTCLIENT = Path(__file__).resolve().parents[2] / "testclient" / "index.html"


def _warm_models() -> None:
    try:
        deps.get_vad()
        deps.get_stt().warmup()
        deps.get_tts().warmup()
    except Exception as e:  # noqa: BLE001 - a warm failure must not kill the server
        print(f"[warm] speech model warmup failed: {e!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "PYTEST_CURRENT_TEST" not in os.environ:
        threading.Thread(target=_warm_models, name="warm-models", daemon=True).start()
    yield


app = FastAPI(title="NTRVSTA Interview Engine", version=__version__, lifespan=lifespan)
app.include_router(preflight_router)
app.include_router(sessions_router)
app.include_router(rounds_router)
app.include_router(reports_router)
app.include_router(ws_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/testclient", include_in_schema=False)
async def testclient() -> FileResponse:
    return FileResponse(_TESTCLIENT)


if __name__ == "__main__":
    import uvicorn

    # ws="wsproto": uvicorn's auto-detected `websockets` backend breaks across
    # websockets releases (17.x vs uvicorn 0.52); wsproto is stable.
    # ws_ping_interval=0: a long CPU-bound TTS synth can freeze the event loop
    # past the default keepalive timeout and drop the connection.
    uvicorn.run(
        "engine.api.main:app",
        host="0.0.0.0",
        port=8000,
        ws="wsproto",
        ws_ping_interval=0,
    )
