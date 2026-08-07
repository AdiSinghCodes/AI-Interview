from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import settings
from routers import interview, voice, feedback
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.voice_analyzer import VoiceAnalyzer
from services.interview_manager import InterviewManager
from services.feedback_engine import FeedbackEngine
from services.stt_service import STTService
from proctoring.engine import ProctoringEngine
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm_service = LLMService()
    app.state.tts_service = TTSService()
    app.state.voice_analyzer = VoiceAnalyzer()
    app.state.stt_service = STTService()
    app.state.interview_manager = InterviewManager(app.state.llm_service)
    app.state.feedback_engine = FeedbackEngine()
    print("Services initialized")
    yield
    print("Services cleaned up")

app = FastAPI(title="AI Interview Pro API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interview.router)
app.include_router(voice.router)
app.include_router(feedback.router)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "AI Interview Pro API is running", "docs": "/docs"}

@app.websocket("/ws/interview/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    proctor = ProctoringEngine()
    
    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                data = json.loads(text_data)
                msg_type = data.get("type")
                
                if msg_type == "video_frame":
                    result = proctor.process_frame(data.get("frame", ""))
                    await websocket.send_json({
                        "type": "proctoring_update",
                        "data": result
                    })
                
                elif msg_type == "voice_level":
                    level = data.get("level", 0)
                    feedback_msgs = []
                    if level < 15:
                        feedback_msgs.append("Your voice is too quiet, please speak louder")
                    elif level > 85:
                        feedback_msgs.append("You're speaking too loudly")
                    
                    if feedback_msgs:
                        await websocket.send_json({
                            "type": "voice_feedback",
                            "messages": feedback_msgs
                        })
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception as e:
                print(f"Error processing WS text: {e}")
                
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
