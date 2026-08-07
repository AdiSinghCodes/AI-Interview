from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io

router = APIRouter(prefix="/api", tags=["voice"])

@router.post("/voice/analyze")
async def analyze_voice(request: Request, file: UploadFile = File(...), transcript: str = Form(None)):
    analyzer = request.app.state.voice_analyzer
    try:
        content = await file.read()
        res = analyzer.analyze_audio(content, transcript)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice/transcribe")
async def transcribe_voice(request: Request, file: UploadFile = File(...)):
    stt = request.app.state.stt_service
    try:
        content = await file.read()
        res = stt.transcribe(content)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tts/speak")
async def speak_text(request: Request, text: str, voice: str = "en-US-GuyNeural"):
    tts = request.app.state.tts_service
    try:
        audio_bytes = await tts.synthesize(text, voice)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
