from fastapi import APIRouter, Request, HTTPException
from models.schemas import InterviewStartRequest, AnswerSubmitRequest

router = APIRouter(prefix="/api/interview", tags=["interview"])

@router.post("/start")
async def start_interview(request: Request, body: InterviewStartRequest):
    manager = request.app.state.interview_manager
    try:
        res = await manager.start_session(body.role, body.difficulty, body.num_questions)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/answer")
async def submit_answer(request: Request, session_id: str, body: AnswerSubmitRequest):
    manager = request.app.state.interview_manager
    try:
        res = await manager.submit_answer(session_id, body.answer_text, body.voice_metrics)
        return res
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/question")
async def get_question(request: Request, session_id: str):
    manager = request.app.state.interview_manager
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status == "completed" or session.current_question_index >= len(session.questions):
        return {"status": "completed", "question": None}
    return {"status": "active", "question": session.questions[session.current_question_index]}

@router.post("/{session_id}/end")
async def end_interview(request: Request, session_id: str):
    manager = request.app.state.interview_manager
    res = manager.end_session(session_id)
    if res["status"] == "error":
        raise HTTPException(status_code=404, detail=res["message"])
    return res
