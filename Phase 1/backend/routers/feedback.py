from fastapi import APIRouter, Request, HTTPException

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

@router.get("/{session_id}")
async def get_feedback(request: Request, session_id: str):
    manager = request.app.state.interview_manager
    try:
        res = await manager.get_feedback(session_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
