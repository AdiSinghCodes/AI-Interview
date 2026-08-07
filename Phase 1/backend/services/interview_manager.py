import uuid
from typing import Dict, Any
from models.schemas import InterviewSession, InterviewStartResponse, FeedbackResponse
from services.llm_service import LLMService

class InterviewManager:
    def __init__(self, llm_service: LLMService):
        self.sessions: Dict[str, InterviewSession] = {}
        self.llm_service = llm_service
        
    async def start_session(self, role: str, difficulty: str, num_questions: int) -> InterviewStartResponse:
        session_id = str(uuid.uuid4())
        questions = await self.llm_service.generate_questions(role, difficulty, num_questions)
        
        session = InterviewSession(
            session_id=session_id,
            role=role,
            difficulty=difficulty,
            questions=questions
        )
        self.sessions[session_id] = session
        
        greeting = f"Welcome to your {difficulty} {role} interview. I have {num_questions} questions for you today. Let's begin."
        return InterviewStartResponse(
            session_id=session_id,
            greeting=greeting,
            first_question=questions[0] if questions else None
        )
        
    async def submit_answer(self, session_id: str, answer_text: str, voice_metrics: Dict[str, Any]) -> dict:
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
            
        current_q = session.questions[session.current_question_index]
        
        evaluation = await self.llm_service.evaluate_answer(
            current_q.question, answer_text, session.role, session.difficulty
        )
        
        session.answers.append({
            "question_id": current_q.id,
            "answer": answer_text,
            "evaluation": evaluation.model_dump()
        })
        session.voice_metrics_history.append(voice_metrics)
        
        session.current_question_index += 1
        is_complete = session.current_question_index >= len(session.questions)
        
        if is_complete:
            session.status = "completed"
            
        next_q = None if is_complete else session.questions[session.current_question_index]
        
        return {
            "evaluation": evaluation.model_dump(),
            "next_question": next_q,
            "is_complete": is_complete
        }
        
    def get_session(self, session_id: str) -> InterviewSession:
        return self.sessions.get(session_id)
        
    async def get_feedback(self, session_id: str) -> FeedbackResponse:
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        return await self.llm_service.generate_feedback(session)
        
    def end_session(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if session:
            session.status = "completed"
            return {"status": "success", "message": "Interview ended"}
        return {"status": "error", "message": "Session not found"}
