from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class InterviewStartRequest(BaseModel):
    role: str
    difficulty: str
    num_questions: int = 5

class QuestionSchema(BaseModel):
    id: str
    question: str
    category: str
    difficulty: str

class InterviewStartResponse(BaseModel):
    session_id: str
    greeting: str
    first_question: QuestionSchema

class AnswerSubmitRequest(BaseModel):
    answer_text: str
    audio_duration: float = 0.0
    voice_metrics: Dict[str, Any] = Field(default_factory=dict)

class AnswerEvaluation(BaseModel):
    scores: Dict[str, int]
    feedback: str
    follow_up_question: Optional[str] = None

class VoiceAnalysisResult(BaseModel):
    volume_db: float
    speaking_rate_wpm: float
    filler_count: int
    fillers_detected: List[str]
    silence_ratio: float
    feedback_messages: List[str]

class FeedbackResponse(BaseModel):
    overall_score: float
    category_scores: Dict[str, float]
    per_question_feedback: List[Dict[str, Any]]
    voice_summary: Dict[str, Any]
    proctoring_summary: Dict[str, Any]
    improvement_tips: List[str]

class InterviewSession(BaseModel):
    session_id: str
    role: str
    difficulty: str
    questions: List[QuestionSchema]
    current_question_index: int = 0
    answers: List[Dict[str, Any]] = Field(default_factory=list)
    voice_metrics_history: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "active"
