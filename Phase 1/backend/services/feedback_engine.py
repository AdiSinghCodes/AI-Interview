from typing import Dict, Any, List
from models.schemas import InterviewSession

class FeedbackEngine:
    def compute_scores(self, session: InterviewSession) -> Dict[str, float]:
        if not session.answers:
            return {"answer_quality": 0, "communication": 0, "confidence": 0, "voice_clarity": 0}
            
        total_rel = sum(a["evaluation"]["scores"].get("relevance", 5) for a in session.answers)
        total_depth = sum(a["evaluation"]["scores"].get("depth", 5) for a in session.answers)
        total_acc = sum(a["evaluation"]["scores"].get("accuracy", 5) for a in session.answers)
        total_comm = sum(a["evaluation"]["scores"].get("communication", 5) for a in session.answers)
        
        n = len(session.answers)
        answer_quality = (total_rel + total_depth + total_acc) / (3 * n)
        communication = total_comm / n
        
        confidence = 7.5  # Stub
        voice_clarity = 8.0  # Stub
        
        return {
            "answer_quality": round(answer_quality, 1),
            "communication": round(communication, 1),
            "confidence": confidence,
            "voice_clarity": voice_clarity
        }
        
    def generate_improvement_tips(self, scores: Dict[str, float]) -> List[str]:
        tips = []
        if scores.get("answer_quality", 10) < 7:
            tips.append("Try to provide more detailed and accurate answers.")
        if scores.get("communication", 10) < 7:
            tips.append("Work on articulating your thoughts more clearly.")
        if scores.get("confidence", 10) < 7:
            tips.append("Practice to build confidence in your delivery.")
        if not tips:
            tips.append("Keep up the good work!")
        return tips
