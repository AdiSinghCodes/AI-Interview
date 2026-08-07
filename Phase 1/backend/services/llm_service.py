import json
import httpx
from typing import List, Dict, Any
import uuid
from config import settings
from models.schemas import QuestionSchema, AnswerEvaluation, FeedbackResponse

DEFAULT_QUESTIONS = {
    "HR": [
        {"id": "q1", "question": "Tell me about yourself.", "category": "behavioral", "difficulty": "easy"},
        {"id": "q2", "question": "What are your greatest strengths and weaknesses?", "category": "behavioral", "difficulty": "medium"},
        {"id": "q3", "question": "Describe a time you faced a conflict at work.", "category": "situational", "difficulty": "hard"},
        {"id": "q4", "question": "Where do you see yourself in 5 years?", "category": "behavioral", "difficulty": "medium"},
        {"id": "q5", "question": "Why do you want to work here?", "category": "behavioral", "difficulty": "easy"},
    ],
    "Technical": [
        {"id": "q1", "question": "Explain the difference between process and thread.", "category": "technical", "difficulty": "medium"},
        {"id": "q2", "question": "What is REST API?", "category": "technical", "difficulty": "easy"},
        {"id": "q3", "question": "Explain dependency injection.", "category": "technical", "difficulty": "hard"},
        {"id": "q4", "question": "What is CI/CD?", "category": "technical", "difficulty": "medium"},
        {"id": "q5", "question": "How do you handle error logging?", "category": "technical", "difficulty": "medium"},
    ],
    "React": [
        {"id": "q1", "question": "What is the virtual DOM?", "category": "technical", "difficulty": "medium"},
        {"id": "q2", "question": "Explain useEffect hook.", "category": "technical", "difficulty": "medium"},
        {"id": "q3", "question": "What are React Server Components?", "category": "technical", "difficulty": "hard"},
        {"id": "q4", "question": "How do you manage state in React?", "category": "technical", "difficulty": "medium"},
        {"id": "q5", "question": "What are props in React?", "category": "technical", "difficulty": "easy"},
    ],
    "Python": [
        {"id": "q1", "question": "What is the GIL in Python?", "category": "technical", "difficulty": "hard"},
        {"id": "q2", "question": "Explain list comprehensions.", "category": "technical", "difficulty": "easy"},
        {"id": "q3", "question": "What are decorators?", "category": "technical", "difficulty": "medium"},
        {"id": "q4", "question": "Difference between tuple and list.", "category": "technical", "difficulty": "easy"},
        {"id": "q5", "question": "How does garbage collection work in Python?", "category": "technical", "difficulty": "hard"},
    ],
    "Data Analyst": [
        {"id": "q1", "question": "Explain p-value.", "category": "technical", "difficulty": "medium"},
        {"id": "q2", "question": "What is a JOIN in SQL?", "category": "technical", "difficulty": "easy"},
        {"id": "q3", "question": "How do you handle missing values?", "category": "technical", "difficulty": "medium"},
        {"id": "q4", "question": "Explain A/B testing.", "category": "technical", "difficulty": "medium"},
        {"id": "q5", "question": "What is normalization?", "category": "technical", "difficulty": "hard"},
    ],
}

class LLMService:
    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.model_ollama = settings.LLM_MODEL_OLLAMA
        self.model_groq = settings.LLM_MODEL_GROQ
        self.timeout = 30.0
        
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        if self.groq_api_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.groq_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model_groq,
                            "messages": messages,
                            "temperature": 0.7
                        },
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Groq API error: {e}, falling back to Ollama")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.model_ollama,
                        "messages": messages,
                        "stream": False
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
        except Exception as e:
            print(f"Ollama API error: {e}")
            raise Exception("No LLM available")

    async def generate_questions(self, role: str, difficulty: str, num_questions: int) -> List[QuestionSchema]:
        prompt = f"""You are an expert interviewer conducting a {role} interview at {difficulty} level.
Generate exactly {num_questions} interview questions.
Return ONLY a JSON array with this format:
[
  {{"question": "...", "category": "technical|behavioral|situational", "difficulty": "easy|medium|hard"}}
]
"""
        try:
            content = await self._call_llm([{"role": "system", "content": prompt}])
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                data = json.loads(json_str)
                questions = []
                for q in data:
                    questions.append(QuestionSchema(
                        id=str(uuid.uuid4()),
                        question=q["question"],
                        category=q.get("category", "technical"),
                        difficulty=q.get("difficulty", difficulty)
                    ))
                return questions
            else:
                raise ValueError("No JSON array found in response")
        except Exception as e:
            print(f"Error generating questions: {e}. Using fallback.")
            fallback = DEFAULT_QUESTIONS.get(role, DEFAULT_QUESTIONS["Technical"])
            return [QuestionSchema(**q) for q in fallback[:num_questions]]

    async def evaluate_answer(self, question: str, answer: str, role: str, difficulty: str) -> AnswerEvaluation:
        prompt = f"""Evaluate this candidate's answer for a {role} interview ({difficulty} level).
Question: {question}
Answer: {answer}

Return ONLY a JSON object with this format:
{{
  "scores": {{"relevance": 1-10, "depth": 1-10, "accuracy": 1-10, "communication": 1-10}},
  "feedback": "constructive feedback text",
  "follow_up_question": "a follow up question or null"
}}
"""
        try:
            content = await self._call_llm([{"role": "user", "content": prompt}])
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                data = json.loads(json_str)
                return AnswerEvaluation(
                    scores=data.get("scores", {"relevance": 5, "depth": 5, "accuracy": 5, "communication": 5}),
                    feedback=data.get("feedback", "Good attempt."),
                    follow_up_question=data.get("follow_up_question")
                )
            else:
                raise ValueError("No JSON object found in response")
        except Exception as e:
            print(f"Error evaluating answer: {e}. Using fallback.")
            return AnswerEvaluation(
                scores={"relevance": 7, "depth": 7, "accuracy": 7, "communication": 7},
                feedback="Thank you for your answer.",
                follow_up_question=None
            )

    async def generate_feedback(self, session_data: Any) -> FeedbackResponse:
        return FeedbackResponse(
            overall_score=7.5,
            category_scores={"technical": 7.0, "communication": 8.0},
            per_question_feedback=[],
            voice_summary={},
            proctoring_summary={},
            improvement_tips=["Practice speaking more slowly.", "Provide more concrete examples."]
        )
