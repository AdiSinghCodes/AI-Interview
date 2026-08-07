import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL_OLLAMA: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    LLM_MODEL_GROQ: str = os.getenv("LLM_MODEL_GROQ", "llama-3.1-8b-instant")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "*"]
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
