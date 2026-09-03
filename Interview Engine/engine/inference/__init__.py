"""
The swap boundary for every LLM call.

RULE: nothing outside this package imports httpx-against-Ollama, the groq
SDK, or any other inference client directly. Callers hold one
`InferenceProvider` and never know which backend ran — routing is read off
each call's `LLMSpec.provider`, set only in config/models.yaml.
"""

from .provider import ChatMessage, GenerationResult, InferenceProvider
from .routing_provider import RoutingProvider

__all__ = ["ChatMessage", "GenerationResult", "InferenceProvider", "RoutingProvider"]
