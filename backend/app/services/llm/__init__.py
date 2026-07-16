"""
LLM Provider package.

Exposes the factory function as the primary public API.
All consumer code should use `get_llm_provider()` rather than
importing concrete providers directly.
"""

from backend.app.services.llm.factory import get_llm_provider
from backend.app.services.llm.base import LLMProvider, LLMResponse

__all__ = ["get_llm_provider", "LLMProvider", "LLMResponse"]
