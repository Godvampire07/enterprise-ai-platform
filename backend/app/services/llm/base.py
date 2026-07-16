"""
Abstract base class for LLM providers.

Defines the contract that every provider (Gemini, OpenAI, Ollama) must
implement. Uses the Strategy Pattern — the active provider is selected
at runtime via the LLM_PROVIDER environment variable, and all business
logic programs against this interface, never a concrete class.

LLMResponse is a plain dataclass (not Pydantic) because it is an
internal domain object, not an API boundary.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Optional


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider.

    Attributes:
        content: The generated text response.
        model: The model identifier that produced the response.
        usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens).
        finish_reason: Why the model stopped generating (stop, max_tokens, etc.).
    """

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None


class LLMProvider(ABC):
    """Strategy interface for LLM providers.

    Each concrete provider wraps a specific SDK (google-genai, openai, httpx)
    and normalizes the response into an LLMResponse. This allows the
    RAGOrchestrator to remain provider-agnostic.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a single completion given a user prompt and system prompt.

        Args:
            prompt: The user-facing prompt (contains context + question).
            system_prompt: System-level instructions for grounding behavior.
            temperature: Sampling temperature override. Uses provider default if None.
            max_tokens: Maximum output token count override.

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            LLMServiceError: On provider API failures, timeouts, or invalid responses.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream a completion token-by-token.

        Reserved for Phase 2 streaming support. Concrete providers should
        implement this but the orchestrator does not call it yet.

        Yields:
            Individual text chunks as they arrive from the provider.
        """
        ...
