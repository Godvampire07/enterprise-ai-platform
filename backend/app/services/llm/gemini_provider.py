"""
Gemini LLM Provider implementation.

Uses the google-genai SDK to communicate with Google's Gemini models.
Implements the LLMProvider interface with full support for:
  - Async generation via client.aio
  - System instructions via GenerateContentConfig
  - Temperature / max_tokens overrides
  - Retry with exponential backoff
  - Proper error mapping to LLMServiceError

This is the primary provider for the platform.
"""

import asyncio
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types

from backend.app.core.config import settings
from backend.app.core.exceptions import LLMServiceError
from backend.app.core.logging import logger
from backend.app.services.llm.base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Concrete LLM provider for Google Gemini models."""

    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise LLMServiceError(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file to use the Gemini provider."
            )
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = settings.LLM_MODEL
        self._default_temperature = settings.LLM_TEMPERATURE
        self._default_max_tokens = settings.LLM_MAX_TOKENS
        self._timeout = settings.LLM_TIMEOUT
        self._max_retries = settings.LLM_MAX_RETRIES

        logger.info(
            f"GeminiProvider initialized | model={self._model} | "
            f"temperature={self._default_temperature} | "
            f"max_tokens={self._default_max_tokens}"
        )

    def _build_config(
        self,
        system_prompt: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> types.GenerateContentConfig:
        """Build the GenerateContentConfig with system instructions and generation params."""
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature if temperature is not None else self._default_temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a completion using Gemini with retry logic.

        Retries on transient failures with exponential backoff.
        Maps all provider errors to LLMServiceError for consistent
        exception handling upstream.
        """
        config = self._build_config(system_prompt, temperature, max_tokens)
        last_exception: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    f"Gemini generate attempt {attempt}/{self._max_retries} | "
                    f"model={self._model} | prompt_length={len(prompt)}"
                )

                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self._model,
                        contents=prompt,
                        config=config,
                    ),
                    timeout=self._timeout,
                )

                # Extract response content
                if not response or not response.text:
                    raise LLMServiceError(
                        "Gemini returned an empty response. The model may have "
                        "refused to answer or the content was filtered."
                    )

                # Build normalized usage stats
                usage = {}
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": getattr(
                            response.usage_metadata, "prompt_token_count", 0
                        ),
                        "completion_tokens": getattr(
                            response.usage_metadata, "candidates_token_count", 0
                        ),
                        "total_tokens": getattr(
                            response.usage_metadata, "total_token_count", 0
                        ),
                    }

                # Extract finish reason
                finish_reason = None
                if response.candidates and response.candidates[0].finish_reason:
                    finish_reason = str(response.candidates[0].finish_reason)

                logger.info(
                    f"Gemini generation complete | tokens={usage.get('total_tokens', 'N/A')} | "
                    f"finish_reason={finish_reason}"
                )

                return LLMResponse(
                    content=response.text,
                    model=self._model,
                    usage=usage,
                    finish_reason=finish_reason,
                )

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(
                    f"Gemini request timed out after {self._timeout}s"
                )
                logger.warning(
                    f"Gemini timeout on attempt {attempt}/{self._max_retries}"
                )
            except LLMServiceError:
                # Don't retry on our own domain errors (e.g., empty response)
                raise
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Gemini error on attempt {attempt}/{self._max_retries}: {e}"
                )

            # Exponential backoff before retry
            if attempt < self._max_retries:
                backoff = 2 ** (attempt - 1)
                logger.debug(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        # All retries exhausted
        raise LLMServiceError(
            f"Gemini provider failed after {self._max_retries} attempts: "
            f"{str(last_exception)}"
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream a completion from Gemini token-by-token.

        Phase 2 feature — fully implemented but not yet called by the orchestrator.
        """
        config = self._build_config(system_prompt, temperature, max_tokens)

        try:
            response_stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=config,
            )

            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise LLMServiceError(f"Gemini streaming failed: {str(e)}")
