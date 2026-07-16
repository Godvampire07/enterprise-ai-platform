"""
OpenRouter LLM Provider implementation.

Uses the openai SDK configured with OpenRouter base URL to communicate with OpenRouter models.
Implements the LLMProvider interface with full support for:
  - Async generation via client.chat.completions.create
  - System instructions via system role message
  - Temperature / max_tokens overrides
  - Retry with exponential backoff
  - Proper error mapping to LLMServiceError

This is an alternative provider for the platform.
"""

import asyncio
from typing import AsyncIterator, Optional

import openai
from openai import AsyncOpenAI

from backend.app.core.config import settings
from backend.app.core.exceptions import LLMServiceError
from backend.app.core.logging import logger
from backend.app.services.llm.base import LLMProvider, LLMResponse


class OpenRouterProvider(LLMProvider):
    """Concrete LLM provider for OpenRouter (OpenAI-compatible) models."""

    def __init__(self) -> None:
        if not settings.OPENROUTER_API_KEY:
            raise LLMServiceError(
                "OPENROUTER_API_KEY is not configured. "
                "Set it in your .env file to use the OpenRouter provider."
            )
        self._client: Optional[AsyncOpenAI] = None
        self._api_key = settings.OPENROUTER_API_KEY
        self._base_url = settings.OPENROUTER_BASE_URL
        self._model = settings.OPENROUTER_MODEL
        self._default_temperature = settings.LLM_TEMPERATURE
        self._default_max_tokens = settings.LLM_MAX_TOKENS
        self._timeout = settings.LLM_TIMEOUT
        self._max_retries = settings.LLM_MAX_RETRIES

        logger.info(
            f"OpenRouterProvider initialized | model={self._model} | "
            f"temperature={self._default_temperature} | "
            f"max_tokens={self._default_max_tokens}"
        )

    def _get_client(self) -> AsyncOpenAI:
        """Lazily initialize the AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a completion using OpenRouter with retry logic.

        Retries on transient failures with exponential backoff.
        Maps all provider errors to LLMServiceError for consistent
        exception handling upstream.
        """
        client = self._get_client()
        last_exception: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    f"OpenRouter generate attempt {attempt}/{self._max_retries} | "
                    f"model={self._model} | prompt_length={len(prompt)}"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                # Use asyncio.wait_for for timeout handling
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature if temperature is not None else self._default_temperature,
                        max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
                    ),
                    timeout=self._timeout,
                )

                # Extract response content
                if not response or not response.choices or not response.choices[0].message.content:
                    raise LLMServiceError(
                        "OpenRouter returned an empty response. The model may have "
                        "refused to answer or the content was filtered."
                    )

                # Build normalized usage stats
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }

                # Extract finish reason
                finish_reason = None
                if response.choices and response.choices[0].finish_reason:
                    finish_reason = str(response.choices[0].finish_reason)

                logger.info(
                    f"OpenRouter generation complete | tokens={usage.get('total_tokens', 'N/A')} | "
                    f"finish_reason={finish_reason}"
                )

                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=self._model,
                    usage=usage,
                    finish_reason=finish_reason,
                )

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(
                    f"OpenRouter request timed out after {self._timeout}s"
                )
                logger.warning(
                    f"OpenRouter timeout on attempt {attempt}/{self._max_retries}"
                )
            except LLMServiceError:
                # Don't retry on our own domain errors (e.g., empty response)
                raise
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"OpenRouter error on attempt {attempt}/{self._max_retries}: {e}"
                )

            # Exponential backoff before retry
            if attempt < self._max_retries:
                backoff = 2 ** (attempt - 1)
                logger.debug(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        # All retries exhausted
        raise LLMServiceError(
            f"OpenRouter provider failed after {self._max_retries} attempts: "
            f"{str(last_exception)}"
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream a completion from OpenRouter token-by-token.

        Phase 2 feature — fully implemented but not yet called by the orchestrator.
        """
        client = self._get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response_stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature if temperature is not None else self._default_temperature,
                max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
                stream=True,
            )

            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenRouter streaming error: {e}")
            raise LLMServiceError(f"OpenRouter streaming failed: {str(e)}")
