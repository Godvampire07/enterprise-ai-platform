"""Ollama LLM Provider implementation.

Uses httpx to communicate with a local/remote Ollama instance.
Implements the LLMProvider interface with support for:
  - Async generation via httpx
  - System instructions via system prompt
  - Temperature / max_tokens overrides
  - Retry with exponential backoff
  - Proper error mapping to LLMServiceError
"""

import asyncio
import json
from typing import AsyncIterator, Dict, Optional

import httpx

from backend.app.core.config import settings
from backend.app.core.exceptions import LLMServiceError
from backend.app.core.logging import logger
from backend.app.services.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Concrete LLM provider for Ollama models."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        
        logger.info(
            f"OllamaProvider initialized | base_url={self.base_url} | model={self.model}"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a completion using Ollama with retry logic."""
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = settings.LLM_TEMPERATURE
        
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        else:
            options["num_predict"] = settings.LLM_MAX_TOKENS

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": options,
            "stream": False,
        }

        last_exception: Optional[Exception] = None

        for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"Ollama generate attempt {attempt}/{settings.LLM_MAX_RETRIES} | "
                    f"model={self.model} | prompt_length={len(prompt)}"
                )

                async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                if not data or "response" not in data:
                    raise LLMServiceError("Ollama returned an empty or invalid response.")

                # Build usage stats
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

                return LLMResponse(
                    content=data["response"],
                    model=self.model,
                    usage=usage,
                    finish_reason="stop" if data.get("done") else None,
                )

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"Ollama timeout on attempt {attempt}/{settings.LLM_MAX_RETRIES}"
                )
            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.warning(
                    f"Ollama HTTP error on attempt {attempt}/{settings.LLM_MAX_RETRIES}: {e}"
                )
            except LLMServiceError:
                raise
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Ollama error on attempt {attempt}/{settings.LLM_MAX_RETRIES}: {e}"
                )

            # Exponential backoff before retry
            if attempt < settings.LLM_MAX_RETRIES:
                backoff = 2 ** (attempt - 1)
                logger.debug(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        raise LLMServiceError(
            f"Ollama provider failed after {settings.LLM_MAX_RETRIES} attempts: "
            f"{str(last_exception)}"
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream a completion from Ollama token-by-token."""
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = settings.LLM_TEMPERATURE
        
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        else:
            options["num_predict"] = settings.LLM_MAX_TOKENS

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": options,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                            content = chunk_data.get("response", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode Ollama stream line: {line}")
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            raise LLMServiceError(f"Ollama stream error: {e}") from e
