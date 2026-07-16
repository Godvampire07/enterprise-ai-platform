"""
LLM Provider factory.

Resolves the active LLM provider at runtime based on the LLM_PROVIDER
environment variable. This is the single point of provider resolution —
no other module should import concrete providers directly.

Adding a new provider requires:
  1. Creating a new module in services/llm/ that implements LLMProvider.
  2. Adding a single entry to the PROVIDER_REGISTRY dict below.
"""

from typing import Dict, Type

from backend.app.core.config import settings
from backend.app.core.exceptions import LLMServiceError
from backend.app.core.logging import logger
from backend.app.services.llm.base import LLMProvider


# Lazy import registry — providers are only imported when selected.
# This avoids loading unnecessary SDKs (e.g., openai when using gemini).
def _get_gemini_provider() -> LLMProvider:
    from backend.app.services.llm.gemini_provider import GeminiProvider
    return GeminiProvider()


def _get_openrouter_provider() -> LLMProvider:
    from backend.app.services.llm.openrouter_provider import OpenRouterProvider
    return OpenRouterProvider()

def _get_ollama_provider() -> LLMProvider:
    from backend.app.services.llm.ollama_provider import OllamaProvider
    return OllamaProvider()

# Registry of available providers.
# Phase 2 will add "openai" and "ollama" entries here.
PROVIDER_REGISTRY: Dict[str, callable] = {
    "gemini": _get_gemini_provider,
    "openrouter": _get_openrouter_provider,
    "ollama": _get_ollama_provider,
}


def get_llm_provider() -> LLMProvider:
    """Resolve and instantiate the configured LLM provider.

    Reads LLM_PROVIDER from settings and returns the corresponding
    concrete provider instance.

    Returns:
        An initialized LLMProvider implementation.

    Raises:
        LLMServiceError: If the configured provider is not in the registry.
    """
    provider_name = settings.LLM_PROVIDER.lower()

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise LLMServiceError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Available providers: {available}"
        )

    logger.info(f"Initializing LLM provider: {provider_name}")
    return PROVIDER_REGISTRY[provider_name]()
