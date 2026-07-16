import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from backend.app.core.config import settings
from backend.app.core.exceptions import LLMServiceError
from backend.app.services.llm.base import LLMResponse
from backend.app.services.llm.factory import get_llm_provider
from backend.app.services.llm.openrouter_provider import OpenRouterProvider


@pytest.fixture
def mock_settings():
    old_provider = settings.LLM_PROVIDER
    old_key = settings.OPENROUTER_API_KEY
    old_url = settings.OPENROUTER_BASE_URL
    old_model = settings.OPENROUTER_MODEL
    old_retries = settings.LLM_MAX_RETRIES
    old_timeout = settings.LLM_TIMEOUT
    yield settings
    settings.LLM_PROVIDER = old_provider
    settings.OPENROUTER_API_KEY = old_key
    settings.OPENROUTER_BASE_URL = old_url
    settings.OPENROUTER_MODEL = old_model
    settings.LLM_MAX_RETRIES = old_retries
    settings.LLM_TIMEOUT = old_timeout


def test_openrouter_provider_init_missing_key(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    with pytest.raises(LLMServiceError) as exc_info:
        OpenRouterProvider()
    assert "OPENROUTER_API_KEY is not configured" in str(exc_info.value)


def test_openrouter_provider_init_success(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")
    provider = OpenRouterProvider()
    assert provider._api_key == "fake-key"
    assert provider._client is None  # Should be lazily initialized


def test_factory_selection(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")
    provider = get_llm_provider()
    assert isinstance(provider, OpenRouterProvider)


def test_factory_selection_unsupported(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "invalid-provider")
    with pytest.raises(LLMServiceError) as exc_info:
        get_llm_provider()
    assert "Unknown LLM provider 'invalid-provider'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_success(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)

    provider = OpenRouterProvider()

    mock_client = MagicMock()
    mock_completions = AsyncMock()

    # Mock completion object response
    mock_response = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = "Hello there!"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15
    mock_response.usage = mock_usage

    mock_completions.create.return_value = mock_response
    mock_client.chat.completions = mock_completions

    # Inject mock client
    provider._client = mock_client

    res = await provider.generate(prompt="hi", system_prompt="be nice")

    assert isinstance(res, LLMResponse)
    assert res.content == "Hello there!"
    assert res.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert res.finish_reason == "stop"
    assert res.model == settings.OPENROUTER_MODEL

    # Verify create was called with correct parameters
    mock_completions.create.assert_called_once_with(
        model=settings.OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


@pytest.mark.asyncio
async def test_generate_empty_response(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)

    provider = OpenRouterProvider()

    mock_client = MagicMock()
    mock_completions = AsyncMock()

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_completions.create.return_value = mock_response
    mock_client.chat.completions = mock_completions
    provider._client = mock_client

    with pytest.raises(LLMServiceError) as exc_info:
        await provider.generate(prompt="hi", system_prompt="be nice")
    assert "returned an empty response" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_retry_logic(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 3)

    provider = OpenRouterProvider()

    mock_client = MagicMock()
    mock_completions = AsyncMock()

    # Fail twice, succeed on third attempt
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Success at last!"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_completions.create.side_effect = [
        openai.APIError("API Error 1", request=None, body=None),
        Exception("Unexpected error"),
        mock_response,
    ]
    mock_client.chat.completions = mock_completions
    provider._client = mock_client

    # Mock asyncio.sleep to avoid waiting in tests
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await provider.generate(prompt="hi", system_prompt="be nice")
        assert res.content == "Success at last!"
        assert mock_completions.create.call_count == 3
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_generate_retries_exhausted(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)

    provider = OpenRouterProvider()

    mock_client = MagicMock()
    mock_completions = AsyncMock()

    mock_completions.create.side_effect = openai.APIConnectionError(
        request=None, message="Connection failed"
    )
    mock_client.chat.completions = mock_completions
    provider._client = mock_client

    with patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(LLMServiceError) as exc_info:
            await provider.generate(prompt="hi", system_prompt="be nice")
        assert "failed after 2 attempts" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_timeout_handling(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(settings, "LLM_TIMEOUT", 1)

    provider = OpenRouterProvider()

    mock_client = MagicMock()
    mock_completions = AsyncMock()

    async def slow_create(*args, **kwargs):
        await asyncio.sleep(5)
        return MagicMock()

    mock_completions.create.side_effect = slow_create
    mock_client.chat.completions = mock_completions
    provider._client = mock_client

    with patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(LLMServiceError) as exc_info:
            await provider.generate(prompt="hi", system_prompt="be nice")
        assert "timed out after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_stream_success(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")

    provider = OpenRouterProvider()

    mock_client = MagicMock()
    mock_completions = AsyncMock()

    # Create chunks for async streaming
    class AsyncChunkIterator:
        def __init__(self, chunks):
            self.chunks = chunks
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.chunks):
                raise StopAsyncIteration
            val = self.chunks[self.index]
            self.index += 1
            return val

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta = MagicMock()
    chunk1.choices[0].delta.content = "Chunk 1"

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta = MagicMock()
    chunk2.choices[0].delta.content = " Chunk 2"

    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta = MagicMock()
    chunk3.choices[0].delta.content = None

    iterator = AsyncChunkIterator([chunk1, chunk2, chunk3])
    mock_completions.create.return_value = iterator
    mock_client.chat.completions = mock_completions
    provider._client = mock_client

    chunks = []
    async for chunk in provider.generate_stream(prompt="hi", system_prompt="be nice"):
        chunks.append(chunk)

    assert chunks == ["Chunk 1", " Chunk 2"]


@pytest.mark.asyncio
async def test_generate_stream_error(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake-key")

    provider = OpenRouterProvider()

    mock_client = MagicMock()
    mock_completions = AsyncMock()
    mock_completions.create.side_effect = Exception("Streaming error")
    mock_client.chat.completions = mock_completions
    provider._client = mock_client

    with pytest.raises(LLMServiceError) as exc_info:
        async for chunk in provider.generate_stream(prompt="hi", system_prompt="be nice"):
            pass
    assert "OpenRouter streaming failed" in str(exc_info.value)
