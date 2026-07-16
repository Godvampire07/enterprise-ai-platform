import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from backend.app.core.config import settings
from backend.app.core.exceptions import LLMServiceError
from backend.app.services.llm.base import LLMResponse
from backend.app.services.llm.factory import get_llm_provider
from backend.app.services.llm.ollama_provider import OllamaProvider


@pytest.fixture
def mock_settings():
    old_provider = settings.LLM_PROVIDER
    old_url = settings.OLLAMA_BASE_URL
    old_model = settings.OLLAMA_MODEL
    old_retries = settings.LLM_MAX_RETRIES
    old_timeout = settings.LLM_TIMEOUT
    yield settings
    settings.LLM_PROVIDER = old_provider
    settings.OLLAMA_BASE_URL = old_url
    settings.OLLAMA_MODEL = old_model
    settings.LLM_MAX_RETRIES = old_retries
    settings.LLM_TIMEOUT = old_timeout


def test_ollama_provider_init_success(mock_settings):
    provider = OllamaProvider()
    assert provider.base_url == settings.OLLAMA_BASE_URL
    assert provider.model == settings.OLLAMA_MODEL


def test_factory_selection_ollama(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)


@pytest.mark.asyncio
async def test_generate_success(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)
    provider = OllamaProvider()

    mock_response_data = {
        "response": "Ollama response content",
        "done": True,
        "prompt_eval_count": 8,
        "eval_count": 12
    }

    # Mock httpx.AsyncClient.post
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_response)) as mock_post:
        res = await provider.generate(prompt="hi", system_prompt="be helpful")
        
        assert isinstance(res, LLMResponse)
        assert res.content == "Ollama response content"
        assert res.usage == {
            "prompt_tokens": 8,
            "completion_tokens": 12,
            "total_tokens": 20,
        }
        assert res.finish_reason == "stop"
        assert res.model == settings.OLLAMA_MODEL
        
        # Verify call arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = kwargs["json"]
        assert payload["model"] == settings.OLLAMA_MODEL
        assert payload["prompt"] == "hi"
        assert payload["system"] == "be helpful"
        assert payload["stream"] is False


@pytest.mark.asyncio
async def test_generate_http_error(mock_settings, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)
    provider = OllamaProvider()

    # Mock httpx response that raises an HTTP error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="Internal Server Error",
        request=MagicMock(),
        response=mock_response
    )

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_response)):
        with pytest.raises(LLMServiceError) as exc_info:
            await provider.generate(prompt="hi", system_prompt="be helpful")
        assert "Ollama provider failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_stream(mock_settings):
    provider = OllamaProvider()

    # Mock stream response chunks
    chunks = [
        b'{"response": "Hello", "done": false}',
        b'{"response": " world", "done": false}',
        b'{"response": "!", "done": true}'
    ]

    class MockAsyncIterator:
        def __init__(self):
            self.chunks = chunks

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.chunks:
                raise StopAsyncIteration
            return self.chunks.pop(0).decode("utf-8")

    class MockStreamContext:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines.return_value = MockAsyncIterator()

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext(mock_response)):
        stream_results = []
        async for chunk in provider.generate_stream(prompt="hi", system_prompt="be helpful"):
            stream_results.append(chunk)

        assert "".join(stream_results) == "Hello world!"
