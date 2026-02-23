"""
Tests for the backends module and EmotionAnalyzer backend wiring.
No real API calls are made — all SDK clients are mocked.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from emotion_analysis.backends import (
    AnthropicBackend,
    LLMBackend,
    OpenAIBackend,
    auto_detect_backend,
)
from emotion_analysis.analyzer import EmotionAnalyzer


# ---------------------------------------------------------------------------
# LLMBackend base class
# ---------------------------------------------------------------------------

def test_llm_backend_abstract():
    backend = LLMBackend()
    with pytest.raises(NotImplementedError):
        backend.complete("system", "user")


# ---------------------------------------------------------------------------
# AnthropicBackend
# ---------------------------------------------------------------------------

def _mock_anthropic_client(response_text: str) -> MagicMock:
    client = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    client.messages.create.return_value.content = [content_block]
    return client


def test_anthropic_backend_complete():
    client = _mock_anthropic_client("hello from claude")
    backend = AnthropicBackend(model="claude-opus-4-6", client=client)
    result = backend.complete(system="sys", user="usr")
    assert result == "hello from claude"
    client.messages.create.assert_called_once()


def test_anthropic_backend_passes_model():
    client = _mock_anthropic_client("")
    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", client=client)
    backend.complete("s", "u")
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


def test_anthropic_backend_passes_system_and_user():
    client = _mock_anthropic_client("")
    backend = AnthropicBackend(client=client)
    backend.complete("my system", "my user")
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "my system"
    assert call_kwargs["messages"][0]["content"] == "my user"


def test_anthropic_backend_passes_max_tokens():
    client = _mock_anthropic_client("")
    backend = AnthropicBackend(max_tokens=1234, client=client)
    backend.complete("s", "u")
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 1234


def test_anthropic_backend_default_model():
    assert AnthropicBackend.DEFAULT_MODEL == "claude-opus-4-6"


def test_anthropic_backend_repr():
    client = _mock_anthropic_client("")
    b = AnthropicBackend(model="claude-opus-4-6", client=client)
    assert "claude-opus-4-6" in repr(b)


# ---------------------------------------------------------------------------
# OpenAIBackend
# ---------------------------------------------------------------------------

def _mock_openai_client(response_text: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value.choices = [choice]
    return client


def test_openai_backend_complete():
    client = _mock_openai_client("hello from openai")
    backend = OpenAIBackend(client=client)
    result = backend.complete(system="sys", user="usr")
    assert result == "hello from openai"
    client.chat.completions.create.assert_called_once()


def test_openai_backend_passes_model():
    client = _mock_openai_client("")
    backend = OpenAIBackend(model="openai/gpt-5.2", client=client)
    backend.complete("s", "u")
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-5.2"


def test_openai_backend_messages_format():
    client = _mock_openai_client("")
    backend = OpenAIBackend(client=client)
    backend.complete("my system", "my user")
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "my system"}
    assert msgs[1] == {"role": "user", "content": "my user"}


def test_openai_backend_passes_max_tokens():
    client = _mock_openai_client("")
    backend = OpenAIBackend(max_tokens=2048, client=client)
    backend.complete("s", "u")
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 2048


def test_openai_backend_default_model():
    assert OpenAIBackend.DEFAULT_MODEL == "openai/gpt-5.2"


def test_openai_backend_default_base_url():
    assert "openrouter" in OpenAIBackend.DEFAULT_BASE_URL


def test_openai_backend_repr():
    client = _mock_openai_client("")
    b = OpenAIBackend(model="gpt-4o", client=client)
    assert "gpt-4o" in repr(b)


def test_openai_backend_no_key_raises():
    """Without any key in env or params, instantiation should fail."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove all key env vars
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                OpenAIBackend()  # no client, no key


def test_openai_backend_reads_openrouter_key():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"}):
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = _mock_openai_client("")
            backend = OpenAIBackend()
            # Should have been called with the OR key
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["api_key"] == "or-test-key"


def test_openai_backend_reads_openai_key_fallback():
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env["OPENAI_API_KEY"] = "oai-test-key"
    with patch.dict(os.environ, env, clear=True):
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = _mock_openai_client("")
            backend = OpenAIBackend()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["api_key"] == "oai-test-key"


# ---------------------------------------------------------------------------
# auto_detect_backend
# ---------------------------------------------------------------------------

def test_auto_detect_prefers_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("anthropic.Anthropic"):
        backend = auto_detect_backend()
    assert isinstance(backend, AnthropicBackend)


def test_auto_detect_uses_openrouter_when_no_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    with patch("openai.OpenAI"):
        backend = auto_detect_backend()
    assert isinstance(backend, OpenAIBackend)


def test_auto_detect_uses_openai_as_last_resort(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    with patch("openai.OpenAI"):
        backend = auto_detect_backend()
    assert isinstance(backend, OpenAIBackend)


def test_auto_detect_raises_when_no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No LLM API key"):
        auto_detect_backend()


def test_auto_detect_prefers_explicit_client():
    mock_client = MagicMock()
    mock_client.messages = MagicMock()  # looks like anthropic.Anthropic
    backend = auto_detect_backend(client=mock_client)
    assert isinstance(backend, AnthropicBackend)


# ---------------------------------------------------------------------------
# EmotionAnalyzer backend resolution
# ---------------------------------------------------------------------------

def test_analyzer_accepts_backend_instance():
    client = _mock_anthropic_client("")
    backend = AnthropicBackend(client=client)
    analyzer = EmotionAnalyzer(backend=backend)
    assert analyzer.backend is backend


def test_analyzer_string_anthropic():
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = _mock_anthropic_client("")
        analyzer = EmotionAnalyzer(backend="anthropic")
    assert isinstance(analyzer.backend, AnthropicBackend)


def test_analyzer_string_openai(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    with patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value = _mock_openai_client("")
        analyzer = EmotionAnalyzer(backend="openai")
    assert isinstance(analyzer.backend, OpenAIBackend)


def test_analyzer_string_openrouter_alias(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    with patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value = _mock_openai_client("")
        analyzer = EmotionAnalyzer(backend="openrouter")
    assert isinstance(analyzer.backend, OpenAIBackend)


def test_analyzer_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        EmotionAnalyzer(backend="gemini")


def test_analyzer_model_forwarded_to_anthropic():
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = _mock_anthropic_client("")
        analyzer = EmotionAnalyzer(backend="anthropic", model="claude-haiku-4-5-20251001")
    assert analyzer.backend.model == "claude-haiku-4-5-20251001"


def test_analyzer_model_forwarded_to_openai(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    with patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value = _mock_openai_client("")
        analyzer = EmotionAnalyzer(backend="openai", model="google/gemini-2.5-pro")
    assert analyzer.backend.model == "google/gemini-2.5-pro"
