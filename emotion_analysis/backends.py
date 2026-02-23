"""
LLM backend implementations.

Each backend exposes a single method::

    def complete(self, system: str, user: str) -> str: ...

``EmotionAnalyzer`` delegates all API calls to a backend instance, so
switching providers is a one-line change.

Available backends
------------------
AnthropicBackend
    Uses the ``anthropic`` SDK.  Default model: ``claude-opus-4-6``.
    Reads ``ANTHROPIC_API_KEY`` from the environment by default.

OpenAIBackend
    Uses the ``openai`` SDK against any OpenAI-compatible endpoint.
    Default endpoint: OpenRouter (https://openrouter.ai/api/v1).
    Default model:    ``openai/gpt-5.2`` (OpenRouter model ID).
    Reads ``OPENROUTER_API_KEY`` (preferred) or ``OPENAI_API_KEY`` from
    the environment by default.

    Works with: OpenRouter, OpenAI, Azure OpenAI, Groq, Mistral, Ollama,
    LM Studio, vLLM, and any other OpenAI-compatible server.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-6"
OPENAI_DEFAULT_MODEL = "openai/gpt-5.2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Base / protocol
# ---------------------------------------------------------------------------

class LLMBackend:
    """
    Abstract base class for LLM backends.

    Subclasses must implement ``complete(system, user) -> str``.
    """

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        raise NotImplementedError(
            f"{type(self).__name__} must implement complete(system, user)"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class AnthropicBackend(LLMBackend):
    """
    Backend that calls the Anthropic Messages API via the ``anthropic`` SDK.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Defaults to the ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model ID.  Default: ``claude-opus-4-6``.
    max_tokens:
        Maximum tokens in the response.
    client:
        Pre-configured ``anthropic.Anthropic`` client (optional; useful for
        testing or custom retry/timeout settings).
    """

    DEFAULT_MODEL = ANTHROPIC_DEFAULT_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = ANTHROPIC_DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

        if client is not None:
            self._client = client
        else:
            import anthropic  # deferred import — optional dep
            self._client = anthropic.Anthropic(
                **({"api_key": api_key} if api_key else {})
            )

        logger.debug("AnthropicBackend ready: model=%s", model)

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def __repr__(self) -> str:
        return f"AnthropicBackend(model={self.model!r})"


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible chat-completions endpoint.

    Works with OpenRouter, OpenAI, Azure OpenAI, Groq, Mistral, Ollama,
    LM Studio, vLLM, and others.

    Parameters
    ----------
    api_key:
        API key for the provider.  If not given, the following env vars are
        checked in order:
        1. ``OPENROUTER_API_KEY`` (used when targeting OpenRouter)
        2. ``OPENAI_API_KEY``    (used when targeting OpenAI directly)
    model:
        Model identifier as understood by the endpoint.
        Default: ``openai/gpt-5.2`` (OpenRouter model ID).
        For OpenAI directly use e.g. ``gpt-4o``.
        For Groq use e.g. ``llama-3.3-70b-versatile``.
    base_url:
        Base URL for the API.  Default: ``https://openrouter.ai/api/v1``.
        Set to ``https://api.openai.com/v1`` for direct OpenAI access.
        Set to ``http://localhost:11434/v1`` for Ollama.
    max_tokens:
        Maximum tokens in the response.
    extra_headers:
        Additional HTTP headers forwarded with every request.
        OpenRouter recommends passing ``HTTP-Referer`` and ``X-Title``
        to identify your app in their dashboard.
    client:
        Pre-configured ``openai.OpenAI`` client (optional).

    Examples
    --------
    OpenRouter (default)::

        backend = OpenAIBackend()  # reads OPENROUTER_API_KEY

    OpenAI directly::

        backend = OpenAIBackend(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )

    Ollama (local)::

        backend = OpenAIBackend(
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key="ollama",  # Ollama ignores the key but the SDK requires one
        )

    Groq::

        backend = OpenAIBackend(
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
        )  # reads OPENAI_API_KEY or pass api_key=
    """

    DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
    DEFAULT_BASE_URL = OPENROUTER_BASE_URL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENAI_DEFAULT_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}

        if client is not None:
            self._client = client
        else:
            resolved_key = (
                api_key
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if not resolved_key:
                raise ValueError(
                    "No API key found for OpenAIBackend. "
                    "Set OPENROUTER_API_KEY or OPENAI_API_KEY, "
                    "or pass api_key= explicitly."
                )

            import openai  # deferred import — optional dep
            self._client = openai.OpenAI(
                api_key=resolved_key,
                base_url=base_url,
                default_headers=self.extra_headers,
            )

        logger.debug("OpenAIBackend ready: model=%s base_url=%s", model, base_url)

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"OpenAIBackend(model={self.model!r})"


# ---------------------------------------------------------------------------
# Auto-detection helper (used by EmotionAnalyzer when no backend is given)
# ---------------------------------------------------------------------------

def auto_detect_backend(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Optional[Any] = None,
) -> LLMBackend:
    """
    Automatically select and instantiate the best available backend based on
    environment variables.

    Priority order
    --------------
    1. If ``client`` is a pre-built ``anthropic.Anthropic`` instance → Anthropic.
    2. ``ANTHROPIC_API_KEY`` env var (or explicit ``api_key`` with no base_url) → Anthropic.
    3. ``OPENROUTER_API_KEY`` env var → OpenRouter (openai/gpt-5.2).
    4. ``OPENAI_API_KEY`` env var → OpenAI directly (gpt-4o).
    5. Raise ``ValueError`` with instructions.

    The ``model`` argument overrides the default for whichever backend is chosen.
    """
    # Explicit pre-built client — assume Anthropic (existing behaviour)
    if client is not None:
        return AnthropicBackend(
            model=model or ANTHROPIC_DEFAULT_MODEL,
            max_tokens=max_tokens,
            client=client,
        )

    # Explicit api_key with no way to tell which provider — try Anthropic first
    if api_key and not os.environ.get("OPENROUTER_API_KEY"):
        return AnthropicBackend(
            api_key=api_key,
            model=model or ANTHROPIC_DEFAULT_MODEL,
            max_tokens=max_tokens,
        )

    if os.environ.get("ANTHROPIC_API_KEY") or (
        api_key and not os.environ.get("OPENROUTER_API_KEY")
    ):
        return AnthropicBackend(
            api_key=api_key,
            model=model or ANTHROPIC_DEFAULT_MODEL,
            max_tokens=max_tokens,
        )

    if os.environ.get("OPENROUTER_API_KEY"):
        logger.info("Auto-detected OPENROUTER_API_KEY — using OpenAIBackend (OpenRouter)")
        return OpenAIBackend(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=model or OPENAI_DEFAULT_MODEL,
            base_url=OPENROUTER_BASE_URL,
            max_tokens=max_tokens,
        )

    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Auto-detected OPENAI_API_KEY — using OpenAIBackend (api.openai.com)")
        return OpenAIBackend(
            api_key=os.environ["OPENAI_API_KEY"],
            model=model or "gpt-4o",
            base_url=OPENAI_BASE_URL,
            max_tokens=max_tokens,
        )

    raise ValueError(
        "No LLM API key found. Set one of:\n"
        "  ANTHROPIC_API_KEY    — for Claude models\n"
        "  OPENROUTER_API_KEY   — for any model via OpenRouter\n"
        "  OPENAI_API_KEY       — for OpenAI directly\n"
        "Or pass backend= / api_key= to EmotionAnalyzer explicitly."
    )
