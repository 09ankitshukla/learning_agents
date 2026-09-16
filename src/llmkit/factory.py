"""One entry point every lesson uses to get a model client."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import LLMConfig, load_config
from .types import LLMResponse, Message, ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """The whole surface an agent needs from a model.

    Deliberately tiny. Every lesson in this repo -- tool use, memory, retrieval,
    multi-agent, evaluation -- is built on this one method. If you remember one
    thing from the course, make it this: an agent is a program that calls
    `chat()` in a loop and does something with the result.
    """

    config: LLMConfig

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
        stop: list[str] | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse: ...

    def health(self) -> tuple[bool, str]: ...

    def available_models(self) -> list[str]: ...


def get_client(
    provider: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
) -> LLMClient:
    """Return a client for the configured provider.

    Usage in a lesson:

        from llmkit import get_client, user
        client = get_client()
        print(client.chat([user("hi")]).text)

    Override for a one-off comparison without touching .env:

        fast = get_client(provider="groq", model="llama-3.3-70b-versatile")
    """
    config = load_config(provider=provider, model=model, timeout=timeout)

    if config.openai_compatible:
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(config)

    if config.provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(config)

    raise ValueError(f"No client implementation for provider {config.provider!r}")
