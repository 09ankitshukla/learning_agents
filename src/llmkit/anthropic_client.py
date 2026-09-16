"""Anthropic adapter.

Only needed if someone sets LLM_PROVIDER=anthropic. Kept in its own module so
the default local path never imports it.

Reading this file is worthwhile even if you never use Claude: it shows exactly
which parts of the "message list plus tools" model are universal and which are
provider dressing. The concepts map one-to-one; only the JSON shape differs.

Differences from the OpenAI format:
  * the system prompt is a top-level parameter, not a message
  * tool schemas use `input_schema` rather than nesting under `function`
  * tool calls arrive as `tool_use` content blocks inside the assistant turn
  * tool results go back as `tool_result` blocks in a *user* turn, not a
    dedicated "tool" role
  * max_tokens is required
"""

from __future__ import annotations

import json
import time
from typing import Any

from anthropic import Anthropic

from .config import LLMConfig
from .types import LLMResponse, Message, ToolCall, ToolSpec, Usage


class AnthropicClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = Anthropic(api_key=config.api_key, timeout=config.timeout)

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,  # noqa: ARG002 - handled by prompting instead
        stop: list[str] | None = None,
        reasoning_effort: str | None = None,  # noqa: ARG002 - uses a token budget instead
        tool_choice: str = "auto",
    ) -> LLMResponse:
        system_prompt, converted = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": converted,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
            # Anthropic spells these differently: auto / any / none.
            kwargs["tool_choice"] = {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "none": {"type": "none"},
            }.get(tool_choice, {"type": "auto"})
        if stop:
            kwargs["stop_sequences"] = stop

        started = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        elapsed = time.perf_counter() - started

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                latency_s=elapsed,
            ),
            model=response.model,
            raw=response,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _convert_messages(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Translate OpenAI-format messages into Anthropic's shape."""
        system_chunks: list[str] = []
        out: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                system_chunks.append(msg.get("content") or "")

            elif role == "user":
                out.append({"role": "user", "content": msg.get("content") or ""})

            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for call in msg.get("tool_calls") or []:
                    fn = call["function"]
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": fn["name"],
                            "input": args,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})

            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": msg.get("content") or "",
                }
                # Consecutive tool results must be merged into one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})

        return ("\n\n".join(c for c in system_chunks if c) or None), out

    # ------------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        return True, "hosted provider (no local health check)"

    def available_models(self) -> list[str]:
        return []
