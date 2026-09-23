"""Client for any OpenAI-compatible server.

Covers Ollama, llama.cpp, vLLM, LM Studio, Groq, OpenRouter, Together and
OpenAI. We use the official `openai` SDK purely as a well-tested HTTP client for
that wire format -- it is not tied to OpenAI's own models.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

from .config import ConfigError, LLMConfig
from .types import LLMResponse, Message, ToolCall, ToolSpec, Usage


class LocalServerDown(RuntimeError):
    """The local model server is not reachable. Almost always Ollama not running."""


class PhantomToolCall(RuntimeError):
    """The model tried to call a tool that was never offered to it.

    Not hypothetical, and worth understanding because it surprises people.

    Models are trained with tools, and that behaviour does not switch off when
    you stop offering any. Asked for the current time with `tools=None`,
    gpt-oss-120b attempted to call `container.exec` -- a code-execution tool from
    its training environment -- to run a Python script. Groq refused the
    generation and returned HTTP 400 rather than silently degrading.

    Two lessons. First, "no tools" does not reliably mean "the model will admit
    it cannot know"; it may reach for a phantom capability instead. Second, a
    model requesting a tool you never published is a real runtime event, which is
    exactly why a dispatcher must reject unknown names by default.
    """

    def __init__(self, message: str, attempted: str | None = None) -> None:
        super().__init__(message)
        self.attempted = attempted


class OpenAICompatClient:
    """Thin, synchronous wrapper that normalises requests and responses."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=2,  # network-level retries only; not semantic retries
        )

    # ------------------------------------------------------------------
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
    ) -> LLMResponse:
        """Send a conversation, get one reply back.

        temperature=0.0 is the default on purpose. While learning you want runs
        to be as reproducible as they can be, so that when behaviour changes it
        is because you changed something. Note "as they can be": even at
        temperature 0 most servers are not bit-for-bit deterministic, because
        floating-point reduction order varies with batching. Lesson 6 deals with
        this properly.

        max_tokens defaults to 1024 rather than something tighter because
        reasoning models spend most of their output budget on hidden
        deliberation. Set this too low and you get an empty answer, not a short
        one.

        json_mode asks the server to constrain output to valid JSON. Support is
        uneven across models, which is exactly why lesson 1 also builds a
        parse-and-repair loop that works without it.

        reasoning_effort ("low" | "medium" | "high") is passed through for models
        that support it. It is a real quality/cost dial and it changes answers,
        not just verbosity -- in a measured run on gpt-oss-120b, "low" and
        "medium" produced different categories for the same ticket. Sent via
        extra_body so it degrades gracefully on servers that ignore it.

        tool_choice decides who chooses:
          "auto"     - the model decides whether to call a tool (the default, and
                       what you want almost always)
          "required" - the model MUST call some tool; it cannot reply with prose
          "none"     - tools are visible but must not be called
        Use "required" when a tool call is the only acceptable outcome and you do
        not want to depend on the model's judgement. It is the difference between
        asking nicely and removing the option.
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [t.to_wire() for t in tools]
            kwargs["tool_choice"] = tool_choice
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stop:
            kwargs["stop"] = stop
        if reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except APIConnectionError as exc:
            raise self._connection_error(exc) from exc
        except APIStatusError as exc:
            raise self._status_error(exc) from exc
        elapsed = time.perf_counter() - started

        return self._normalise(completion, elapsed)

    # ------------------------------------------------------------------
    def _normalise(self, completion: Any, elapsed: float) -> LLMResponse:
        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for i, raw_call in enumerate(getattr(message, "tool_calls", None) or []):
            raw_args = raw_call.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
                # Some models wrap arguments in an extra layer or emit a bare
                # scalar. Only a JSON object is usable as kwargs.
                if not isinstance(parsed, dict):
                    raise ValueError("arguments must be a JSON object")
                tool_calls.append(
                    ToolCall(
                        id=raw_call.id or f"call_{i}",
                        name=raw_call.function.name,
                        arguments=parsed,
                    )
                )
            except (json.JSONDecodeError, ValueError):
                # Preserved rather than raised: recovering from this is the
                # agent's job, and small local models do it often.
                tool_calls.append(
                    ToolCall(
                        id=raw_call.id or f"call_{i}",
                        name=raw_call.function.name,
                        malformed_arguments=raw_args,
                    )
                )

        usage_obj = getattr(completion, "usage", None)

        # Reasoning models report hidden deliberation separately. The field is a
        # provider extension, so it arrives in model_extra rather than as a typed
        # attribute -- hence the defensive lookup.
        details = getattr(usage_obj, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) or 0

        usage = Usage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning_tokens,
            latency_s=elapsed,
        )

        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            model=getattr(completion, "model", self.config.model),
            reasoning=self._extract_reasoning(message),
            raw=completion,
        )

    @staticmethod
    def _extract_reasoning(message: Any) -> str | None:
        """Pull the reasoning trace out, whatever the provider chose to call it.

        Groq uses `reasoning`; others use `reasoning_content`. Both are
        non-standard extensions, so they land in the SDK model's extra fields.
        """
        for attr in ("reasoning", "reasoning_content"):
            value = getattr(message, attr, None)
            if value:
                return str(value)
        extra = getattr(message, "model_extra", None) or {}
        for key in ("reasoning", "reasoning_content"):
            if extra.get(key):
                return str(extra[key])
        return None

    # ------------------------------------------------------------------
    def _connection_error(self, exc: Exception) -> Exception:
        if self.config.is_local:
            return LocalServerDown(
                f"Cannot reach {self.config.provider} at {self.config.base_url}\n\n"
                "Fix:\n"
                "  1. Is the server running?  ollama serve\n"
                f"  2. Is the model pulled?    ollama pull {self.config.model}\n"
                "  3. Re-run the check:       uv run lessons/00-setup/check_env.py\n"
                f"\nUnderlying error: {exc}"
            )
        return ConfigError(
            f"Cannot reach {self.config.provider} at {self.config.base_url or 'default endpoint'}.\n"
            f"Check your network and LLM_BASE_URL.\n\nUnderlying error: {exc}"
        )

    def _status_error(self, exc: APIStatusError) -> Exception:
        if exc.status_code == 400:
            phantom = self._as_phantom_tool_call(exc)
            if phantom is not None:
                return phantom

        if exc.status_code == 404 and self.config.is_local:
            return LocalServerDown(
                f"Server is up but model {self.config.model!r} is unknown.\n\n"
                f"Fix:  ollama pull {self.config.model}\n"
                f"See what you have:  ollama list"
            )
        if exc.status_code in (401, 403):
            return ConfigError(
                f"{self.config.provider} rejected the API key ({exc.status_code}).\n"
                "Check LLM_API_KEY in .env, or switch back to LLM_PROVIDER=ollama."
            )
        if exc.status_code == 413:
            # "Request too large" is a different problem from "too many requests",
            # and conflating them sends you down the wrong path. 429 means wait;
            # 413 means the single request you just built does not fit, and waiting
            # will never help. The fix is context management (lesson 4) or fewer
            # tools, not a retry.
            body = exc.body if isinstance(exc.body, dict) else {}
            error = body.get("error") if isinstance(body.get("error"), dict) else body
            provider_message = str(error.get("message", "")).strip()
            return ConfigError(
                f"{self.config.provider} refused the request as too large (HTTP 413).\n\n"
                + (f"{self.config.provider} says:\n  {provider_message}\n\n" if provider_message else "")
                + "Waiting will not help -- this single request exceeds a per-request or\n"
                "per-minute token limit. Reduce what you send:\n"
                "  - apply context management (lesson 4): compress tool results,\n"
                "    trim old exchanges, or summarise\n"
                "  - return fewer or smaller tool results (lower top_k, truncate)\n"
                "  - offer fewer tools: every schema is re-sent on every call\n"
                "  - lower max_tokens if the limit counts input plus output"
            )

        if exc.status_code == 429:
            # Pass the provider's own message through. It usually says which limit
            # was hit (per-minute vs per-day, requests vs tokens) and when to retry,
            # and that distinction decides what you do next: wait 30 seconds, or
            # stop for the day and switch to a local model.
            detail = ""
            body = exc.body if isinstance(exc.body, dict) else {}
            error = body.get("error") if isinstance(body.get("error"), dict) else body
            provider_message = str(error.get("message", "")).strip()
            if provider_message:
                detail = f"\n\n{self.config.provider} says:\n  {provider_message}"

            return ConfigError(
                f"Rate limited by {self.config.provider} (HTTP 429).{detail}\n\n"
                "Options:\n"
                "  - wait for the window to reset (the message above usually says how long)\n"
                "  - switch to a local model, which has no quota:\n"
                "      LLM_PROVIDER=ollama\n"
                "      LLM_MODEL=qwen2.5:7b-instruct\n"
                "  - use a smaller model to stretch a token quota further:\n"
                "      LLM_MODEL=openai/gpt-oss-20b\n\n"
                "Note that agent loops burn quota fast: every step re-sends the whole "
                "conversation, so a handful of multi-step runs can consume a daily "
                "token allowance."
            )
        return exc

    @staticmethod
    def _as_phantom_tool_call(exc: APIStatusError) -> PhantomToolCall | None:
        """Recognise "the model called a tool that wasn't on offer".

        Providers signal this inconsistently, so match on both the error code and
        the message text.
        """
        # Providers disagree about nesting, and the SDK sometimes unwraps for you:
        # Groq surfaces the inner dict directly as `.body`, while others keep it
        # under an "error" key. Accept both rather than guessing.
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        code = str(error.get("code", ""))
        message = str(error.get("message", ""))

        looks_like_phantom = code == "tool_use_failed" or "model called a tool" in message
        if not looks_like_phantom:
            return None

        attempted = error.get("failed_generation")
        detail = f"\n\nWhat it tried to call:\n{attempted}" if attempted else ""

        return PhantomToolCall(
            "The model tried to call a tool that was not offered to it, and the "
            "provider refused the generation.\n\n"
            "This usually happens when you pass no tools but ask for something the "
            "model cannot know (the current time, a live price). Rather than saying "
            "'I cannot know', it reaches for a tool from its training environment - "
            "commonly a code interpreter.\n\n"
            "Fixes: offer a real tool for the capability, or instruct the model in the "
            "system prompt to answer from its own knowledge and state uncertainty "
            "plainly." + detail,
            attempted=str(attempted) if attempted else None,
        )

    # ------------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe used by check_env.py. Never raises.

        Uses the authenticated SDK client rather than a bare HTTP GET, because
        hosted providers answer an unauthenticated /v1/models with 401 -- which
        looks like "server down" when it actually means "server fine, key
        missing". Local servers accept the placeholder key, so one path covers
        both.
        """
        try:
            models = self._client.models.list()
            count = len(models.data)
            return True, f"reachable, {count} model(s) available"
        except APIStatusError as exc:
            if exc.status_code in (401, 403):
                return False, f"HTTP {exc.status_code} - API key rejected"
            return False, f"HTTP {exc.status_code}"
        except APIConnectionError:
            return False, "unreachable - no response from server"
        except Exception as exc:  # noqa: BLE001 - diagnostics, not control flow
            return False, f"unreachable: {type(exc).__name__}"

    def available_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list().data)
        except Exception:  # noqa: BLE001
            return []
