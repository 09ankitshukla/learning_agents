"""Provider-neutral types for talking to a language model.

Design decision worth understanding, because it shapes every lesson:

We use OpenAI's *message format* as the internal representation, even when the
model behind it is Qwen running on your CPU or Claude behind Anthropic's API.
That format won because everyone copied it, so treating it as the lingua franca
means one adapter (Anthropic) instead of N-squared translations.

A "message list" is the entire memory of an agent. There is no hidden state on
the server. Every turn you resend the whole conversation. Internalise that early
-- it explains context limits, cost growth, and most agent bugs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

# A message is a plain dict so it stays JSON-serialisable and easy to inspect,
# log, diff, and store in a test fixture. Resist the urge to wrap it in a class.
Message = dict[str, Any]


# --------------------------------------------------------------------------
# Message constructors
# --------------------------------------------------------------------------
def system(content: str) -> Message:
    """Instructions that frame the whole conversation. Usually message[0]."""
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    """Input from the human (or from an upstream system)."""
    return {"role": "user", "content": content}


def assistant(content: str | None = None, tool_calls: list[ToolCall] | None = None) -> Message:
    """A turn produced by the model.

    An assistant turn can carry text, tool calls, or both. When you append a
    model response back onto the message list you must preserve its tool calls
    verbatim, otherwise the following tool results have nothing to attach to and
    the provider rejects the request.
    """
    msg: Message = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [tc.to_wire() for tc in tool_calls]
    return msg


def tool_result(tool_call_id: str, content: str) -> Message:
    """The output of running a tool, fed back so the model can observe it.

    `tool_call_id` must match the id the model generated. This is how the model
    pairs a result with the request that produced it when several tools run in
    one turn.
    """
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# --------------------------------------------------------------------------
# Tool calling
# --------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """The description of a tool that gets sent to the model.

    This is documentation written for a model rather than a human, and it is the
    single highest-leverage thing you control in an agent. The model decides
    whether to call your tool based *only* on this text. Vague descriptions are
    the most common root cause of "my agent picked the wrong tool".
    """

    name: str
    description: str
    parameters: dict[str, Any]  # a JSON Schema object

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A model's request to run a tool.

    `arguments` is what the model produced, parsed into a dict. Note
    `malformed_arguments`: small local models regularly emit invalid JSON here.
    We capture that instead of raising, because "the model wrote bad JSON" is a
    normal runtime condition an agent has to recover from, not a crash. Lesson 1
    builds the recovery loop.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    malformed_arguments: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.malformed_arguments is None

    def to_wire(self) -> dict[str, Any]:
        raw = (
            self.malformed_arguments
            if self.malformed_arguments is not None
            else json.dumps(self.arguments)
        )
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": raw},
        }


@dataclass
class Usage:
    """Token and timing accounting for a single model call.

    Tracked from lesson 1 even though nothing uses it yet, because retrofitting
    measurement into an agent later is painful. Lesson 8 turns this into real
    cost and latency budgets.

    `reasoning_tokens` is a subset of `completion_tokens`, not an addition to
    them. Reasoning models (GPT-OSS, o-series, DeepSeek-R1, Qwen3 in thinking
    mode) generate hidden deliberation before their visible answer, and you pay
    for it at the output rate. It routinely dominates: in measured runs against
    gpt-oss-120b, 170 of 180 completion tokens were reasoning. If you budget
    `max_tokens` as though it all goes to the answer, you will be wrong by an
    order of magnitude.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def answer_tokens(self) -> int:
        """Completion tokens that became visible output."""
        return max(0, self.completion_tokens - self.reasoning_tokens)

    @property
    def tokens_per_second(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return self.completion_tokens / self.latency_s

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            latency_s=self.latency_s + other.latency_s,
        )


@dataclass
class LLMResponse:
    """One model reply, normalised across providers."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    reasoning: str | None = None
    raw: Any = None  # the untouched provider payload, for when you need to debug

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def truncated(self) -> bool:
        """Generation hit max_tokens instead of finishing.

        Check this before concluding a model "failed". A truncated reply produces
        unparseable JSON and half-written tool calls, which look like model
        incompetence but are really a budget you set too low.
        """
        return self.finish_reason == "length"

    @property
    def starved(self) -> bool:
        """A reasoning model spent the whole budget thinking and said nothing.

        A specific and confusing failure: the request succeeded, tokens were
        billed, and `text` is empty. Measured against gpt-oss-120b at
        reasoning_effort="high", 798 of 800 tokens went to reasoning and the
        visible answer never arrived. The fix is a larger max_tokens or lower
        reasoning effort -- not a different prompt.
        """
        return (
            not self.text
            and not self.tool_calls
            and self.usage.reasoning_tokens > 0
            and self.truncated
        )

    def as_message(self) -> Message:
        """Convert back into a message so it can be appended to the history.

        Note that `reasoning` is deliberately dropped. Providers do not accept
        reasoning back as input, and the model does not need it -- each turn
        re-derives its own thinking. Feeding it back wastes context.
        """
        return assistant(self.text, self.tool_calls)
