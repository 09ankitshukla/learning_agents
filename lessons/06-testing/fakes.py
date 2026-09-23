"""Test doubles for a language model, and the recorder that makes them realistic.

The central problem of testing an agent: the model is non-deterministic, slow,
costs money, and needs a network. Every one of those is fatal to a test suite you
will actually run.

The solution is not "mock the HTTP layer". If you patch `httpx` or the OpenAI SDK
you end up asserting that your code can talk to a fake HTTP server, which is a
test of the SDK rather than of your agent. The interesting logic -- did the
dispatcher refuse that tool, did the loop stop for the right reason, did trimming
orphan a tool result -- is all above that layer.

Substitute at the seam instead. `llmkit.LLMClient` is a Protocol with one method
that matters:

    chat(messages, tools=None, ...) -> LLMResponse

So anything with that shape is a drop-in model. That Protocol was defined in
lesson 0 partly for provider-swapping and partly for exactly this: it is the seam
that makes an agent testable. Designing for a substitutable boundary before you
need one is most of what makes code testable later.

Two doubles, for two different jobs:

  * `ScriptedClient` -- you write the responses by hand. Best for testing specific
    behaviour, especially failures you cannot reliably provoke from a real model
    (malformed tool arguments, a phantom tool, token starvation).
  * `CassetteClient` -- replays responses recorded from a real model. Best for
    testing realistic flows, because it preserves the model's actual quirks:
    `content: null` on tool turns, reasoning-token accounting, the exact JSON it
    emits.

Use both. Scripted doubles test the code paths you care about; cassettes stop your
fakes from drifting into fiction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmkit import LLMConfig, LLMResponse, Message, ToolCall, ToolSpec, Usage

CASSETTE_DIR = Path(__file__).parent / "cassettes"


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------
def fake_config(model: str = "fake-model") -> LLMConfig:
    return LLMConfig(
        provider="fake",
        model=model,
        base_url=None,
        api_key="not-needed",
        timeout=10.0,
        openai_compatible=True,
    )


def text_response(text: str, *, reasoning_tokens: int = 0) -> LLMResponse:
    """A model turn that returns prose and asks for nothing. Ends the agent loop."""
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=max(1, len(text) // 4),
            reasoning_tokens=reasoning_tokens,
            latency_s=0.0,
        ),
        model="fake-model",
    )


def tool_response(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str = "call_1",
    malformed: str | None = None,
    text: str | None = None,
) -> LLMResponse:
    """A model turn that requests one tool.

    Note `content=None` by default: a real tool-calling turn carries no prose, and
    a fake that returns text here would hide bugs in code that assumes otherwise.
    `malformed` produces the invalid-JSON case that small models really do emit.
    """
    call = (
        ToolCall(id=call_id, name=name, malformed_arguments=malformed)
        if malformed is not None
        else ToolCall(id=call_id, name=name, arguments=arguments or {})
    )
    return LLMResponse(
        text=text,
        tool_calls=[call],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=120, completion_tokens=30, latency_s=0.0),
        model="fake-model",
    )


def multi_tool_response(calls: list[tuple[str, dict[str, Any]]]) -> LLMResponse:
    """A turn requesting several tools at once.

    Worth testing explicitly: every requested call needs its own result appended
    before the next model call, and getting that wrong is rejected by real
    providers. A suite that only ever exercises one tool per turn never checks it.
    """
    return LLMResponse(
        text=None,
        tool_calls=[
            ToolCall(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls, start=1)
        ],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=140, completion_tokens=50, latency_s=0.0),
        model="fake-model",
    )


def starved_response() -> LLMResponse:
    """A reasoning model that spent its whole budget thinking. Lesson 0's trap.

    Empty text, no tool calls, finish_reason="length", reasoning tokens billed.
    Impossible to provoke on demand from a real model, trivial to construct here --
    which is precisely the argument for hand-written doubles.
    """
    return LLMResponse(
        text="",
        tool_calls=[],
        finish_reason="length",
        usage=Usage(
            prompt_tokens=100, completion_tokens=800, reasoning_tokens=798, latency_s=0.0
        ),
        model="fake-model",
    )


# ---------------------------------------------------------------------------
# ScriptedClient
# ---------------------------------------------------------------------------
@dataclass
class RecordedCall:
    """What the code under test actually sent. Assert on this."""

    messages: list[Message]
    tools: list[str]
    max_tokens: int
    tool_choice: str

    @property
    def roles(self) -> list[str]:
        return [m.get("role", "?") for m in self.messages]


class ScriptedClient:
    """Returns pre-programmed responses in order. No network, no cost, no variance.

    Also records every request, which matters as much as the responses: a lot of
    agent bugs are in what you *send* (a trimmed conversation that orphans a tool
    result, tool schemas missing, a budget that is too small), and those are only
    visible if the double keeps the requests.
    """

    def __init__(self, responses: list[LLMResponse], *, strict: bool = True) -> None:
        self._responses = list(responses)
        self._position = 0
        self.calls: list[RecordedCall] = []
        self.config = fake_config()
        #: When strict, running out of scripted responses is an error rather than
        #: silently repeating the last one. Silent repetition turns "my loop ran
        #: 40 times" into a passing test.
        self.strict = strict

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
        # Copy the messages: the caller keeps mutating its own list, so storing a
        # reference would leave every recorded call pointing at the final state.
        self.calls.append(
            RecordedCall(
                messages=[dict(m) for m in messages],
                tools=[t.name for t in (tools or [])],
                max_tokens=max_tokens,
                tool_choice=tool_choice,
            )
        )

        if self._position >= len(self._responses):
            if self.strict:
                raise AssertionError(
                    f"ScriptedClient ran out of responses after {self._position} call(s). "
                    f"The code under test made more model calls than the script expected."
                )
            return self._responses[-1]

        response = self._responses[self._position]
        self._position += 1
        return response

    # Protocol conformance.
    def health(self) -> tuple[bool, str]:
        return True, "scripted"

    def available_models(self) -> list[str]:
        return ["fake-model"]

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def exhausted(self) -> bool:
        return self._position >= len(self._responses)


# ---------------------------------------------------------------------------
# Cassettes
# ---------------------------------------------------------------------------
def request_key(messages: list[Message], tools: list[str]) -> str:
    """Stable identity for a request, used to look up a recorded response.

    Hashing the whole conversation means a cassette entry matches only the exact
    request that produced it. That is the right trade for tests: if you change the
    system prompt, the cassette misses and you are told to re-record, rather than
    being handed a response recorded for a different prompt and quietly testing
    nothing.
    """
    payload = json.dumps(
        {"messages": messages, "tools": sorted(tools)}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _response_to_json(response: LLMResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "finish_reason": response.finish_reason,
        "model": response.model,
        "reasoning": response.reasoning,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "reasoning_tokens": response.usage.reasoning_tokens,
        },
        "tool_calls": [
            {
                "id": c.id,
                "name": c.name,
                "arguments": c.arguments,
                "malformed_arguments": c.malformed_arguments,
            }
            for c in response.tool_calls
        ],
    }


def _response_from_json(raw: dict[str, Any]) -> LLMResponse:
    usage = raw.get("usage") or {}
    return LLMResponse(
        text=raw.get("text"),
        tool_calls=[
            ToolCall(
                id=c["id"],
                name=c["name"],
                arguments=c.get("arguments") or {},
                malformed_arguments=c.get("malformed_arguments"),
            )
            for c in raw.get("tool_calls") or []
        ],
        finish_reason=raw.get("finish_reason"),
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            latency_s=0.0,
        ),
        model=raw.get("model", "cassette"),
        reasoning=raw.get("reasoning"),
    )


@dataclass
class Cassette:
    """A recorded conversation with a real model, replayable offline."""

    name: str
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return CASSETTE_DIR / f"{self.name}.json"

    def save(self) -> Path:
        CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"name": self.name, "entries": self.entries}, indent=2),
            encoding="utf-8",
        )
        return self.path

    @classmethod
    def load(cls, name: str) -> Cassette:
        path = CASSETTE_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No cassette at {path}.\n"
                f"Record one with:\n"
                f"  uv run lessons/06-testing/record.py --scenario {name}"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(name=raw["name"], entries=raw.get("entries", []))


class CassetteClient:
    """Replays recorded model responses. Deterministic, offline, and realistic.

    Looks up by request hash first, then falls back to sequential order. The
    fallback exists because an agent's later requests contain earlier tool results,
    so a hash mismatch is common as soon as anything upstream changes -- and a
    sequential replay of a real run is still far better than a hand-written guess.
    """

    def __init__(self, cassette: Cassette, *, strict_match: bool = False) -> None:
        self.cassette = cassette
        self.strict_match = strict_match
        # Map each key to the entry indices recorded under it. A list rather than a
        # single index because a conversation can legitimately repeat a request.
        self._by_key: dict[str, list[int]] = {}
        for index, entry in enumerate(cassette.entries):
            self._by_key.setdefault(entry["key"], []).append(index)
        #: Entries already served. Tracked so key lookups and sequential fallback
        #: cannot hand out the same response twice.
        #:
        #: The first version of this class kept a separate `_position` counter that
        #: key hits did not advance, so a key hit followed by a sequential fallback
        #: replayed entry 0 again. The agent then saw two identical tool calls and
        #: the loop reported STALLED -- a bug in the test double masquerading as a
        #: bug in the code under test. Caught by test_cassettes.py.
        self._used: set[int] = set()
        self.calls: list[RecordedCall] = []
        self.config = fake_config(model=f"cassette:{cassette.name}")
        self.key_hits = 0
        self.sequential_hits = 0

    def _next_unused(self) -> int | None:
        for index in range(len(self.cassette.entries)):
            if index not in self._used:
                return index
        return None

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
        tool_names = [t.name for t in (tools or [])]
        self.calls.append(
            RecordedCall(
                messages=[dict(m) for m in messages],
                tools=tool_names,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
            )
        )

        key = request_key(messages, tool_names)
        for index in self._by_key.get(key, []):
            if index not in self._used:
                self._used.add(index)
                self.key_hits += 1
                return _response_from_json(self.cassette.entries[index]["response"])

        if self.strict_match:
            raise AssertionError(
                f"Cassette {self.cassette.name!r} has no unused entry for this request "
                f"(key {key}). The conversation changed since recording; re-record "
                f"or set strict_match=False."
            )

        index = self._next_unused()
        if index is None:
            raise AssertionError(
                f"Cassette {self.cassette.name!r} exhausted after "
                f"{len(self._used)} call(s). The code under test made more model "
                f"calls than were recorded."
            )
        self._used.add(index)
        self.sequential_hits += 1
        return _response_from_json(self.cassette.entries[index]["response"])

    def health(self) -> tuple[bool, str]:
        return True, f"cassette:{self.cassette.name}"

    def available_models(self) -> list[str]:
        return [self.config.model]


class RecordingClient:
    """Wraps a real client and records every exchange into a cassette.

    Deliberately a wrapper rather than a flag on the real client: recording is a
    development-time concern and should not be able to leak into a normal run.
    """

    def __init__(self, inner, cassette_name: str) -> None:
        self.inner = inner
        self.cassette = Cassette(name=cassette_name)
        self.config = inner.config

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        response = self.inner.chat(messages, tools=tools, **kwargs)
        tool_names = [t.name for t in (tools or [])]
        self.cassette.entries.append(
            {
                "key": request_key([dict(m) for m in messages], tool_names),
                "roles": [m.get("role") for m in messages],
                "tools_offered": tool_names,
                "response": _response_to_json(response),
            }
        )
        return response

    def health(self):
        return self.inner.health()

    def available_models(self):
        return self.inner.available_models()

    def save(self) -> Path:
        return self.cassette.save()
