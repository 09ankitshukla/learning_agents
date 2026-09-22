"""Saving and resuming a conversation.

Deliberately small. Persistence is plumbing rather than a concept, and the concept
it rests on was already settled in lesson 0: **the message list is the entire
state of an agent.** There is nothing on the server. So "save a session" means
"write a JSON array to disk", and "resume" means "read it back and keep
appending".

That is genuinely the whole idea. The reason it gets its own file is the three
things that go wrong.

**1. Tool calls must round-trip exactly.** `tool_call_id` values pair a result with
its request. Lose them, reorder them, or regenerate them on load and the provider
rejects the conversation. Since messages are plain JSON-serialisable dicts
throughout this repo (a decision made in `llmkit/types.py` for exactly this
reason), `json.dump` preserves them for free -- but only if you never "helpfully"
normalise the structure on the way in or out.

**2. A saved conversation is untrusted input.** It is a file on disk that a
process reads and sends to a model. If something else can write it, something else
controls your agent's instructions. Hence `load_session` validates structure and
rejects anything malformed rather than trusting it.

**3. Resumed conversations are already long.** A session saved at 6,000 tokens
reloads at 6,000 tokens, so it needs compaction before the first new call, not
after. Persistence and context management are the same problem viewed twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmkit import Message

from context import conversation_tokens, validate

SESSION_VERSION = 1

VALID_ROLES = {"system", "user", "assistant", "tool"}


class SessionError(RuntimeError):
    """A session file is unusable, with a reason a human can act on."""


@dataclass
class Session:
    """A conversation plus the metadata needed to make sense of it later."""

    messages: list[Message] = field(default_factory=list)
    question: str | None = None
    model: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    #: Free-form counters: steps taken, tokens spent, compactions applied.
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return conversation_tokens(self.messages)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": SESSION_VERSION,
            "question": self.question,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stats": self.stats,
            "messages": self.messages,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_session(path: str | Path, session: Session) -> Path:
    """Write a session to disk as JSON.

    Validates before writing. Saving a conversation that is already structurally
    broken just means the bug reappears later, somewhere harder to trace.
    """
    problems = validate(session.messages)
    if problems:
        raise SessionError(
            "Refusing to save a structurally invalid conversation:\n  - "
            + "\n  - ".join(problems)
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    session.created_at = session.created_at or _now()
    session.updated_at = _now()
    session.stats.setdefault("messages", len(session.messages))
    session.stats["estimated_tokens"] = session.tokens

    # Write to a temporary file and replace, so an interrupted save cannot leave a
    # half-written session that fails to parse on the next run.
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(session.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def load_session(path: str | Path) -> Session:
    """Read a session back, validating as we go.

    Treats the file as untrusted. Everything here is a check that a hand-edited or
    corrupted file cannot quietly turn into a malformed API request, or worse, an
    injected system prompt.
    """
    source = Path(path)
    if not source.exists():
        raise SessionError(f"No session file at {source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionError(f"{source} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise SessionError(f"{source} should contain a JSON object.")

    version = raw.get("version")
    if version != SESSION_VERSION:
        # A real product would migrate. Refusing loudly beats loading a shape you
        # no longer understand and failing three steps later.
        raise SessionError(
            f"{source} has version {version!r}, expected {SESSION_VERSION}. "
            f"Delete it or migrate it."
        )

    messages = raw.get("messages")
    if not isinstance(messages, list):
        raise SessionError(f"{source}: 'messages' must be a list.")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise SessionError(f"{source}: message {index} is not an object.")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise SessionError(
                f"{source}: message {index} has role {role!r}; "
                f"expected one of {sorted(VALID_ROLES)}."
            )
        if role == "tool" and not message.get("tool_call_id"):
            raise SessionError(
                f"{source}: message {index} is a tool result with no tool_call_id. "
                f"It cannot be matched to a request."
            )

    problems = validate(messages)
    if problems:
        raise SessionError(
            f"{source} contains a structurally invalid conversation:\n  - "
            + "\n  - ".join(problems)
        )

    return Session(
        messages=messages,
        question=raw.get("question"),
        model=raw.get("model"),
        created_at=raw.get("created_at"),
        updated_at=raw.get("updated_at"),
        stats=raw.get("stats") or {},
    )
