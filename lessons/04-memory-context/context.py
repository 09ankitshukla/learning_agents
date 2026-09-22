"""Context management: keeping a conversation inside a token budget.

Lesson 3 ended with a measurement, not a cliffhanger. Prompt tokens across four
steps of the research task:

    step 1:   824
    step 2: 1,127   (+303)
    step 3: 3,466   (+2,339)
    step 4: 6,122   (+2,656)

11,539 input tokens to answer one question. The model is stateless (lesson 0), so
the entire conversation -- every tool result included -- is re-sent on every call.
Cost therefore grows with roughly the *square* of the step count. A 10-step agent
is nearer 50x a 1-step agent on input tokens, not 10x.

Two things eventually break:

  * **Cost and latency**, immediately and quadratically.
  * **The context window**, a hard wall. Cross it and you get an error, or worse,
    silent truncation that looks like the agent suddenly forgetting the task.

This file is the toolkit for staying inside the budget. The important idea is
that context management is a **transformation of the message list**, applied just
before sending. The agent loop does not change at all -- lesson 3's loop gained a
single optional hook and nothing else.

The one trap that will bite you: a message list is not a flat sequence of
independent items. An assistant turn carrying tool_calls and the tool messages
answering it are an **atomic group**. Split them and the provider rejects the
entire request. `trim_naive` demonstrates this failure on purpose; `trim_safe`
respects it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from llmkit import LLMClient, Message, system, user

# ---------------------------------------------------------------------------
# 1. Measuring
# ---------------------------------------------------------------------------
# You cannot know the exact token count without the model's own tokenizer, and
# there is no universal one -- every model family has its own vocabulary. Options,
# in order of accuracy:
#
#   1. Ask the provider. `usage.prompt_tokens` is exact, but only *after* the call,
#      which is too late to decide what to send.
#   2. Run the real tokenizer locally (tiktoken for OpenAI models,
#      transformers.AutoTokenizer for open weights). Accurate, but it is a real
#      dependency and you must match the tokenizer to the model or it silently
#      lies to you.
#   3. Estimate from character count. Crude, dependency-free, and good enough for
#      budgeting *if* you verify the error and lean conservative.
#
# This lesson uses (3) and then calibrates against (1), because the habit that
# matters is checking your estimator against reality rather than trusting it.
# `agent.py --calibrate` reports the measured error.

# The constants below are CALIBRATED, not guessed, and the calibration mattered
# enormously. The first version of this file used chars/3.7 plus 4 tokens per
# message and nothing else. Measured against Groq's reported `prompt_tokens` it
# was **91% too low** on a short prompt: it estimated 7 tokens where the provider
# charged 79.
#
# The missing piece was a large fixed per-*request* cost. gpt-oss is served with
# the "harmony" chat template, which injects its own preamble (date, reasoning
# configuration, channel markers) before your messages ever appear. You pay for
# that on every call regardless of what you send.
#
# The lesson generalises past this model: an uncalibrated token estimator is not
# conservative, it is simply wrong, and wrong in the dangerous direction. Run
# `agent.py --calibrate` whenever you change model or provider.

#: Characters per token for ordinary English prose.
CHARS_PER_TOKEN_PROSE = 3.9

#: Characters per token for JSON, code and paths. Denser: more punctuation and
#: more rare tokens, so the same character count costs more tokens.
CHARS_PER_TOKEN_STRUCTURED = 2.8

#: Per-message structural overhead (role markers, separators).
PER_MESSAGE_OVERHEAD = 4

#: Fixed per-request overhead from the chat template. Measured ~70 for
#: gpt-oss-120b via Groq. Re-measure for any other model -- this is the single
#: biggest source of estimator error on short conversations.
REQUEST_OVERHEAD = 70


def _chars_per_token(text: str) -> float:
    """Pick a density based on how structured the text looks.

    Crude but effective: JSON and code are punctuation-heavy, and punctuation
    tokenizes far less efficiently than words. A tool result full of file paths
    and braces costs roughly 40% more tokens per character than prose.
    """
    if not text:
        return CHARS_PER_TOKEN_PROSE
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return (
        CHARS_PER_TOKEN_STRUCTURED
        if symbols / len(text) > 0.12
        else CHARS_PER_TOKEN_PROSE
    )


def estimate_tokens(text: str | None) -> int:
    """Rough token count for a string. Deliberately an estimate, not a promise."""
    if not text:
        return 0
    return max(1, int(len(text) / _chars_per_token(text)))


def message_tokens(message: Message) -> int:
    """Estimated cost of one message, including its tool calls.

    Note that tool_calls are counted too. It is easy to forget they exist -- the
    assistant message that requests a tool often has `content: null`, so a naive
    len(content) counter scores it as zero while it may carry several hundred
    tokens of JSON arguments.
    """
    total = PER_MESSAGE_OVERHEAD
    total += estimate_tokens(message.get("content"))

    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        total += estimate_tokens(function.get("name"))
        total += estimate_tokens(function.get("arguments"))
        total += 8  # id, type, and JSON structure

    return total


def conversation_tokens(messages: Iterable[Message]) -> int:
    """Estimated cost of sending this conversation, including request overhead.

    REQUEST_OVERHEAD is added once here rather than per message, because it is a
    property of the request rather than of the content. Leaving it out is what
    made the original estimator 91% low on short conversations.
    """
    return REQUEST_OVERHEAD + sum(message_tokens(m) for m in messages)


def tool_schema_tokens(specs: Iterable) -> int:
    """Tool schemas are prompt text too, and they are re-sent on every call.

    Worth measuring separately, because it is a fixed cost you pay on every single
    step regardless of how well you trim the conversation. Six verbose tools can
    outweigh the entire early conversation.
    """
    return sum(estimate_tokens(json.dumps(spec.to_wire())) for spec in specs)


@dataclass
class Breakdown:
    """Where the budget actually went. Usually a surprise the first time."""

    rows: list[tuple[int, str, int, str]] = field(default_factory=list)
    total: int = 0

    @property
    def by_role(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for _, role, tokens, _ in self.rows:
            totals[role] = totals.get(role, 0) + tokens
        return totals

    @property
    def largest(self) -> tuple[int, str, int, str] | None:
        return max(self.rows, key=lambda r: r[2]) if self.rows else None


def breakdown(messages: list[Message]) -> Breakdown:
    """Per-message token accounting, so you can see what to attack first."""
    result = Breakdown()
    for index, message in enumerate(messages):
        tokens = message_tokens(message)
        preview = (message.get("content") or "")
        if not preview and message.get("tool_calls"):
            preview = "[tool_calls] " + ", ".join(
                c["function"]["name"] for c in message["tool_calls"]
            )
        result.rows.append((index, message.get("role", "?"), tokens, str(preview)[:70]))
        result.total += tokens
    return result


# ---------------------------------------------------------------------------
# 2. Atomic groups: the trap
# ---------------------------------------------------------------------------
@dataclass
class Group:
    """An indivisible unit of conversation.

    Most messages stand alone. But an assistant turn with `tool_calls` and the
    `tool` messages that answer it must travel together: each tool result carries
    a `tool_call_id` that has to match a request the model can see. Break the pair
    and the provider rejects the whole conversation with a 400.

    This is the single most common bug in hand-written context management, and it
    does not show up until the conversation is long enough to trim.
    """

    indices: list[int]
    messages: list[Message]
    kind: str  # "system" | "task" | "exchange" | "single"

    @property
    def tokens(self) -> int:
        return conversation_tokens(self.messages)

    @property
    def droppable(self) -> bool:
        # The system prompt defines behaviour and the first user message is the
        # task itself. Dropping either is never an optimisation -- an agent that
        # forgets what it was asked is worse than one that runs out of budget.
        return self.kind not in ("system", "task")


def group_messages(messages: list[Message]) -> list[Group]:
    """Partition a conversation into units that can be kept or dropped whole."""
    groups: list[Group] = []
    index = 0
    seen_first_user = False

    while index < len(messages):
        message = messages[index]
        role = message.get("role")

        if role == "system":
            groups.append(Group([index], [message], "system"))
            index += 1
            continue

        if role == "user" and not seen_first_user:
            seen_first_user = True
            groups.append(Group([index], [message], "task"))
            index += 1
            continue

        if role == "assistant" and message.get("tool_calls"):
            # Absorb every tool result that follows, however many there are.
            span = [index]
            payload = [message]
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                span.append(cursor)
                payload.append(messages[cursor])
                cursor += 1
            groups.append(Group(span, payload, "exchange"))
            index = cursor
            continue

        groups.append(Group([index], [message], "single"))
        index += 1

    return groups


def validate(messages: list[Message]) -> list[str]:
    """Check a message list for problems a provider would reject. Returns issues.

    Run this in tests and after any trimming. It catches the orphan bug locally
    instead of as an opaque HTTP 400 three steps into a run.
    """
    problems: list[str] = []
    open_ids: set[str] = set()

    for index, message in enumerate(messages):
        role = message.get("role")

        if role == "assistant":
            for call in message.get("tool_calls") or []:
                open_ids.add(call["id"])

        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in open_ids:
                problems.append(
                    f"message {index}: tool result for '{call_id}' has no matching "
                    f"assistant tool_call earlier in the list (orphaned result)"
                )
            else:
                open_ids.discard(call_id)

    if open_ids:
        problems.append(
            f"tool_call(s) {sorted(open_ids)} were requested but never answered "
            f"(orphaned request)"
        )

    return problems


# ---------------------------------------------------------------------------
# 3. Strategies
# ---------------------------------------------------------------------------
@dataclass
class Compaction:
    """What a strategy did, so it can be reported and measured."""

    messages: list[Message]
    strategy: str
    before: int
    after: int
    dropped_groups: int = 0
    compressed: int = 0
    summarised: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def saved(self) -> int:
        return self.before - self.after

    @property
    def ratio(self) -> float:
        return (self.after / self.before) if self.before else 1.0


def trim_naive(messages: list[Message], keep_last: int = 6) -> Compaction:
    """Keep the system prompt and the last N messages. BROKEN ON PURPOSE.

    This is the obvious implementation, and it is what almost everyone writes
    first. It treats the list as flat, so a cut can land between an assistant's
    tool_calls and the tool results answering them.

    Run `agent.py --orphan` to watch a real provider reject the output of this
    function. It is included precisely so the failure is memorable.
    """
    before = conversation_tokens(messages)
    head = [m for m in messages if m.get("role") == "system"]
    tail = messages[-keep_last:] if keep_last else []
    kept = head + [m for m in tail if m.get("role") != "system"]

    result = Compaction(
        messages=kept,
        strategy="naive",
        before=before,
        after=conversation_tokens(kept),
        dropped_groups=len(messages) - len(kept),
    )
    issues = validate(kept)
    if issues:
        result.notes.append(f"INVALID: {issues[0]}")
    return result


def trim_safe(
    messages: list[Message],
    budget: int,
    keep_recent_exchanges: int = 2,
) -> Compaction:
    """Drop whole exchanges, oldest first, until the conversation fits.

    Three rules, and each exists because of a specific way the naive version
    fails:

      1. Never split an exchange (assistant tool_calls + its tool results).
      2. Never drop the system prompt or the original task.
      3. Always keep the most recent exchanges -- they are what the model needs
         next. Dropping the newest to keep the oldest is backwards.

    When something is dropped we leave a marker in its place. Silent amnesia makes
    a model confabulate about work it can no longer see; a note saying "3 earlier
    steps were removed" lets it say "I checked earlier but no longer have the
    detail", which is far better behaviour.
    """
    before = conversation_tokens(messages)
    groups = group_messages(messages)

    protected_head = [g for g in groups if not g.droppable]
    body = [g for g in groups if g.droppable]

    keep_tail = body[-keep_recent_exchanges:] if keep_recent_exchanges else []
    candidates = body[: len(body) - len(keep_tail)]

    fixed_cost = sum(g.tokens for g in protected_head + keep_tail)
    room = budget - fixed_cost

    kept_middle: list[Group] = []
    # Walk newest-to-oldest so the most recent survivors win the remaining room.
    for group in reversed(candidates):
        if group.tokens <= room:
            kept_middle.insert(0, group)
            room -= group.tokens

    dropped = len(candidates) - len(kept_middle)

    rebuilt: list[Message] = []
    for group in protected_head:
        rebuilt.extend(group.messages)
    if dropped:
        rebuilt.append(
            user(
                f"[Note: {dropped} earlier tool exchange(s) were removed from this "
                f"conversation to stay within the context budget. If you need that "
                f"information again, call the tool again rather than guessing.]"
            )
        )
    for group in kept_middle + keep_tail:
        rebuilt.extend(group.messages)

    result = Compaction(
        messages=rebuilt,
        strategy="safe-trim",
        before=before,
        after=conversation_tokens(rebuilt),
        dropped_groups=dropped,
    )
    issues = validate(rebuilt)
    result.notes.append("valid" if not issues else f"INVALID: {issues[0]}")

    # Trimming has a floor, and it is easy to set a budget below it. The system
    # prompt, the task, and the most recent exchanges are all protected, so if
    # they alone exceed the budget no amount of dropping will reach it.
    #
    # Worth reporting rather than silently returning something too big: the fix is
    # a shorter system prompt, fewer protected exchanges, or summarisation -- not
    # more trimming.
    if result.after > budget:
        protected_cost = sum(g.tokens for g in protected_head)
        result.notes.append(
            f"FLOOR: cannot reach {budget} tokens. Protected content (system prompt "
            f"+ task = {protected_cost}, plus {len(keep_tail)} recent exchange(s)) "
            f"already costs {result.after}. Shorten the system prompt, protect fewer "
            f"recent exchanges, or summarise instead."
        )
    return result


def compress_tool_results(messages: list[Message], max_chars: int = 600) -> Compaction:
    """Shorten oversized tool results in place.

    Usually the highest-value change you can make, and the cheapest. In the
    lesson 3 growth table the +2,339 and +2,656 token jumps were both single
    `read_file` results. Nothing else came close.

    Why it works: a tool result is often 90% padding for the model's purposes.
    A file listing, a search dump, a JSON blob -- the model needed one fact from
    it, but the whole thing now sits in the conversation forever, re-sent on every
    subsequent call.

    Note that we keep the head *and* the tail. The head carries the structure
    (headers, first rows) and the tail often carries the conclusion. Cutting only
    the end loses whichever mattered, and the marker in the middle tells the model
    the result was abridged so it can re-read if it must.
    """
    before = conversation_tokens(messages)
    out: list[Message] = []
    compressed = 0

    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str) and len(content) > max_chars:
            head = content[: int(max_chars * 0.6)]
            tail = content[-int(max_chars * 0.25) :]
            removed = len(content) - len(head) - len(tail)
            new = (
                f"{head}\n"
                f"... [{removed} characters removed to save context. "
                f"Call the tool again with a narrower range if you need the rest.] ...\n"
                f"{tail}"
            )
            copy = dict(message)
            copy["content"] = new
            out.append(copy)
            compressed += 1
        else:
            out.append(message)

    return Compaction(
        messages=out,
        strategy="compress-tool-results",
        before=before,
        after=conversation_tokens(out),
        compressed=compressed,
    )


SUMMARY_MARKER = "[Summary of earlier work in this conversation]"


def summarise_history(
    client: LLMClient,
    messages: list[Message],
    keep_recent_exchanges: int = 2,
    # 1200, not 400. The first version of this function used 400 and reliably came
    # back EMPTY against gpt-oss: a reasoning model spends its output budget on
    # hidden deliberation first, so a tight cap yields no visible summary at all.
    # Lesson 0 documented that trap and this function still walked into it, which
    # is a good illustration of how it hides -- the call succeeds, tokens are
    # billed, and you get an empty string.
    max_summary_tokens: int = 1200,
    fallback_budget: int | None = None,
) -> Compaction:
    """Replace old exchanges with a model-written summary.

    Strictly more expensive than trimming -- it costs an extra model call -- and
    the reason to pay is that trimming forgets while summarising remembers the
    gist. For a long research task, "I already checked files A, B and C and they
    did not mention X" is worth keeping even when the full text is not.

    Two things to be clear-eyed about:

      * **Summarising is lossy and the loss is not uniform.** Exact values, ids and
        quotes are exactly what a summary drops and exactly what an agent later
        needs. Mitigate by asking for facts and identifiers explicitly, as the
        prompt below does.
      * **Summaries compound.** Summarise a conversation that already contains a
        summary and you are summarising a summary. Detail decays geometrically. So
        we fold the previous summary into the new one rather than stacking them.
    """
    before = conversation_tokens(messages)
    groups = group_messages(messages)

    protected = [g for g in groups if not g.droppable]
    body = [g for g in groups if g.droppable]
    keep_tail = body[-keep_recent_exchanges:] if keep_recent_exchanges else []
    to_summarise = body[: len(body) - len(keep_tail)]

    if not to_summarise:
        return Compaction(
            messages=list(messages),
            strategy="summarise",
            before=before,
            after=before,
            notes=["nothing old enough to summarise"],
        )

    transcript_parts: list[str] = []
    for group in to_summarise:
        for message in group.messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                calls = ", ".join(
                    f"{c['function']['name']}({c['function']['arguments']})"
                    for c in message["tool_calls"]
                )
                transcript_parts.append(f"ASSISTANT called: {calls}")
            elif role == "tool":
                transcript_parts.append(f"TOOL RESULT: {message.get('content')}")
            elif message.get("content"):
                transcript_parts.append(f"{str(role).upper()}: {message['content']}")

    transcript = "\n".join(transcript_parts)

    summary_reply = client.chat(
        [
            system(
                "You compress an AI agent's working notes so it can continue with "
                "less context. Preserve: every concrete fact discovered, exact "
                "values, file paths, identifiers, and which tools were already "
                "tried with what outcome. Discard: narration, restated reasoning, "
                "and raw text that has already been reduced to a fact. Write terse "
                "bullet points, not prose. Never invent anything not present."
            ),
            user(f"Compress these agent working notes:\n\n{transcript}"),
        ],
        max_tokens=max_summary_tokens,
    )

    summary_text = (summary_reply.text or "").strip()
    if not summary_text:
        # Reasoning models can still spend the whole budget thinking and return
        # nothing. Falling back to a trim beats sending an empty summary and
        # silently losing the history.
        #
        # The fallback budget must be the caller's real budget. An earlier version
        # hardcoded 10,000 here, which was above the conversation size, so the
        # fallback dropped nothing and "saved" zero tokens while reporting success.
        # A fallback that silently does nothing is worse than no fallback.
        budget = fallback_budget if fallback_budget is not None else int(before * 0.6)
        fallback = trim_safe(
            messages, budget=budget, keep_recent_exchanges=keep_recent_exchanges
        )
        fallback.strategy = "summarise->trim fallback"
        fallback.notes.append(
            f"summary came back empty ({summary_reply.usage.reasoning_tokens} reasoning "
            f"tokens, finish_reason={summary_reply.finish_reason}); trimmed to "
            f"{budget} instead"
        )
        return fallback

    rebuilt: list[Message] = []
    for group in protected:
        rebuilt.extend(group.messages)
    rebuilt.append(user(f"{SUMMARY_MARKER}\n{summary_text}"))
    for group in keep_tail:
        rebuilt.extend(group.messages)

    result = Compaction(
        messages=rebuilt,
        strategy="summarise",
        before=before,
        after=conversation_tokens(rebuilt),
        dropped_groups=len(to_summarise),
        summarised=1,
    )
    result.notes.append(
        f"summary cost {summary_reply.usage.total_tokens} tokens to produce "
        f"(an extra model call)"
    )
    issues = validate(rebuilt)
    result.notes.append("valid" if not issues else f"INVALID: {issues[0]}")
    return result


# ---------------------------------------------------------------------------
# 4. The compactor: what the loop actually calls
# ---------------------------------------------------------------------------
@dataclass
class ContextManager:
    """Decides, before each model call, whether and how to shrink the history.

    Designed to be passed straight to lesson 3's `run_agent(compactor=...)`.

    The policy below is ordered cheapest-first, which is the right instinct:

      1. Under budget? Do nothing. Compaction is not free.
      2. Over? Compress oversized tool results. No model call, big win, keeps
         every step's structure intact.
      3. Still over? Summarise or trim the oldest exchanges.

    `budget` is the input-token allowance for the conversation, not the model's
    context window. Leave room for the tool schemas and the reply.
    """

    budget: int = 3000
    strategy: str = "auto"  # auto | compress | trim | summarise | none
    client: LLMClient | None = None
    keep_recent_exchanges: int = 2
    tool_result_max_chars: int = 600
    verbose: bool = False
    history: list[Compaction] = field(default_factory=list)

    def __call__(self, messages: list[Message], step: int) -> list[Message]:
        current = conversation_tokens(messages)

        if self.strategy == "none" or current <= self.budget:
            return messages

        working = messages
        applied: list[Compaction] = []

        if self.strategy in ("auto", "compress"):
            step_result = compress_tool_results(working, self.tool_result_max_chars)
            if step_result.compressed:
                applied.append(step_result)
                working = step_result.messages

        still_over = conversation_tokens(working) > self.budget

        if still_over and self.strategy in ("auto", "summarise") and self.client is not None:
            step_result = summarise_history(
                self.client,
                working,
                self.keep_recent_exchanges,
                fallback_budget=self.budget,
            )
            applied.append(step_result)
            working = step_result.messages
        elif still_over and self.strategy in ("auto", "trim", "summarise"):
            step_result = trim_safe(working, self.budget, self.keep_recent_exchanges)
            applied.append(step_result)
            working = step_result.messages

        for item in applied:
            item.notes.insert(0, f"step {step}")
            self.history.append(item)
            if self.verbose:
                print(
                    f"    [context] {item.strategy}: {item.before} -> {item.after} "
                    f"tokens ({item.saved} saved)"
                )

        return working

    @property
    def total_saved(self) -> int:
        return sum(c.saved for c in self.history)
