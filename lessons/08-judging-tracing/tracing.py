"""Tracing and cost accounting: answering "why did this run do that?".

The useful realisation here is that **tracing is a view, not a collection
problem.** Lesson 3's `Trajectory` already records every step, every tool call,
every result, and per-call token counts and latency. Nothing new needs to be
instrumented; the data has been there since lesson 3.

That is not luck, it is what "build observability in from the start" buys you.
Lesson 3's notes said the trajectory would become what lesson 8 traces, and the
whole of `build_trace` below is a transformation of data already in hand. Had the
loop only returned a string, this lesson would begin by rewriting lesson 3.

On cost: the numbers are computed from real token counts and a price table that is
**illustrative and will be wrong by the time you read this**. Provider prices change
and the free tier is genuinely $0. The point is never the absolute figure. It is
the *shape*: which step dominates, how cost scales with steps, and cost per
**successful** answer -- because an agent that is cheap and wrong is not cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACES_DIR = Path(__file__).parent / "traces"


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
#: USD per 1,000,000 tokens, as (input, output).
#:
#: ILLUSTRATIVE ONLY. These were plausible published figures when this lesson was
#: written; they change, and Groq's free tier bills nothing at all. Treat every
#: dollar figure in this lesson as a relative comparison, not an invoice.
#:
#: Kept as an explicit, editable table rather than fetched at runtime: a hidden
#: price lookup that silently changes your cost report between runs is worse than a
#: number you know is stale.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "qwen/qwen3.8-27b": (0.15, 0.60),
    "qwen2.5:7b-instruct": (0.0, 0.0),   # local: electricity, not tokens
    "qwen2.5:3b-instruct": (0.0, 0.0),
}

#: Used when a model is not in the table, so a cost report never silently reads
#: zero for an unknown model -- which would look like "free" rather than "unknown".
FALLBACK_PRICE = (0.20, 0.80)


def price_for(model: str) -> tuple[tuple[float, float], bool]:
    """Return ((input, output) per 1M tokens, is_known)."""
    if model in PRICES:
        return PRICES[model], True
    for name, price in PRICES.items():
        if name.split(":")[0] in model or model in name:
            return price, True
    return FALLBACK_PRICE, False


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    (price_in, price_out), _ = price_for(model)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------
@dataclass
class Span:
    """One timed unit of work: a model call, or a tool execution.

    `parent` makes the trace a tree rather than a list, which is what lets you see
    that three tool calls belonged to one model turn. A flat log of events loses
    exactly the structure you need when debugging.
    """

    name: str
    kind: str                      # "step" | "model_call" | "tool_call"
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    ok: bool = True
    detail: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, model: str) -> float:
        return cost_usd(model, self.prompt_tokens, self.completion_tokens)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "duration_s": round(self.duration_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "ok": self.ok,
            "detail": self.detail,
            "attributes": self.attributes,
            "children": [c.to_json() for c in self.children],
        }


@dataclass
class Trace:
    """A complete, inspectable record of one agent run."""

    question: str
    model: str
    stop_reason: str
    final_answer: str | None
    spans: list[Span] = field(default_factory=list)
    created_at: str = ""

    # -- rollups --------------------------------------------------------
    def _walk(self):
        stack = list(self.spans)
        while stack:
            span = stack.pop()
            yield span
            stack.extend(span.children)

    # Rollups count ONLY model_call spans, and that is load-bearing.
    #
    # A step span carries the same token counts as its model_call child, so that the
    # tree can show a per-step total. Summing over every span therefore counted each
    # step twice and reported exactly double -- caught by a test asserting the rollup
    # equalled the sum of the step spans.
    #
    # It is worth dwelling on how this would have failed in the wild: nothing
    # crashes, no verdict changes, and every cost figure is 2x. A doubled cost report
    # looks entirely plausible, which is why the arithmetic in a measurement tool
    # deserves a test even when it is "obviously" right.
    #
    # Filtering on kind is also the semantically correct rule: only model calls
    # consume tokens. Tool executions consume none, and step spans are containers.
    def _token_spans(self):
        return (s for s in self._walk() if s.kind == "model_call")

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self._token_spans())

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self._token_spans())

    @property
    def reasoning_tokens(self) -> int:
        return sum(s.reasoning_tokens for s in self._token_spans())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_cost(self) -> float:
        return cost_usd(self.model, self.prompt_tokens, self.completion_tokens)

    @property
    def model_time_s(self) -> float:
        return sum(s.duration_s for s in self._walk() if s.kind == "model_call")

    @property
    def tool_time_s(self) -> float:
        return sum(s.duration_s for s in self._walk() if s.kind == "tool_call")

    @property
    def failed_spans(self) -> list[Span]:
        return [s for s in self._walk() if not s.ok]

    def save(self, name: str) -> Path:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACES_DIR / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "question": self.question,
                    "model": self.model,
                    "stop_reason": self.stop_reason,
                    "final_answer": self.final_answer,
                    "created_at": self.created_at or _now(),
                    "spans": [s.to_json() for s in self.spans],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Building a trace from a trajectory
# ---------------------------------------------------------------------------
def build_trace(trajectory, model: str) -> Trace:
    """Turn lesson 3's Trajectory into a span tree.

    Every number here was already recorded. The value added is *structure*: one
    span per step, with the model call and its tool executions nested underneath,
    so the tree shows which turn triggered which work.
    """
    trace = Trace(
        question=trajectory.question,
        model=model,
        stop_reason=trajectory.stop_reason.value,
        final_answer=trajectory.final_answer,
        created_at=_now(),
    )

    for step in trajectory.steps:
        response = step.response
        step_span = Span(
            name=f"step {step.index}",
            kind="step",
            attributes={"tools_requested": [c.name for c in response.tool_calls]},
        )

        model_span = Span(
            name="model call",
            kind="model_call",
            duration_s=response.usage.latency_s,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            detail=(
                f"requested {len(response.tool_calls)} tool(s)"
                if response.tool_calls
                else f"returned text ({response.finish_reason})"
            ),
            attributes={"finish_reason": response.finish_reason},
        )
        step_span.children.append(model_span)

        for execution in step.executions:
            step_span.children.append(
                Span(
                    name=execution.name,
                    kind="tool_call",
                    duration_s=execution.duration_s,
                    ok=execution.ok,
                    detail=(
                        execution.result[:120].replace("\n", " ")
                        if execution.ok
                        else f"{execution.failure_kind}: {execution.result[:100]}"
                    ),
                    attributes={"arguments": execution.arguments},
                )
            )

        step_span.duration_s = sum(c.duration_s for c in step_span.children)
        step_span.prompt_tokens = model_span.prompt_tokens
        step_span.completion_tokens = model_span.completion_tokens
        step_span.reasoning_tokens = model_span.reasoning_tokens
        step_span.ok = all(c.ok for c in step_span.children)
        trace.spans.append(step_span)

    return trace


# ---------------------------------------------------------------------------
# Cost analysis
# ---------------------------------------------------------------------------
@dataclass
class CostReport:
    """Cost framed the way it should be read.

    `cost_per_success` is the number that matters and the one nobody reports. An
    agent that is half the price and fails twice as often costs *more* per answer
    you can actually use, and a per-run figure hides that completely.
    """

    model: str
    price_known: bool
    runs: int
    successes: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int

    @property
    def total_cost(self) -> float:
        return cost_usd(self.model, self.prompt_tokens, self.completion_tokens)

    @property
    def cost_per_run(self) -> float:
        return self.total_cost / self.runs if self.runs else 0.0

    @property
    def cost_per_success(self) -> float:
        """Infinite when nothing succeeded, which is the honest answer."""
        return self.total_cost / self.successes if self.successes else float("inf")

    @property
    def input_share(self) -> float:
        return self.prompt_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def reasoning_share(self) -> float:
        return (
            self.reasoning_tokens / self.completion_tokens if self.completion_tokens else 0.0
        )

    @property
    def wasted_cost(self) -> float:
        """What the failures cost. Usually the most persuasive number in a review."""
        failures = self.runs - self.successes
        return self.cost_per_run * failures


def cost_from_eval_run(run) -> CostReport:
    """Build a cost report from a lesson 7 EvalRun.

    Reuses the per-case token counts lesson 7 already records, which is the second
    time in this lesson that earlier bookkeeping removes the need for new machinery.
    """
    (_, _), known = price_for(run.model)
    return CostReport(
        model=run.model,
        price_known=known,
        runs=run.total,
        successes=run.passed,
        prompt_tokens=sum(r.prompt_tokens for r in run.results),
        completion_tokens=sum(r.completion_tokens for r in run.results),
        # EvalRun does not record reasoning tokens separately; traces do.
        reasoning_tokens=0,
    )
