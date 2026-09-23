"""Sub-agents as tools. The whole lesson is in one idea.

A sub-agent is a tool whose implementation happens to be another agent loop.

That is not a simplification for teaching purposes — it is the design. `run_agent`
needs no changes, the wire protocol needs no changes, and lesson 2's dispatcher is
already the security boundary. The parent model sees a tool called `ask_researcher`
with a description and a `task` parameter, exactly like `convert_currency`, and
cannot tell the difference. Every "multi-agent framework" you will read about is
this plus naming.

Which means the interesting content is not *how* to delegate. It is the four things
that go wrong once you do:

**A sub-agent's failure looks exactly like its answer.** Both are a string returned
from a tool. Lesson 6's finding — `COMPLETED` means "the model stopped asking for
tools", not "the answer is correct" — now compounds: a sub-agent that gives up
politely returns prose, the tool returns that prose, and the parent builds on it as
though it were research. Every result here is therefore prefixed with its provenance,
and every non-completion is returned as an explicit error.

**Cost disappears.** A sub-agent's tokens are spent *inside a tool execution*, where
the parent's `Trajectory.usage` cannot see them. So lesson 7's per-case token counts
and lesson 8's `CostReport` both silently under-report a delegating agent — by most of
its actual cost. That is the fifth silent measurement bug this project has hit, and
this time it was predicted rather than discovered, which is why `DelegationLog`
exists.

**Recursion is free and unbounded.** Give a sub-agent the delegation tools and it can
delegate to itself. `DelegationBudget` caps depth and total calls, defaulting to a
depth of 1: sub-agents get no delegation tools unless you explicitly ask for them.

**Least privilege stops being optional.** `ToolRegistry.subset` was written in lesson 2
and never used until now. A writer agent that can read files will read files; a critic
that can run the calculator will do arithmetic instead of criticising.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llmkit import LLMClient, ToolRegistry, ToolSpec, Usage
from llmkit.tools import ToolError


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubAgent:
    """One specialist, described twice.

    The two descriptions do different jobs and confusing them is the most common
    multi-agent mistake:

    `description` is read by the **parent model**, to decide whether to delegate. It
    should describe a capability and its limits, like any tool description.

    `system_prompt` is read by the **sub-agent**, to decide how to behave. It should
    describe a job, not a capability.

    Writing one and reusing it for both produces an agent that is either
    under-delegated to (the parent cannot tell what it is for) or badly behaved (the
    sub-agent has no instructions of its own).
    """

    name: str
    description: str
    system_prompt: str
    #: Names of tools this sub-agent may use, drawn from the parent registry.
    #: Deliberately a whitelist rather than a blacklist: a new tool added to the
    #: project should reach a specialist only when someone decides it should.
    tools: list[str]
    max_steps: int = 5
    #: Tokens per model call inside the sub-agent. Kept at lesson 0's floor because a
    #: starved reasoning model returns an empty string, and an empty sub-agent answer
    #: is indistinguishable from a sub-agent that had nothing to say.
    max_tokens: int = 1024


# ---------------------------------------------------------------------------
@dataclass
class DelegationBudget:
    """Caps on delegation, shared by reference across a whole run.

    Two separate limits, because they stop different failures:

    `max_depth` stops recursion. A sub-agent holding delegation tools can call itself,
    or call a peer that calls back — and unlike lesson 3's step cap, nothing about the
    parent loop notices, because from the parent's view it is still one tool call that
    simply takes a while. The default of 1 means sub-agents get no delegation tools at
    all, which is the right default: nested delegation is a thing you should have to
    ask for.

    `max_calls` stops breadth. A parent can call six specialists in one step and then
    six more next step, each costing a full agent run. Depth alone would not catch it.
    """

    max_depth: int = 1
    max_calls: int = 6
    depth: int = 0
    calls: int = 0

    def child(self) -> DelegationBudget:
        """A budget for one level down, sharing nothing but the counters it must."""
        return DelegationBudget(
            max_depth=self.max_depth,
            max_calls=self.max_calls,
            depth=self.depth + 1,
            calls=self.calls,
        )

    @property
    def exhausted_depth(self) -> bool:
        return self.depth >= self.max_depth

    @property
    def exhausted_calls(self) -> bool:
        return self.calls >= self.max_calls


# ---------------------------------------------------------------------------
@dataclass
class Delegation:
    """What one sub-agent call cost and whether it worked."""

    agent: str
    task: str
    stop_reason: str
    steps: int
    tool_sequence: list[str]
    usage: Usage
    duration_s: float
    answer: str | None
    ok: bool
    note: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.usage.prompt_tokens + self.usage.completion_tokens


@dataclass
class DelegationLog:
    """Accounting for work done inside tool calls, where nothing else can see it.

    This is the piece that stops a delegating agent from lying about its cost. A
    sub-agent's model calls happen while `registry.dispatch()` is executing, which is
    a place lesson 3's `Trajectory` does not look and was never designed to. So:

        trajectory.usage           -> the parent's tokens only
        trajectory.usage + log.usage -> what the run actually cost

    The gap is not a rounding error. In this lesson's pipeline the parent accounts for
    well under half the total, so a cost report built on `Trajectory.usage` alone is
    wrong by more than it is right. Every cost figure in lessons 7 and 8 has this
    blind spot the moment a sub-agent is involved, which is why `--cost` in this
    lesson's CLI reports both numbers side by side rather than one.
    """

    delegations: list[Delegation] = field(default_factory=list)

    def record(self, delegation: Delegation) -> None:
        self.delegations.append(delegation)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for delegation in self.delegations:
            total = total + delegation.usage
        return total

    @property
    def total_tokens(self) -> int:
        return self.usage.prompt_tokens + self.usage.completion_tokens

    @property
    def calls(self) -> int:
        return len(self.delegations)

    @property
    def failures(self) -> list[Delegation]:
        return [d for d in self.delegations if not d.ok]

    @property
    def agents_used(self) -> list[str]:
        return [d.agent for d in self.delegations]

    def hidden_share(self, parent_tokens: int) -> float:
        """Fraction of the run's tokens that the parent's own accounting misses."""
        total = parent_tokens + self.total_tokens
        return self.total_tokens / total if total else 0.0

    def effective_tool_sequence(self, parent_sequence: list[str]) -> list[str]:
        """The parent's tool sequence with delegations expanded into real tool use.

        This exists because of the sharpest finding in the lesson, and it was nearly
        missed. A probe reported that `arith_precision` failed under delegation while
        the answer was visibly correct -- "0.285714 (to six decimal places)". The
        failing check was not the answer at all, it was `used_tools(["calculate"])`:
        the coordinator's `tool_sequence` reads `["ask_calculator"]`, because the
        calculator ran one level down where `Trajectory` cannot see it.

        So the blind spot that hides a sub-agent's *tokens* also hides its *tool use*,
        and every `used_tools` assertion in lesson 7's dataset -- 10 of 16 cases, plus
        `answered_without_tools` on an eleventh -- silently measures the wrong thing
        the moment work is delegated. A full suite run would have reported a large
        regression that was mostly an artifact of the instrument.

        Expanding the sequence is the fix, and it is the multi-agent equivalent of
        lesson 8's rule that a trace is a view over data you already recorded. Nothing
        new had to be captured; the tool names were in the log the whole time.
        """
        # One cursor per agent so repeated delegations to the same specialist expand in
        # the order they happened, rather than all resolving to the first call.
        cursors: dict[str, int] = {}
        by_agent: dict[str, list[Delegation]] = {}
        for delegation in self.delegations:
            by_agent.setdefault(delegation.agent, []).append(delegation)

        out: list[str] = []
        for name in parent_sequence:
            calls = by_agent.get(name)
            if not calls:
                out.append(name)
                continue
            index = cursors.get(name, 0)
            if index >= len(calls):
                # More parent entries than recorded delegations: the call failed before
                # anything was logged. Keep the name rather than dropping it silently.
                out.append(name)
                continue
            cursors[name] = index + 1
            out.extend(calls[index].tool_sequence)
        return out

    def as_spans(self) -> list:
        """Lesson 8 spans, so a delegating run is still one inspectable tree.

        `Span` already supports children, so nesting a sub-agent under the tool call
        that invoked it needed no new machinery — which is the payoff for building
        tracing as a view over recorded data rather than as a collection mechanism.
        """
        from tracing import Span

        return [
            Span(
                name=f"{d.agent} (sub-agent)",
                kind="model_call",
                duration_s=d.duration_s,
                prompt_tokens=d.usage.prompt_tokens,
                completion_tokens=d.usage.completion_tokens,
                reasoning_tokens=d.usage.reasoning_tokens,
                ok=d.ok,
                detail=f"{d.stop_reason}, {d.steps} step(s), tools: "
                f"{'->'.join(d.tool_sequence) or 'none'}",
                attributes={"task": d.task[:160], "agent": d.agent},
            )
            for d in self.delegations
        ]


# ---------------------------------------------------------------------------
# Turning a sub-agent into a tool
# ---------------------------------------------------------------------------
def _parameters() -> dict:
    return {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The task for this agent, stated completely. It cannot see your "
                    "conversation, so include every fact it needs."
                ),
            },
        },
        "required": ["task"],
    }


def as_tool(
    sub: SubAgent,
    client: LLMClient,
    base_registry: ToolRegistry,
    log: DelegationLog,
    budget: DelegationBudget,
) -> tuple[ToolSpec, object]:
    """Wrap a sub-agent so the parent can call it like any other tool.

    Note what the parameters do *not* include: any way to pass the parent's
    conversation. That is deliberate and it is the central trade-off of delegation.
    The sub-agent starts fresh, which is why delegation is affordable at all — but it
    means the parent must restate everything relevant, and a parent that restates
    badly produces a sub-agent that answers the wrong question. Lesson 4's whole
    context problem, reappearing as an interface design decision.
    """
    spec = ToolSpec(
        name=sub.name,
        description=sub.description,
        parameters=_parameters(),
    )

    def call(task: str) -> str:
        from loop import run_agent

        if budget.exhausted_calls:
            # A ToolError rather than an exception, so the parent sees it as an
            # observation and can answer with what it already has. Lesson 2's rule.
            raise ToolError(
                f"Delegation budget exhausted ({budget.max_calls} sub-agent calls "
                f"used). Answer with what you already have."
            )
        budget.calls += 1

        child_budget = budget.child()
        tool_names = [name for name in sub.tools if name in base_registry]
        sub_registry = base_registry.subset(tool_names)

        started = time.perf_counter()
        try:
            trajectory = run_agent(
                client,
                task,
                sub_registry,
                max_steps=sub.max_steps,
                system_prompt=sub.system_prompt,
                max_tokens=sub.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            log.record(
                Delegation(
                    agent=sub.name,
                    task=task,
                    stop_reason="error",
                    steps=0,
                    tool_sequence=[],
                    usage=Usage(),
                    duration_s=elapsed,
                    answer=None,
                    ok=False,
                    note=f"{type(exc).__name__}: {str(exc)[:160]}",
                )
            )
            raise ToolError(
                f"The {sub.name} agent failed to run ({type(exc).__name__}). "
                f"Do not treat this as a finding; either try a different approach or "
                f"say plainly that you could not get this information."
            ) from exc

        elapsed = time.perf_counter() - started
        succeeded = trajectory.succeeded
        log.record(
            Delegation(
                agent=sub.name,
                task=task,
                stop_reason=trajectory.stop_reason.value,
                steps=len(trajectory.steps),
                tool_sequence=trajectory.tool_sequence,
                usage=trajectory.usage,
                duration_s=elapsed,
                answer=trajectory.final_answer,
                ok=succeeded,
                note=trajectory.note,
            )
        )
        _ = child_budget  # depth is enforced when building the registry, not here

        if not succeeded:
            # THE important branch in this file. A sub-agent that ran out of steps
            # still produces prose, and returning that prose as a successful tool
            # result is how a parent ends up confidently reporting a guess as a
            # finding. The failure has to arrive as a failure.
            partial = (trajectory.final_answer or "").strip()
            raise ToolError(
                f"The {sub.name} agent did not finish "
                f"(stop_reason={trajectory.stop_reason.value} after "
                f"{len(trajectory.steps)} step(s))."
                + (
                    f' Its unfinished output was: "{partial[:300]}" — this is '
                    f"UNVERIFIED and may be incomplete or wrong."
                    if partial
                    else " It produced no output."
                )
                + " Do not present it as fact."
            )

        # Provenance on every result, not just failures. The parent is a model, and a
        # bare string carries no hint that it came from a different agent with
        # different tools and no view of this conversation.
        return (
            f"[from {sub.name}: completed in {len(trajectory.steps)} step(s), "
            f"tools used: {' -> '.join(trajectory.tool_sequence) or 'none'}]\n"
            f"{trajectory.final_answer}"
        )

    return spec, call


def build_team_registry(
    base_registry: ToolRegistry,
    sub_agents: list[SubAgent],
    client: LLMClient,
    log: DelegationLog,
    budget: DelegationBudget | None = None,
    *,
    keep_own_tools: list[str] | None = None,
) -> ToolRegistry:
    """A registry where some tools are other agents.

    `keep_own_tools` decides what the *coordinator* can still do itself. Passing an
    empty list produces a pure router, which sounds clean and measures badly: the
    coordinator then has to delegate a two-digit multiplication, paying a whole agent
    run for it. Leaving it as None keeps every base tool, so delegation is a choice
    the model makes rather than a constraint you imposed — which is also the only
    version the eval can fairly compare against the solo agent.
    """
    budget = budget or DelegationBudget()
    if budget.exhausted_depth:
        # Depth is enforced here rather than at call time, by simply not offering the
        # delegation tools. A capability the model cannot see is a capability it cannot
        # be talked into using -- strictly better than refusing the call afterwards.
        return base_registry

    kept = base_registry.names if keep_own_tools is None else keep_own_tools
    registry = base_registry.subset([n for n in kept if n in base_registry])
    for sub in sub_agents:
        spec, fn = as_tool(sub, client, base_registry, log, budget)
        registry.add(spec, fn)
    return registry


# ---------------------------------------------------------------------------
# The team
# ---------------------------------------------------------------------------
RESEARCHER = SubAgent(
    name="ask_researcher",
    description=(
        "Delegate a lookup to a research agent that can search and read this "
        "project's files. Give it one specific question. It returns what it found, "
        "or an error if it could not find it. It cannot do arithmetic and cannot see "
        "your conversation, so state the question completely."
    ),
    system_prompt=(
        "You are a research agent. You find information in this project's files and "
        "report it accurately.\n"
        "Search, then read the most promising file, then answer. Quote exact wording "
        "when the question asks what something says.\n"
        "If you cannot find it, say so plainly and name where you looked. Never "
        "guess, and never fill a gap with what you think is probably true."
    ),
    tools=["list_files", "read_file", "search_files"],
    max_steps=5,
)

CALCULATOR = SubAgent(
    name="ask_calculator",
    description=(
        "Delegate a numeric task to an arithmetic agent with a calculator and a "
        "currency converter. Give it the full expression or conversion in words. It "
        "returns the number, or an error. It cannot read files."
    ),
    system_prompt=(
        "You are an arithmetic agent. You compute exact values using your tools and "
        "never in your head.\n"
        "Show the final number clearly. If a conversion is not supported, call the "
        "tool anyway, read the error, and report exactly what it said.\n"
        "Answer with the number and nothing else you were not asked for."
    ),
    tools=["calculate", "convert_currency", "get_current_time"],
    max_steps=5,
)

CRITIC = SubAgent(
    name="ask_critic",
    description=(
        "Have a draft answer checked before you give it. Pass the question and your "
        "draft. Returns either an approval or a specific list of problems. It has no "
        "tools, so it can only judge what you send it."
    ),
    system_prompt=(
        "You review a draft answer to a question and decide whether it is safe to "
        "send.\n"
        "Check three things: does it actually answer the question asked; does it "
        "state anything as fact that was not established; and does it hedge so much "
        "that it says nothing.\n"
        "If it is fine, reply exactly 'APPROVED' and one sentence saying why. "
        "Otherwise list the specific problems. Do not rewrite the answer."
    ),
    tools=[],
    max_steps=2,
)

TEAM = [RESEARCHER, CALCULATOR, CRITIC]

COORDINATOR_PROMPT = (
    "You are a coordinator with your own tools and a team of specialist agents.\n"
    "Prefer doing simple work yourself. Delegate when a task needs several steps of "
    "searching and reading, or when you want a draft checked.\n"
    "A specialist cannot see this conversation, so state its task completely, "
    "including any numbers or names it needs.\n"
    "A specialist's answer arrives labelled with where it came from. If one returns "
    "an error or says it could not finish, that is not a finding: do not present its "
    "guess as fact. Say plainly what you could not establish.\n"
    "When you have what you need, answer concisely."
)
