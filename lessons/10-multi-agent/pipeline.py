"""Handoff: a fixed sequence of agents, each continuing where the last stopped.

Delegation and handoff get used interchangeably and they are not the same thing.

**Delegation** (`team.py`) returns control. The parent asks, gets an answer, and
carries on deciding. The model chooses whether to delegate, so the sequence is
emergent and the parent stays responsible for the final answer.

**Handoff** (this file) passes control on. Stage 2 does not report back to stage 1;
it continues the work and hands to stage 3. The sequence is fixed in code, so no
model decides it — which makes it cheaper, more predictable, and unable to adapt
when a stage produces something unexpected.

The rule of thumb that falls out: **if the sequence is known in advance, a pipeline
beats a delegating agent**, because you are paying a model to make a decision you
have already made. Delegation earns its cost only when the route genuinely depends on
what is found along the way.

Two ways to pass state between stages, and the choice is the hard part:

`relay` sends only the previous stage's *output*. Cheap, and lossy in a specific way:
stage 3 cannot see what stage 1 actually read, so it cannot tell a quotation from a
paraphrase.

`shared` continues the whole message list, so a later stage sees every tool result
directly. Complete, and it pays lesson 3's quadratic growth across the entire
pipeline rather than within one agent.

Both are implemented and `--compare` measures them, because the interesting question
is not which is architecturally nicer but what the difference costs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llmkit import LLMClient, Message, ToolRegistry, Usage, system

from team import CRITIC, RESEARCHER, SubAgent


# ---------------------------------------------------------------------------
WRITER = SubAgent(
    name="writer",
    description="Turns findings into a clear answer. Has no tools.",
    system_prompt=(
        "You write a clear, direct answer from findings you are given.\n"
        "Use only what is in front of you. If the findings do not cover part of the "
        "question, say so rather than filling the gap — you have no way to check, and "
        "a plausible invention is worse than an admitted gap.\n"
        "Two short paragraphs at most. No preamble."
    ),
    tools=[],
    max_steps=2,
)

REVISER = SubAgent(
    name="reviser",
    description="Applies a critic's specific corrections to a draft. Has no tools.",
    system_prompt=(
        "You revise a draft using a list of problems found by a reviewer.\n"
        "Fix exactly what was raised and change nothing else. If a problem cannot be "
        "fixed from the material you have — for instance it asks for a fact nobody "
        "established — remove the unsupported claim rather than rewording it.\n"
        "Return only the revised answer."
    ),
    tools=[],
    max_steps=2,
)


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline: who runs, and what they are told to do.

    `instruction` is a template over `{topic}` and `{previous}`. Keeping it here
    rather than inside the `SubAgent` matters: the same researcher should be reusable
    in a different pipeline with a different brief, and a sub-agent that hardcodes its
    place in one sequence is not a specialist, it is a step.
    """

    agent: SubAgent
    #: Template over `{topic}`, `{previous}`, and any earlier stage's name — so
    #: `{writer}` interpolates what the writer produced. That addressing exists
    #: because `{previous}` alone is not enough: the reviser needs both the critique
    #: (the previous stage) *and* the draft (two stages back), and in `relay` mode
    #: nothing carries the draft forward unless you ask for it by name. Which is the
    #: lossiness of relay mode, made concrete: every fact a later stage needs has to
    #: be plumbed explicitly, and the failure mode is a stage quietly working from
    #: less than you assumed.
    instruction: str
    #: When False, a failure here does not stop the pipeline. Used for the critic:
    #: losing a review is worse than having one, but it is not a reason to throw away
    #: a finished draft.
    required: bool = True


@dataclass
class StageResult:
    agent: str
    output: str | None
    stop_reason: str
    steps: int
    usage: Usage
    duration_s: float
    ok: bool
    #: Messages handed in. In `shared` mode this is the growth you are paying for.
    messages_in: int = 0
    note: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.usage.prompt_tokens + self.usage.completion_tokens


@dataclass
class PipelineRun:
    topic: str
    mode: str
    stages: list[StageResult] = field(default_factory=list)
    final_answer: str | None = None
    stopped_early: bool = False

    @property
    def usage(self) -> Usage:
        total = Usage()
        for stage in self.stages:
            total = total + stage.usage
        return total

    @property
    def total_tokens(self) -> int:
        return self.usage.prompt_tokens + self.usage.completion_tokens

    @property
    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.stages)

    @property
    def failures(self) -> list[StageResult]:
        return [s for s in self.stages if not s.ok]

    @property
    def critic_approved(self) -> bool | None:
        """None when no critic ran, which is different from 'not approved'."""
        for stage in self.stages:
            if stage.agent == "ask_critic" and stage.output:
                return stage.output.strip().upper().startswith("APPROVED")
        return None


# ---------------------------------------------------------------------------
def handoff_messages(messages: list[Message], new_system_prompt: str) -> list[Message]:
    """Continue a conversation as a different agent.

    This exists because of a limitation worth knowing: **you cannot change a system
    prompt mid-conversation**. `run_agent(initial_messages=...)` resumes a message
    list as-is, so a naive handoff leaves the *previous* agent's instructions in
    place, and you get one agent wearing a second agent's name. The fix is to replace
    the system message, which is the only message you may rewrite retroactively
    without corrupting the record — every other message is something that actually
    happened.

    Note what survives: all the tool results. That is the point of `shared` mode, and
    also its cost.
    """
    body = [m for m in messages if m.get("role") != "system"]
    return [system(new_system_prompt), *body]


def run_pipeline(
    client: LLMClient,
    topic: str,
    base_registry: ToolRegistry,
    stages: list[Stage],
    *,
    mode: str = "relay",
    on_stage=None,
) -> PipelineRun:
    """Run each stage in order, threading state according to `mode`."""
    from loop import run_agent

    if mode not in ("relay", "shared"):
        raise ValueError(f"mode must be 'relay' or 'shared', got {mode!r}")

    run = PipelineRun(topic=topic, mode=mode)
    previous = ""
    carried: list[Message] = []
    #: Every stage's output, addressable by agent name. Defaults so a template
    #: referencing a stage that has not run yet (or failed) renders an honest
    #: placeholder instead of raising KeyError mid-pipeline.
    outputs: dict[str, str] = {s.agent.name: "(not available)" for s in stages}

    for stage in stages:
        sub = stage.agent
        instruction = stage.instruction.format(
            topic=topic, previous=previous or "(nothing yet)", **outputs
        )
        tool_names = [n for n in sub.tools if n in base_registry]
        registry = base_registry.subset(tool_names)

        initial = handoff_messages(carried, sub.system_prompt) if (mode == "shared" and carried) else None

        started = time.perf_counter()
        try:
            trajectory = run_agent(
                client,
                instruction,
                registry,
                max_steps=sub.max_steps,
                system_prompt=sub.system_prompt,
                max_tokens=sub.max_tokens,
                initial_messages=initial,
            )
        except Exception as exc:  # noqa: BLE001
            result = StageResult(
                agent=sub.name,
                output=None,
                stop_reason="error",
                steps=0,
                usage=Usage(),
                duration_s=time.perf_counter() - started,
                ok=False,
                messages_in=len(initial or []),
                note=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            run.stages.append(result)
            if on_stage:
                on_stage(result)
            if stage.required:
                run.stopped_early = True
                return run
            continue

        result = StageResult(
            agent=sub.name,
            output=trajectory.final_answer,
            stop_reason=trajectory.stop_reason.value,
            steps=len(trajectory.steps),
            usage=trajectory.usage,
            duration_s=time.perf_counter() - started,
            ok=trajectory.succeeded,
            messages_in=len(initial or []),
            note=trajectory.note,
        )
        run.stages.append(result)
        if on_stage:
            on_stage(result)

        if not result.ok and stage.required:
            # Same rule as delegation: an unfinished stage is not an output. Passing
            # its partial text on as though it were finished is how a pipeline
            # launders a failure into a confident final answer three stages later.
            run.stopped_early = True
            run.final_answer = None
            return run

        if result.ok:
            previous = result.output or ""
            outputs[sub.name] = previous
            carried = list(trajectory.messages)
            run.final_answer = result.output
        else:
            previous = f"(the {sub.name} stage did not finish; nothing usable from it)"
            outputs[sub.name] = previous

    return run


# ---------------------------------------------------------------------------
# The canonical pipeline
# ---------------------------------------------------------------------------
def research_write_critique(*, revise: bool = False) -> list[Stage]:
    """research -> write -> critique, optionally -> revise.

    The critic is `required=False` on purpose. Losing the review is a worse outcome
    than no review, but it is not a reason to discard a finished draft — and marking
    it required would mean a critic that ran out of steps destroyed the whole run.
    """
    stages = [
        # The same RESEARCHER and CRITIC the delegating agent uses, deliberately. A
        # difference between the two architectures should be a difference in wiring,
        # not in the quality of the specialists -- otherwise the comparison measures
        # prompt-writing effort rather than architecture.
        Stage(
            agent=RESEARCHER,
            instruction=(
                "Find out the following from this project's files: {topic}\n"
                "Report what the files actually say, quoting exact wording where it "
                "matters, and name the file you found it in."
            ),
        ),
        Stage(
            agent=WRITER,
            instruction=(
                "Question: {topic}\n\n"
                "Findings from the research agent:\n{previous}\n\n"
                "Write the answer using only these findings."
            ),
        ),
        Stage(
            agent=CRITIC,
            instruction="Question: {topic}\n\nDraft answer to review:\n{previous}",
            required=False,
        ),
    ]
    if revise:
        stages.append(
            Stage(
                agent=REVISER,
                # Addresses two earlier stages by name. `{previous}` would give it the
                # critique alone, leaving it revising a draft it cannot see.
                instruction=(
                    "Question: {topic}\n\n"
                    "Draft:\n{writer}\n\n"
                    "Reviewer's verdict:\n{ask_critic}\n\n"
                    "If the verdict begins with APPROVED, return the draft unchanged. "
                    "Otherwise apply the corrections."
                ),
                required=False,
            )
        )
    return stages
