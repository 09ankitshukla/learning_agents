"""The agent loop. This file is the whole lesson.

Lesson 2 ended stuck. Its assistant handled exactly one round of tool calls, so a
question needing three tools in sequence -- get the time, convert a currency, then
take a percentage of the converted amount -- could not be answered. The model
cannot ask for the percentage until it has seen the conversion result.

The fix is not a better prompt and not a bigger model. It is a `while` loop:

    while not done:
        reply = model(messages)          # think
        if reply has no tool calls:      # the stopping condition
            return reply.text
        for call in reply.tool_calls:    # act
            messages.append(result)      # observe

That is it. That is an agent. Every framework you will ever use is this loop plus
conveniences. Once you have read this file you will be able to look at any agent
framework and identify which part is this loop and which part is decoration.

The loop is twenty lines. The other 200 in this file are the things that keep it
from hurting you:

  * a step cap, because an unbounded loop calling a paid API is a way to lose money
  * a stop reason, because "finished" and "gave up" must be distinguishable
  * stall detection, because models get stuck repeating themselves
  * a trajectory record, because you cannot debug what you did not observe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from llmkit import (
    Execution,
    LLMClient,
    LLMResponse,
    Message,
    ToolRegistry,
    Usage,
    system,
    tool_result,
    user,
)
from llmkit.openai_compat import PhantomToolCall


class StopReason(str, Enum):
    """Why the loop ended.

    Returning this rather than just an answer string is the single most important
    API decision in this file. "The agent produced text" and "the agent ran out of
    steps and gave up" are completely different outcomes, and a caller that cannot
    distinguish them will happily show a user a half-finished answer as though it
    were complete.

    Only COMPLETED means the model decided it was done.
    """

    COMPLETED = "completed"          # model returned prose: success
    MAX_STEPS = "max_steps"          # hit the cap: incomplete
    STALLED = "stalled"              # repeating itself: incomplete
    NO_ANSWER = "no_answer"          # stopped without text (e.g. token starvation)
    PHANTOM_TOOL = "phantom_tool"    # asked for a tool that was never offered
    ERROR = "error"                  # something unexpected broke

    @property
    def succeeded(self) -> bool:
        return self is StopReason.COMPLETED


@dataclass
class Step:
    """One iteration: one model call, plus any tools it triggered."""

    index: int
    response: LLMResponse
    executions: list[Execution] = field(default_factory=list)

    @property
    def usage(self) -> Usage:
        return self.response.usage

    @property
    def tool_names(self) -> list[str]:
        return [e.name for e in self.executions]

    @property
    def signature(self) -> tuple[str, ...]:
        """Identity of this step's actions, used to detect stalling.

        A model that requests the same tool with the same arguments twice running
        has stopped making progress -- it is not going to discover something new
        by repeating an identical call.
        """
        return tuple(
            f"{e.name}({sorted(e.arguments.items())!r})" for e in self.executions
        )


@dataclass
class Trajectory:
    """The complete record of a run.

    Deliberately much richer than "the answer". This object is what lesson 6
    writes assertions against, what lesson 8 turns into traces, and what lesson 7
    scores. Building it now costs almost nothing; retrofitting observability into
    a working agent is miserable.
    """

    question: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    stop_reason: StopReason = StopReason.ERROR
    messages: list[Message] = field(default_factory=list)
    note: str | None = None

    #: True if the loop told the model it was nearly out of steps. Recorded
    #: because it changes what COMPLETED means: the model may have wrapped up
    #: early under pressure rather than because it was genuinely finished.
    budget_warned: bool = False

    @property
    def usage(self) -> Usage:
        total = Usage()
        for step in self.steps:
            total = total + step.usage
        return total

    @property
    def succeeded(self) -> bool:
        return self.stop_reason.succeeded and bool(self.final_answer)

    @property
    def tool_sequence(self) -> list[str]:
        """Flat list of tools used, in order. The most useful debugging view.

        Assert on this rather than on the final text: it is far more stable across
        runs, and it tells you whether the agent reasoned correctly even when the
        prose varies.
        """
        return [name for step in self.steps for name in step.tool_names]

    @property
    def failed_executions(self) -> list[Execution]:
        return [e for step in self.steps for e in step.executions if not e.ok]


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant that solves problems using tools.\n"
    "Work one step at a time: call a tool, look at the result, then decide what to "
    "do next. Some questions need several tools in sequence, where a later call "
    "depends on an earlier result.\n"
    "Never guess a value a tool could give you exactly, and never invent a tool "
    "that was not offered to you.\n"
    "When you have everything you need, stop calling tools and give the final "
    "answer in plain prose."
)


def run_agent(
    client: LLMClient,
    question: str,
    registry: ToolRegistry,
    max_steps: int = 8,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
    on_step: object = None,  # callable(Step) -> None, for live output
    warn_near_limit: bool = True,
) -> Trajectory:
    """Run the agent loop until the model stops asking for tools.

    `max_steps` is not optional and has no "unlimited" setting. A model that
    misunderstands a task can request tools forever, and each iteration is a
    billed API call that re-sends the entire growing conversation. An unbounded
    agent loop is the classic way to turn a bug into an invoice.

    Eight is a reasonable default: enough for genuinely multi-step work, small
    enough that a confused agent stops quickly.

    `warn_near_limit` tells the model when it has one step left, which reliably
    turns "ran off the cliff mid-task" into "produced a partial answer". Good
    product behaviour, and it comes with a cost worth understanding: the run now
    ends as COMPLETED even though the model stopped because we pushed it. So the
    trajectory records `budget_warned`, and a COMPLETED run with that flag set
    deserves less trust than one without. Pass warn_near_limit=False to see the
    unvarnished MAX_STEPS failure instead.

    A caveat on COMPLETED generally: it means "the model stopped asking for
    tools", not "the answer is correct". An agent that politely explains it cannot
    do the task also completes. Distinguishing a real answer from a graceful
    refusal needs task-level scoring, which is lesson 7.
    """
    messages: list[Message] = [
        system(system_prompt or DEFAULT_SYSTEM_PROMPT),
        user(question),
    ]
    trajectory = Trajectory(question=question, messages=messages)
    seen_signatures: list[tuple[str, ...]] = []

    for index in range(1, max_steps + 1):
        # ---- THINK ----------------------------------------------------
        try:
            response = client.chat(
                messages,
                tools=registry.specs,
                max_tokens=max_tokens,
            )
        except PhantomToolCall as exc:
            trajectory.stop_reason = StopReason.PHANTOM_TOOL
            trajectory.note = str(exc)
            return trajectory

        step = Step(index=index, response=response)
        messages.append(response.as_message())

        # ---- STOP? ----------------------------------------------------
        # The stopping condition is simply "the model stopped asking for tools".
        # It decides when it is finished; we only decide when to give up.
        if not response.wants_tools:
            trajectory.steps.append(step)
            trajectory.final_answer = response.text

            if response.text:
                trajectory.stop_reason = StopReason.COMPLETED
                if trajectory.budget_warned:
                    # Honesty matters more than a clean-looking result. The model
                    # stopped because we told it to, not because it was done, so
                    # COMPLETED here is weaker than COMPLETED without the warning.
                    trajectory.note = (
                        "Answered on the final step, after being warned it was nearly "
                        "out of budget. Treat this as possibly incomplete: the model "
                        "wrapped up under pressure rather than because it finished. "
                        "Re-run with a higher max_steps to see the unhurried answer."
                    )
            else:
                # No tool calls and no text. Usually a reasoning model that spent
                # its whole budget thinking (lesson 0). Distinct from success.
                trajectory.stop_reason = StopReason.NO_ANSWER
                trajectory.note = (
                    "The model returned neither text nor a tool call. "
                    f"finish_reason={response.finish_reason}, "
                    f"{response.usage.reasoning_tokens} reasoning tokens. "
                    "Raise max_tokens."
                )
            if callable(on_step):
                on_step(step)
            return trajectory

        # ---- ACT ------------------------------------------------------
        # A model may request several tools in ONE response. Every one of them
        # needs a result appended before the next model call, each matched by its
        # own tool_call_id -- send back two results for three requests and the
        # provider rejects the whole conversation.
        for call in response.tool_calls:
            execution = registry.dispatch(call)
            step.executions.append(execution)
            # ---- OBSERVE ----------------------------------------------
            # Note that failures are appended exactly like successes. A tool error
            # is an observation the model can recover from, which is why
            # dispatch() returns a string instead of raising. Lesson 2, scenario 6.
            messages.append(tool_result(call.id, execution.result))

        trajectory.steps.append(step)
        if callable(on_step):
            on_step(step)

        # ---- STALL DETECTION ------------------------------------------
        # A model repeating an identical call is not going to learn anything new.
        # Left alone it will burn every remaining step. Cheap heuristic, real bug.
        signature = step.signature
        if signature and signature in seen_signatures:
            trajectory.stop_reason = StopReason.STALLED
            trajectory.note = (
                f"Step {index} repeated an identical tool call: "
                f"{', '.join(signature)}. Stopped early to avoid burning steps."
            )
            return trajectory
        seen_signatures.append(signature)

        # ---- BUDGET NUDGE ---------------------------------------------
        # Telling the model its remaining budget measurably improves the odds of
        # a useful partial answer instead of running off the cliff mid-task.
        # It is a nudge, not a guarantee -- the cap above is the real control.
        if warn_near_limit and index == max_steps - 1:
            messages.append(
                user(
                    "You have one step left. Give your best final answer now, "
                    "using what you already know. Say plainly what is still "
                    "uncertain rather than calling another tool."
                )
            )
            trajectory.budget_warned = True

    # Fell out of the loop: the model never stopped asking for tools.
    trajectory.stop_reason = StopReason.MAX_STEPS
    trajectory.note = (
        f"Reached the {max_steps}-step limit while the model was still requesting "
        f"tools. The answer is incomplete. Either the task needs more steps, or "
        f"the agent is confused about how to finish."
    )
    # Salvage any prose from the last step so a caller has something to show,
    # while stop_reason keeps it honest about being unfinished.
    for step in reversed(trajectory.steps):
        if step.response.text:
            trajectory.final_answer = step.response.text
            break
    return trajectory
