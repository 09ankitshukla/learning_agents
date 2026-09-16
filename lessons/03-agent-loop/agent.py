"""Lesson 3 deliverable: a real agent, and the ways a loop goes wrong.

    uv run lessons/03-agent-loop/agent.py                  # solves lesson 2's cliffhanger
    uv run lessons/03-agent-loop/agent.py --research       # multi-step file research
    uv run lessons/03-agent-loop/agent.py --question "..."
    uv run lessons/03-agent-loop/agent.py --max-steps 2    # watch it run out of steps
    uv run lessons/03-agent-loop/agent.py --impossible     # a task the tools cannot do
    uv run lessons/03-agent-loop/agent.py --sandbox        # try to escape the sandbox
    uv run lessons/03-agent-loop/agent.py --growth         # how context grows per step

The loop itself lives in loop.py and is about twenty lines. This file is the
harness around it: live step output, a trajectory view, and five experiments.
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llmkit import ConfigError, ToolCall, console, get_client
from loop import Step, StopReason, Trajectory, run_agent
from toolset import SANDBOX, build_registry

# Lesson 2 ended on this question. It needs three tools in sequence: the current
# time, a USD->INR conversion, then a percentage of the converted amount. One
# round cannot do it, because the percentage depends on the conversion's result.
CLIFFHANGER = (
    "What time is it in Mumbai right now, and if I invoice 2,450 USD today, "
    "how much is that in INR? Also what is 8.25% of that INR amount?"
)

RESEARCH = (
    "In this project, which lesson explains why we must never use eval() on "
    "model-generated input, and what does it recommend instead? "
    "Cite the file you found it in."
)

IMPOSSIBLE = (
    "What is the current share price of Amazon, and how many shares could I buy "
    "with 5,000 EUR?"
)


# ---------------------------------------------------------------------------
# Live output
# ---------------------------------------------------------------------------
def _brief(arguments: dict, limit: int = 70) -> str:
    """Compact argument rendering that never cuts a JSON string mid-token."""
    rendered = json.dumps(arguments)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 1] + "\u2026"


def make_step_printer(verbose: bool = True):
    def print_step(step: Step) -> None:
        if not verbose:
            return
        reply = step.response
        console.print(f"[bold cyan]step {step.index}[/bold cyan]", end="  ")

        if not reply.wants_tools:
            console.print("[green]no tools requested -> finishing[/green]")
            return

        console.print(f"[dim]{len(reply.tool_calls)} tool call(s)[/dim]")
        for execution in step.executions:
            style = "green" if execution.ok else "yellow"
            marker = "ok" if execution.ok else f"FAILED:{execution.failure_kind}"
            console.print(
                f"    [{style}]{marker}[/{style}] {execution.name}"
                f"({_brief(execution.arguments, 90)})"
            )
            first_line = execution.result.splitlines()[0] if execution.result else ""
            console.print(f"        [dim]{first_line[:110]}[/dim]")

    return print_step


def render_trajectory(trajectory: Trajectory, show_messages: bool = False) -> None:
    console.print(Rule("Trajectory"))

    table = Table(show_lines=True)
    table.add_column("step", width=5)
    table.add_column("tools requested", overflow="fold")
    table.add_column("outcome", overflow="fold")
    table.add_column("tokens", width=8)

    for step in trajectory.steps:
        if step.executions:
            tools = "\n".join(f"{e.name}({_brief(e.arguments)})" for e in step.executions)
            outcome = "\n".join(
                ("ok" if e.ok else f"FAILED:{e.failure_kind}") for e in step.executions
            )
        else:
            tools = "[dim]none -- returned prose[/dim]"
            outcome = "[green]final answer[/green]"
        table.add_row(str(step.index), tools, outcome, str(step.usage.total_tokens))

    console.print(table)

    reason = trajectory.stop_reason
    style = "green" if reason.succeeded else "yellow"
    flag = " [yellow](under budget pressure)[/yellow]" if trajectory.budget_warned else ""
    console.print(f"stop_reason: [{style}]{reason.value}[/{style}]{flag}")
    if trajectory.note:
        console.print(f"[yellow]note:[/yellow] {trajectory.note}")

    console.print(
        f"tool sequence: [cyan]{' -> '.join(trajectory.tool_sequence) or '(none)'}[/cyan]"
    )

    usage = trajectory.usage
    console.print(
        f"[dim]{len(trajectory.steps)} model call(s) | {usage.total_tokens} tokens "
        f"({usage.prompt_tokens} in, {usage.completion_tokens} out"
        + (f", {usage.reasoning_tokens} reasoning" if usage.reasoning_tokens else "")
        + f") | {usage.latency_s:.1f}s[/dim]"
    )
    console.print(
        "[dim]Assert on the tool sequence, not the prose. It is far more stable "
        "across runs and it tells you whether the agent reasoned correctly even "
        "when the wording changes. Lesson 6 builds on this.[/dim]"
    )

    if show_messages:
        console.print(Rule("Messages"))
        for i, msg in enumerate(trajectory.messages):
            role = msg["role"]
            body = msg.get("content") or ""
            if msg.get("tool_calls"):
                body = (body + " " + " ".join(
                    f"->{c['function']['name']}({c['function']['arguments'][:60]})"
                    for c in msg["tool_calls"]
                )).strip()
            console.print(f"[dim]{i:>2}[/dim] [cyan]{role:<10}[/cyan] {str(body)[:150]}")


def show_answer(trajectory: Trajectory) -> None:
    if trajectory.succeeded:
        console.print(Panel(trajectory.final_answer or "", title="final answer", style="green"))
        return

    title = f"incomplete ({trajectory.stop_reason.value})"
    body = trajectory.final_answer or "(no answer produced)"
    console.print(Panel(body, title=title, style="yellow"))
    console.print(
        "[bold]This is why the loop returns a stop_reason and not just a string.[/bold]\n"
        "A caller that only received the text above would show it to a user as "
        "though it were a finished answer. `trajectory.succeeded` is False here, so "
        "your code can retry, escalate, or say plainly that it did not finish."
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def experiment_step_limits(client, registry) -> None:
    """The same task under tightening budgets."""
    console.print(Rule("Step limits: the same task, less room"))
    console.print(f"[dim]{CLIFFHANGER}[/dim]\n")

    table = Table()
    table.add_column("max_steps", width=10)
    table.add_column("stop_reason", width=14)
    table.add_column("steps used", width=11)
    table.add_column("tool sequence", overflow="fold")

    for limit in (1, 2, 3, 6):
        trajectory = run_agent(client, CLIFFHANGER, registry, max_steps=limit)
        style = "green" if trajectory.succeeded else "yellow"
        table.add_row(
            str(limit),
            f"[{style}]{trajectory.stop_reason.value}[/{style}]",
            str(len(trajectory.steps)),
            " -> ".join(trajectory.tool_sequence) or "(none)",
        )
        console.print(f"[dim]  max_steps={limit} done[/dim]")

    console.print(table)
    console.print(
        Panel(
            "max_steps=1 reproduces lesson 2 exactly: one round, unfinished. Each "
            "extra step lets the agent get one dependency further.\n\n"
            "This is also the honest answer to 'what should max_steps be?' -- it is a "
            "property of the task, not a universal constant. Measure the shape of your "
            "real workload, then set the cap a little above it. What you must not do is "
            "leave it unbounded: every step is a billed call that re-sends the whole "
            "conversation.",
            style="cyan",
        )
    )


def experiment_sandbox(registry) -> None:
    """Try to read outside the project. No model needed -- these are direct calls."""
    console.print(Rule("Sandbox: path traversal attempts"))
    console.print(
        "Filesystem tools introduce a risk arithmetic did not. A path is a way out of\n"
        "your process. These are hand-built tool calls, exactly as in lesson 2's\n"
        f"failures.py -- no model required.\n\nSandbox root: [cyan]{SANDBOX}[/cyan]\n"
    )

    attempts = [
        ("classic traversal", {"path": "../../../../etc/passwd"}),
        ("windows traversal", {"path": "..\\..\\..\\Windows\\System32\\drivers\\etc\\hosts"}),
        ("absolute path", {"path": "C:/Windows/System32/config/SAM"}),
        ("the API key", {"path": ".env"}),
        ("sneaky re-entry", {"path": "lessons/../../.env"}),
        ("legitimate read", {"path": "README.md"}),
    ]

    table = Table(show_lines=True)
    table.add_column("attempt", width=18)
    table.add_column("path", overflow="fold")
    table.add_column("result", overflow="fold")

    for label, args in attempts:
        execution = registry.dispatch(ToolCall(id="probe", name="read_file", arguments=args))
        if execution.ok:
            outcome = f"[yellow]ALLOWED[/yellow] {execution.result.splitlines()[0][:60]}"
        else:
            outcome = f"[green]refused[/green] {execution.result[:90]}"
        table.add_row(label, str(args["path"]), outcome)

    console.print(table)
    console.print(
        Panel(
            "The control that does the work is `resolve()` followed by "
            "`is_relative_to(SANDBOX)`. Resolving first is what matters: it collapses "
            "'..' and follows symlinks, so a traversal becomes an absolute path that "
            "plainly fails containment. Checking the raw string instead is the classic "
            "path-traversal bug.\n\n"
            "`.env` is refused by a separate denylist, and the distinction is worth "
            "keeping straight. Containment is an allowlist and stops traversal. The "
            "denylist expresses 'this specific file is secret' -- it holds your API key, "
            "which an agent has no business reading and which you really do not want "
            "pasted into a prompt and shipped to a model provider.\n\n"
            "Only the last row succeeds, and it should.",
            style="red",
        )
    )


def experiment_growth(client, registry) -> None:
    """Watch the conversation grow, step by step."""
    console.print(Rule("Context growth: why long loops get expensive"))

    sizes: list[tuple[int, int, int]] = []

    def record(step: Step) -> None:
        sizes.append((step.index, step.usage.prompt_tokens, step.usage.completion_tokens))

    trajectory = run_agent(client, RESEARCH, registry, max_steps=8, on_step=record)

    table = Table()
    table.add_column("step", width=6)
    table.add_column("prompt tokens", width=15)
    table.add_column("growth", width=10)
    table.add_column("output tokens", width=14)

    previous = None
    for index, prompt_tokens, completion_tokens in sizes:
        delta = "" if previous is None else f"+{prompt_tokens - previous}"
        table.add_row(str(index), str(prompt_tokens), delta, str(completion_tokens))
        previous = prompt_tokens

    console.print(table)
    total = trajectory.usage
    console.print(
        f"total: {total.prompt_tokens} input tokens across {len(sizes)} calls, "
        f"{total.completion_tokens} output"
    )
    console.print(
        Panel(
            "Input tokens climb every step, because the entire conversation -- including "
            "every tool result -- is re-sent on each call. Nothing accumulates on the "
            "server (lesson 0).\n\n"
            "So cost grows roughly with the square of the step count, not linearly. A "
            "10-step agent is not 10x a 1-step agent; it is closer to 50x on input "
            "tokens. This is the single biggest reason agents surprise people on their "
            "bill, and it is what lesson 4 (trimming and summarising history) and "
            "lesson 8 (measuring it) exist to address.\n\n"
            "It is also why read_file truncates: one unbounded file read can push the "
            "actual task out of the context window.",
            style="cyan",
        )
    )


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="An agent loop, built by hand.")
    parser.add_argument("--question", help="Ask your own question.")
    parser.add_argument("--research", action="store_true", help="Multi-step file research task.")
    parser.add_argument(
        "--impossible",
        action="store_true",
        help="A task the tools cannot satisfy, to see graceful failure.",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Iteration cap (default 8).")
    parser.add_argument("--step-limits", action="store_true", help="Compare max_steps 1/2/3/6.")
    parser.add_argument("--sandbox", action="store_true", help="Path traversal attempts.")
    parser.add_argument("--growth", action="store_true", help="Show context growth per step.")
    parser.add_argument("--messages", action="store_true", help="Print the full message list.")
    parser.add_argument("--quiet", action="store_true", help="Hide live step output.")
    parser.add_argument(
        "--no-nudge",
        action="store_true",
        help="Disable the one-step-left warning, to see the raw max_steps failure.",
    )
    args = parser.parse_args()

    registry = build_registry()

    # No model needed for the sandbox demo.
    if args.sandbox:
        experiment_sandbox(registry)
        return 0

    try:
        client = get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(Panel.fit(f"{client.config.describe()}  |  {len(registry)} tools", style="bold cyan"))

    if args.step_limits:
        experiment_step_limits(client, registry)
        return 0
    if args.growth:
        experiment_growth(client, registry)
        return 0

    if args.question:
        question = args.question
    elif args.research:
        question = RESEARCH
    elif args.impossible:
        question = IMPOSSIBLE
    else:
        question = CLIFFHANGER

    console.print(Panel(question, title="task", style="blue"))

    if question == CLIFFHANGER:
        console.print(
            "[dim]This is the question lesson 2 could not answer. It needs three tools "
            "in sequence, and the third depends on the second's result.[/dim]\n"
        )
    if args.impossible:
        console.print(
            "[dim]None of the six tools can fetch a share price. Watch how the agent "
            "handles a task it cannot complete -- that behaviour matters more than the "
            "happy path.[/dim]\n"
        )

    trajectory = run_agent(
        client,
        question,
        registry,
        max_steps=args.max_steps,
        on_step=make_step_printer(not args.quiet),
        warn_near_limit=not args.no_nudge,
    )

    show_answer(trajectory)
    if args.impossible:
        console.print(
            Panel(
                "Look at the stop_reason below: most likely [bold]completed[/bold], with no tools "
                "used at all. The agent recognised that none of its six tools can fetch "
                "a share price and said so.\n\n"
                "That is the right behaviour and a genuine limitation of the loop at the "
                "same time. COMPLETED means [italic]'the model stopped asking for tools'[/italic] -- not "
                "'the answer is correct'. A graceful refusal and a correct answer are "
                "indistinguishable at this level.\n\n"
                "You cannot fix that inside the loop, because the loop has no notion of "
                "what a good answer looks like. It needs task-level scoring against "
                "expected outcomes, which is exactly what lesson 7 builds.",
                title="the limit of stop_reason",
                style="cyan",
            )
        )
    render_trajectory(trajectory, show_messages=args.messages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
