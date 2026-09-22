"""Lesson 4 deliverable: an agent that manages its own context.

    uv run lessons/04-memory-context/agent.py                # measure where tokens go
    uv run lessons/04-memory-context/agent.py --compare      # before/after, measured
    uv run lessons/04-memory-context/agent.py --orphan        # the trap, for real
    uv run lessons/04-memory-context/agent.py --strategies    # 4 strategies side by side
    uv run lessons/04-memory-context/agent.py --calibrate     # is the estimator honest?
    uv run lessons/04-memory-context/agent.py --save mine.json
    uv run lessons/04-memory-context/agent.py --resume mine.json

The agent loop is unchanged from lesson 3. Context management is a transformation
applied to the message list just before sending, which is why lesson 3's loop
needed exactly one optional hook and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llmkit import ConfigError, console, get_client, system, user

# Lesson 3's loop and tools are reused unchanged. Lesson folders are not
# importable packages (digit-leading names), so the path is added explicitly.
_LESSON_03 = Path(__file__).resolve().parents[1] / "03-agent-loop"
if str(_LESSON_03) not in sys.path:
    sys.path.insert(0, str(_LESSON_03))

from loop import run_agent  # noqa: E402
from toolset import build_registry  # noqa: E402

from context import (  # noqa: E402
    ContextManager,
    breakdown,
    compress_tool_results,
    conversation_tokens,
    estimate_tokens,
    group_messages,
    summarise_history,
    tool_schema_tokens,
    trim_naive,
    trim_safe,
    validate,
)
from session import Session, SessionError, load_session, save_session  # noqa: E402

# A task that needs several file reads, so tool results genuinely dominate the
# context. This is the same shape as lesson 3's research task, made longer.
TASK = (
    "Look through this project's lesson notes and tell me three distinct things "
    "the project learned about reasoning models and token budgets. "
    "Cite the file each one came from."
)

#: Deliberately small so the machinery is visible in seconds.
#:
#: gpt-oss-120b's real window is far larger; filling it naturally would be slow
#: and expensive. Every number produced under this budget is about the *mechanism*,
#: not the model's true limits.
#:
#: Set low enough that compaction reliably fires. An earlier value of 2500 was
#: never crossed on short runs, which produced a before/after table comparing
#: nothing -- the two runs differed only by ordinary run-to-run variance. If you
#: raise this and see "0 compactions", the comparison is void, not favourable.
SYNTHETIC_BUDGET = 1200


# ---------------------------------------------------------------------------
def experiment_measure(client, registry) -> None:
    """Run the task and show exactly where the tokens went."""
    console.print(Rule("Where the budget actually goes"))

    schema_cost = tool_schema_tokens(registry.specs)
    console.print(
        f"[dim]Tool schemas alone: ~{schema_cost} tokens, re-sent on every single "
        f"call before any conversation exists.[/dim]\n"
    )

    trajectory = run_agent(client, TASK, registry, max_steps=6)

    table = Table(show_lines=False)
    table.add_column("#", width=4)
    table.add_column("role", width=10)
    table.add_column("est. tokens", width=12, justify="right")
    table.add_column("content", overflow="fold")

    report = breakdown(trajectory.messages)
    for index, role, tokens, preview in report.rows:
        # Highlight the messages worth attacking first.
        shown = f"[yellow]{tokens}[/yellow]" if tokens > 400 else str(tokens)
        table.add_row(str(index), role, shown, preview)

    console.print(table)

    console.print(f"\nconversation total: [bold]{report.total}[/bold] estimated tokens")
    console.print(f"by role: {report.by_role}")
    largest = report.largest
    if largest:
        console.print(
            f"largest single message: #{largest[0]} ({largest[1]}) at {largest[2]} tokens"
        )
    console.print(
        Panel(
            "The `tool` role almost always dominates, and usually one or two "
            "messages account for most of it. That tells you where to spend effort: "
            "compressing tool results beats clever trimming of the dialogue.\n\n"
            "Note also that the assistant messages requesting tools have "
            "content=null but are not free -- their tool_call arguments are real "
            "tokens. A naive counter that only measures `content` scores them zero.",
            style="cyan",
        )
    )


def experiment_orphan(client) -> None:
    """Break a conversation the way naive trimming does, and let the API reject it."""
    console.print(Rule("The trap: naive trimming orphans tool results"))
    console.print(
        "A message list is not a flat sequence. An assistant turn with tool_calls and\n"
        "the tool messages answering it are one atomic unit, paired by tool_call_id.\n"
        "Cut between them and the provider refuses the whole request.\n"
    )

    registry = build_registry()
    trajectory = run_agent(client, "What time is it in Tokyo, and what is 71 * 89?", registry, max_steps=5)
    full = trajectory.messages
    console.print(f"Built a real conversation: {len(full)} messages, "
                  f"{conversation_tokens(full)} est. tokens")
    console.print(f"[dim]roles: {' -> '.join(m['role'] for m in full)}[/dim]\n")

    groups = group_messages(full)
    console.print("Atomic groups the conversation actually contains:")
    for group in groups:
        marker = "protected" if not group.droppable else "droppable"
        console.print(
            f"  indices {group.indices} [{'yellow' if not group.droppable else 'green'}]"
            f"{group.kind}[/] ({marker}), {group.tokens} tokens"
        )

    # --- the naive cut ---------------------------------------------------
    # Note that naive trimming is not *reliably* broken, which is what makes it
    # dangerous. Whether a cut lands on a group boundary is luck, so the bug
    # appears intermittently and looks like a flaky provider. Search for a
    # keep_last that actually splits an exchange, and say so if none does.
    console.print(Rule("naive: keep system + last N messages", style="red"))

    bad = None
    for keep_last in range(2, len(full)):
        candidate = trim_naive(full, keep_last=keep_last)
        if validate(candidate.messages):
            bad = candidate
            console.print(f"[dim]keep_last={keep_last} splits an exchange[/dim]")
            break
        console.print(f"[dim]keep_last={keep_last} happened to be valid (luck, not design)[/dim]")

    if bad is None:
        # Fall back to constructing the failure explicitly, so the demo is
        # deterministic regardless of this run's conversation shape.
        console.print("[dim]no naive cut split this conversation; orphaning one deliberately[/dim]")
        owner = next(
            (i for i, m in enumerate(full) if m.get("role") == "assistant" and m.get("tool_calls")),
            None,
        )
        broken = [m for i, m in enumerate(full) if i != owner]
        bad = trim_naive(broken, keep_last=len(broken))
        bad.messages = broken

    console.print(f"[dim]roles: {' -> '.join(m['role'] for m in bad.messages)}[/dim]")
    problems = validate(bad.messages)
    for problem in problems:
        console.print(f"  [red]local check: {problem}[/red]")

    # Tools are passed here and in the safe case below, so the *only* difference
    # between the two requests is the structural break. Omitting them would also
    # trigger lesson 2's phantom-tool bug and muddy the result.
    console.print("\n[bold]Now sending it to the provider anyway:[/bold]")
    try:
        client.chat(bad.messages, tools=registry.specs, max_tokens=200)
        console.print("  [yellow]accepted -- this provider is lenient here. Many are not, "
                      "and a lenient provider silently gives the model a conversation "
                      "that references a request it cannot see.[/yellow]")
    except Exception as exc:  # noqa: BLE001
        message = str(exc).split("\n")[0][:240]
        console.print(f"  [red]REJECTED: {type(exc).__name__}[/red]")
        console.print(f"  [red]{message}[/red]")

    # --- the safe cut ----------------------------------------------------
    # Budget set below the conversation's own size, otherwise nothing needs to be
    # dropped and the comparison proves nothing.
    console.print(Rule("safe: drop whole exchanges", style="green"))
    tight = max(150, int(conversation_tokens(full) * 0.55))
    console.print(f"[dim]budget {tight} tokens vs conversation {conversation_tokens(full)}[/dim]")
    good = trim_safe(full, budget=tight, keep_recent_exchanges=1)
    console.print(f"[dim]roles: {' -> '.join(m['role'] for m in good.messages)}[/dim]")
    console.print(f"  local check: [green]{'; '.join(good.notes)}[/green]")
    console.print(f"  {good.before} -> {good.after} tokens, {good.dropped_groups} exchange(s) dropped")
    try:
        reply = client.chat(good.messages, tools=registry.specs, max_tokens=300)
        outcome = reply.text or f"(requested {reply.tool_calls[0].name})" if reply.tool_calls else reply.text
        console.print(f"  [green]accepted[/green] -> {str(outcome)[:110]}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]unexpectedly rejected: {str(exc)[:120]}[/red]")

    console.print(
        Panel(
            "Both lists are shorter. Only one is valid.\n\n"
            "Three things to take from this.\n\n"
            "**Naive trimming is not reliably broken, which is what makes it "
            "dangerous.** Whether a cut lands on a group boundary is luck, so the bug "
            "surfaces intermittently and looks like a flaky provider rather than your "
            "bookkeeping. That is why `context.py` groups messages first and why "
            "`validate()` exists -- catching it locally beats a 400 three steps into "
            "a run.\n\n"
            "**A budget can be unreachable.** The trim reported a FLOOR: the system "
            "prompt and task alone cost more than the budget asked for, so no amount "
            "of dropping gets there. The fix is a shorter system prompt or "
            "summarisation, not more trimming. Silently returning something too big "
            "would have hidden that.\n\n"
            "**Trimming makes the agent redo work.** The safe-trimmed conversation was "
            "accepted, and the model immediately re-requested a tool whose result we "
            "had just deleted. That is correct behaviour and it is the real price of "
            "trimming: you trade context tokens for repeated tool calls. Sometimes "
            "that is a good trade and sometimes it is a loop.",
            style="cyan",
        )
    )


def experiment_strategies(client) -> None:
    """Four strategies on the same conversation, measured."""
    console.print(Rule("Four strategies, one conversation"))

    registry = build_registry()
    trajectory = run_agent(client, TASK, registry, max_steps=6)
    original = trajectory.messages
    console.print(
        f"Source conversation: {len(original)} messages, "
        f"{conversation_tokens(original)} est. tokens\n"
    )

    results = []
    results.append(("do nothing", None, conversation_tokens(original), "baseline"))

    compressed = compress_tool_results(original, max_chars=500)
    results.append(
        ("compress tool results", compressed, compressed.after, "no model call")
    )

    trimmed = trim_safe(original, budget=1500, keep_recent_exchanges=2)
    results.append(("safe trim", trimmed, trimmed.after, "forgets detail"))

    summarised = summarise_history(client, original, keep_recent_exchanges=2)
    results.append(
        ("summarise", summarised, summarised.after, "costs an extra model call")
    )

    table = Table()
    table.add_column("strategy", width=22)
    table.add_column("tokens", justify="right", width=8)
    table.add_column("saved", justify="right", width=8)
    table.add_column("% of original", justify="right", width=14)
    table.add_column("trade-off", overflow="fold")

    baseline = conversation_tokens(original)
    for label, result, after, note in results:
        saved = baseline - after
        table.add_row(
            label,
            str(after),
            str(saved) if saved else "-",
            f"{after / baseline:.0%}",
            note,
        )
    console.print(table)

    for label, result, _, _ in results:
        if result and result.notes:
            console.print(f"[dim]{label}: {'; '.join(result.notes)}[/dim]")

    console.print(
        Panel(
            "Order your policy cheapest-first, which is what ContextManager does:\n\n"
            "1. Under budget? Do nothing. Compaction is not free.\n"
            "2. Compress tool results. No model call, usually the biggest single win, "
            "and it keeps every step's structure intact.\n"
            "3. Only then trim or summarise, which lose information or cost a call.\n\n"
            "Trimming forgets; summarising remembers the gist for the price of an "
            "extra request. Neither is free, and summaries compound -- summarise a "
            "conversation containing a summary and detail decays geometrically.",
            style="cyan",
        )
    )


def experiment_compare(client, registry, budget: int) -> None:
    """The measured before/after: same task, with and without management."""
    console.print(Rule("Measured: same task, with and without context management"))
    console.print(f"[dim]Synthetic budget: {budget} tokens (see note below)[/dim]\n")

    console.print("[bold]run 1: no context management[/bold]")
    plain = run_agent(client, TASK, registry, max_steps=6)
    console.print(
        f"  steps={len(plain.steps)} stop={plain.stop_reason.value} "
        f"input_tokens={plain.usage.prompt_tokens}"
    )

    console.print("\n[bold]run 2: with context management[/bold]")
    manager = ContextManager(
        budget=budget, strategy="auto", client=client, verbose=True
    )
    managed = run_agent(client, TASK, registry, max_steps=6, compactor=manager)
    console.print(
        f"  steps={len(managed.steps)} stop={managed.stop_reason.value} "
        f"input_tokens={managed.usage.prompt_tokens}"
    )

    table = Table()
    table.add_column("", width=24)
    table.add_column("no management", justify="right", width=15)
    table.add_column("managed", justify="right", width=12)
    table.add_column("change", justify="right", width=12)

    def row(label: str, a: float, b: float, fmt: str = "{:,.0f}") -> None:
        delta = b - a
        pct = f"{delta / a:+.0%}" if a else "-"
        table.add_row(label, fmt.format(a), fmt.format(b), f"{pct}")

    row("input tokens", plain.usage.prompt_tokens, managed.usage.prompt_tokens)
    row("output tokens", plain.usage.completion_tokens, managed.usage.completion_tokens)
    row("total tokens", plain.usage.total_tokens, managed.usage.total_tokens)
    row("steps", len(plain.steps), len(managed.steps), "{:.0f}")
    # Input tokens per step is the metric that actually isolates the effect.
    # Totals move with step count, which varies run to run for reasons unrelated
    # to context management.
    row(
        "input tokens / step",
        plain.usage.prompt_tokens / max(1, len(plain.steps)),
        managed.usage.prompt_tokens / max(1, len(managed.steps)),
    )
    console.print(table)

    console.print(f"\ncompactions applied: [bold]{len(manager.history)}[/bold]")
    for item in manager.history:
        console.print(
            f"  [dim]{item.strategy}: {item.before} -> {item.after} "
            f"({'; '.join(item.notes)})[/dim]"
        )

    if not manager.history:
        console.print(
            Panel(
                "[bold]No compaction happened, so this comparison is void.[/bold]\n\n"
                f"The conversation never exceeded the {budget} token budget, so the "
                "managed run did exactly what the unmanaged run did. Any difference in "
                "the table above is ordinary run-to-run variance -- different step "
                "counts, different tool choices -- and none of it is attributable to "
                "context management.\n\n"
                "Lower the budget and try again:\n"
                "  [cyan]uv run lessons/04-memory-context/agent.py --compare --budget 800[/cyan]\n\n"
                "This is worth seeing once. A measurement harness that reports a "
                "plausible-looking table when the thing under test never ran is how "
                "false conclusions get made. Always check that the mechanism fired.",
                title="void result",
                style="red",
            )
        )
        return

    console.print(
        "\n[yellow]On latency:[/yellow] wall-clock times on a free tier are dominated by "
        "queue waiting, not by your code. Compare token counts, which are "
        "deterministic given the same messages; treat latency as unreliable here."
    )

    # Summarisation can cost more than it saves. Worth computing rather than
    # assuming, because the intuition runs the wrong way.
    for item in manager.history:
        if item.strategy == "summarise":
            cost_note = next((n for n in item.notes if "summary cost" in n), "")
            if cost_note and item.saved >= 0:
                console.print(
                    f"\n[yellow]Note the economics:[/yellow] that summarisation saved "
                    f"{item.saved} tokens and {cost_note.split('summary cost ')[1].split(' ')[0]} "
                    f"tokens to produce.\n"
                    "Summarisation only pays off if many calls follow it, because you "
                    "pay once and save on every subsequent request. Summarise near the "
                    "end of a run and it is a straight loss. Compressing tool results "
                    "has no such break-even -- it costs nothing but CPU."
                )
                break

    console.print(Rule("answers, for quality comparison"))
    console.print(Panel((plain.final_answer or "(none)")[:700], title="unmanaged", style="blue"))
    console.print(Panel((managed.final_answer or "(none)")[:700], title="managed", style="green"))

    console.print(
        Panel(
            "Read both answers before celebrating the token saving. Context "
            "management is a [bold]trade[/bold], not a free optimisation: you are deleting "
            "information the model might have used. The honest way to report a "
            "result here is tokens saved [italic]and[/italic] quality retained, which is why "
            "lesson 7 builds scoring -- eyeballing two answers does not scale, and "
            "neither of us can tell from one run whether quality held up.\n\n"
            f"Also note the {budget}-token budget is artificial. The real "
            "window here is far larger; a small budget makes the mechanism visible "
            "in seconds instead of requiring an expensive run.",
            style="yellow",
        )
    )


def experiment_calibrate(client) -> None:
    """Check the token estimator against what the provider actually charged."""
    console.print(Rule("Is the estimator honest?"))
    console.print(
        "The estimator is chars/3.7 plus per-message overhead. No tokenizer, no\n"
        "dependency, and therefore no guarantee. So measure the error rather than\n"
        "trusting it.\n"
    )

    probes = [
        ("short prose", [user("What is 2 + 2?")]),
        ("longer prose", [user("Explain in three sentences why agents need tools. " * 6)]),
        (
            "json-ish",
            [user('{"path": "docs/glossary.md", "start_line": 1, "max_lines": 120}' * 8)],
        ),
        (
            "with system prompt",
            [system("You are concise and precise."), user("Name two primary colours.")],
        ),
    ]

    table = Table()
    table.add_column("sample", width=20)
    table.add_column("estimated", justify="right", width=11)
    table.add_column("actual", justify="right", width=9)
    table.add_column("error", justify="right", width=10)

    errors = []
    for label, messages in probes:
        estimated = conversation_tokens(messages)
        reply = client.chat(messages, max_tokens=16)
        actual = reply.usage.prompt_tokens
        error = (estimated - actual) / actual if actual else 0
        errors.append(error)
        style = "green" if abs(error) < 0.25 else "yellow"
        table.add_row(label, str(estimated), str(actual), f"[{style}]{error:+.0%}[/{style}]")

    console.print(table)
    worst = max(errors, key=abs) if errors else 0
    console.print(f"\nworst error: {worst:+.0%}")
    console.print(
        Panel(
            "This experiment is in the lesson because it caught a real bug.\n\n"
            "The first estimator was chars/3.7 plus 4 tokens per message. Measured, it "
            "was [red]91% too low[/red] on a short prompt: 7 estimated against 79 charged. "
            "Two causes, both invisible without measuring:\n\n"
            "1. **A large fixed per-request cost.** gpt-oss is served with the "
            "\"harmony\" chat template, which injects its own preamble before your "
            "messages exist. ~70 tokens on every call, whatever you send.\n"
            "2. **Density varies with content.** JSON, paths and code tokenize roughly "
            "40% denser per character than prose, because punctuation tokenizes badly.\n\n"
            "Correcting both took the worst error from -91% to about +19%, and the "
            "remaining error now [bold]over[/bold]-estimates, which is the safe direction.\n\n"
            "**An uncalibrated estimator is not conservative, it is just wrong** -- and "
            "wrong in the direction that overflows your context window. Re-run this "
            "whenever you change model or provider.\n\n"
            "Use estimates to decide what to send, and `usage.prompt_tokens` afterwards "
            "to check yourself. If you need real accuracy before sending, run the "
            "model's own tokenizer (tiktoken for OpenAI models, "
            "transformers.AutoTokenizer for open weights) -- and make sure it matches "
            "the model, or it will lie to you confidently.",
            style="cyan",
        )
    )


def experiment_persistence(client, registry, save_to: str | None, resume_from: str | None) -> None:
    """Save a conversation, or resume one."""
    if resume_from:
        console.print(Rule(f"Resuming from {resume_from}"))
        try:
            session = load_session(resume_from)
        except SessionError as exc:
            console.print(f"[red]{exc}[/red]")
            return

        console.print(
            f"loaded {len(session.messages)} messages, ~{session.tokens} tokens, "
            f"saved {session.updated_at}"
        )
        console.print(f"[dim]original question: {session.question}[/dim]")
        console.print(
            "\n[yellow]Note it reloads at full size.[/yellow] A session saved at "
            f"{session.tokens} tokens costs that on its very first new call, so "
            "compaction belongs before the next request, not after.\n"
        )

        follow_up = "Based on what you already found, which single finding is most important and why?"
        console.print(f"[bold]follow-up:[/bold] {follow_up}\n")

        # Sizing rule, learned the hard way: a resumed session's budget must
        # exceed the session's own size, with headroom.
        #
        # With the default 1200-token budget against a 2645-token session, the
        # compactor immediately deleted the findings the follow-up was asking
        # about. The agent then re-ran searches to rediscover what it already knew
        # and ran out of steps without answering. Over-aggressive compaction does
        # not just lose detail; it destroys the premise of the question.
        resume_budget = max(SYNTHETIC_BUDGET, session.tokens + 1000)
        console.print(
            f"[dim]budget for this run: {resume_budget} tokens "
            f"(session is {session.tokens}; a budget below that would compact away "
            f"the very work we just loaded)[/dim]\n"
        )

        # Continue the loop from the loaded conversation rather than making a
        # single call. The model may well want another tool, and only a loop can
        # handle that -- which is the whole point of lesson 3.
        manager = ContextManager(
            budget=resume_budget, strategy="auto", client=client, verbose=True
        )
        trajectory = run_agent(
            client,
            follow_up,
            registry,
            max_steps=6,
            compactor=manager,
            initial_messages=session.messages,
        )
        console.print(
            Panel(
                trajectory.final_answer or "(no answer)",
                title=f"answer ({trajectory.stop_reason.value})",
                style="green" if trajectory.succeeded else "yellow",
            )
        )
        console.print(
            f"[dim]tools used on resume: "
            f"{' -> '.join(trajectory.tool_sequence) or '(none needed)'}[/dim]"
        )
        console.print(
            "[dim]The model built on work done by a previous process that no longer "
            "exists. That is the whole point: state lives in the message list, so it "
            "survives a restart -- and it can keep using tools from there.[/dim]"
        )
        return

    console.print(Rule("Running the task, then saving the session"))
    trajectory = run_agent(client, TASK, registry, max_steps=6)
    session = Session(
        messages=trajectory.messages,
        question=TASK,
        model=client.config.model,
        stats={
            "steps": len(trajectory.steps),
            "stop_reason": trajectory.stop_reason.value,
            "prompt_tokens": trajectory.usage.prompt_tokens,
        },
    )
    try:
        path = save_session(save_to or "runs/session.json", session)
    except SessionError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    console.print(f"saved to [cyan]{path}[/cyan]")
    console.print(f"  {len(session.messages)} messages, ~{session.tokens} est. tokens")
    console.print(
        f"\nResume it with:\n  [cyan]uv run lessons/04-memory-context/agent.py "
        f"--resume {path}[/cyan]"
    )
    console.print(
        "[dim]runs/ is gitignored. A saved session contains the full conversation, "
        "which may include file contents and anything else a tool returned -- treat "
        "it as sensitive.[/dim]"
    )


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="An agent that manages its own context.")
    parser.add_argument("--compare", action="store_true", help="Measured before/after.")
    parser.add_argument("--orphan", action="store_true", help="The trimming trap, for real.")
    parser.add_argument("--strategies", action="store_true", help="Four strategies compared.")
    parser.add_argument("--calibrate", action="store_true", help="Check the token estimator.")
    parser.add_argument("--save", metavar="PATH", nargs="?", const="runs/session.json",
                        help="Run the task and save the session.")
    parser.add_argument("--resume", metavar="PATH", help="Resume a saved session.")
    parser.add_argument("--budget", type=int, default=SYNTHETIC_BUDGET, help="Token budget.")
    args = parser.parse_args()

    try:
        client = get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    registry = build_registry()
    console.print(Panel.fit(f"{client.config.describe()}  |  {len(registry)} tools", style="bold cyan"))

    if args.orphan:
        experiment_orphan(client)
    elif args.strategies:
        experiment_strategies(client)
    elif args.calibrate:
        experiment_calibrate(client)
    elif args.compare:
        experiment_compare(client, registry, args.budget)
    elif args.resume or args.save:
        experiment_persistence(client, registry, args.save, args.resume)
    else:
        experiment_measure(client, registry)

    return 0


if __name__ == "__main__":
    sys.exit(main())
