"""Lesson 10 deliverable: delegation, handoff, and whether either is worth it.

    uv run lessons/10-multi-agent/multi.py --team                  # show the wiring, free
    uv run lessons/10-multi-agent/multi.py --ask "QUESTION"        # delegating agent
    uv run lessons/10-multi-agent/multi.py --pipeline "TOPIC"      # research/write/critique
    uv run lessons/10-multi-agent/multi.py --pipeline "TOPIC" --mode shared --revise
    uv run lessons/10-multi-agent/multi.py --recursion             # the depth guard
    uv run lessons/10-multi-agent/multi.py --probe CASE_ID         # one eval case, both ways
    uv run lessons/10-multi-agent/multi.py --evaluate              # the honest comparison

`--evaluate` runs the 16-case suite through the delegating agent and judges it with
lesson 9's rule, appending the result to the shared attempt log. Budget for it: a
delegating run costs roughly the sum of its agents, so expect two to three times the
usual ~33,000 tokens. Follow lesson 9's own advice and `--probe` a couple of cases
first; it is a twentieth of the price and usually enough to decide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

from llmkit import ConfigError, console, get_client

_LESSONS = Path(__file__).resolve().parents[1]
for _folder in ("01-structured-output", "02-tool-calling", "03-agent-loop",
                "07-evaluation", "08-judging-tracing", "09-iteration", "10-multi-agent"):
    _path = str(_LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from toolset import build_registry  # noqa: E402
from tracing import cost_usd  # noqa: E402

from dataset import CASES, by_id  # noqa: E402
from experiments import EVAL_MODEL  # noqa: E402
from harness import EvalRun, compare, run_eval  # noqa: E402
from iteration import Changelog, Decision, build_attempt, decide, load_noise  # noqa: E402
from pipeline import research_write_critique, run_pipeline  # noqa: E402
from team import (  # noqa: E402
    COORDINATOR_PROMPT,
    TEAM,
    DelegationBudget,
    DelegationLog,
    build_team_registry,
)

DEFAULT_PAUSE = 20.0


# ---------------------------------------------------------------------------
def show_team() -> int:
    """The wiring, with no model involved."""
    base = build_registry()
    console.print(Rule("the team"))

    table = Table()
    table.add_column("agent", style="cyan", width=16)
    table.add_column("tools it may use", width=42, overflow="fold")
    table.add_column("steps", justify="right", width=6)
    table.add_row("coordinator", ", ".join(base.names) + ", + the agents below", "8")
    for sub in TEAM:
        table.add_row(sub.name, ", ".join(sub.tools) or "[dim](none)[/dim]", str(sub.max_steps))
    console.print(table)

    log = DelegationLog()
    client = None
    try:
        client = get_client(model=EVAL_MODEL)
    except ConfigError:
        pass
    if client is not None:
        team_registry = build_team_registry(base, TEAM, client, log)
        console.print(
            f"\ncoordinator sees {len(team_registry)} tools: "
            f"{', '.join(team_registry.names)}"
        )
        nested = build_team_registry(
            base, TEAM, client, log, budget=DelegationBudget(depth=1, max_depth=1)
        )
        console.print(
            f"a sub-agent at depth 1 sees {len(nested)} tools: "
            f"{', '.join(nested.names)}  [dim](no delegation tools)[/dim]"
        )

    console.print(
        Panel(
            "A sub-agent is a tool whose implementation is another agent loop.\n\n"
            "`run_agent` is unchanged from lesson 3. The wire protocol is unchanged. "
            "Lesson 2's dispatcher is still the security boundary. The parent model "
            "sees `ask_researcher` exactly as it sees `convert_currency` and cannot "
            "tell the difference — which is why this needed no new machinery, only "
            "`ToolRegistry.subset`, written in lesson 2 and unused until now.\n\n"
            "Note the second row: a sub-agent is offered no delegation tools. Depth is "
            "enforced by *not showing* the capability rather than by refusing the call "
            "afterwards, which is strictly better — a tool the model cannot see is a "
            "tool it cannot be talked into using.",
            style="cyan",
        )
    )
    return 0


# ---------------------------------------------------------------------------
def ask(client, question: str, max_steps: int, keep_own: bool) -> int:
    """Run the delegating coordinator on one question."""
    from loop import run_agent

    base = build_registry()
    log = DelegationLog()
    registry = build_team_registry(
        base, TEAM, client, log, keep_own_tools=None if keep_own else []
    )

    console.print(Panel(question, title="question", style="blue"))
    if not keep_own:
        console.print(
            "[yellow]--router:[/yellow] the coordinator has no tools of its own, so "
            "every piece of work costs a full sub-agent run.\n"
        )

    def on_step(step) -> None:
        names = [c.name for c in step.response.tool_calls]
        console.print(f"[dim]step {step.index}:[/dim] {', '.join(names) or 'final answer'}")

    trajectory = run_agent(
        client,
        question,
        registry,
        max_steps=max_steps,
        system_prompt=COORDINATOR_PROMPT,
        on_step=on_step,
    )

    console.print(Rule("answer"))
    console.print(trajectory.final_answer or "[red](none)[/red]")
    console.print(f"\n[dim]stop_reason: {trajectory.stop_reason.value}[/dim]")

    _print_delegations(log)
    _print_hidden_cost(trajectory.usage, log, client.config.model)
    return 0


def _print_delegations(log: DelegationLog) -> None:
    if not log.delegations:
        console.print("\n[dim]no delegation happened — the coordinator did it all itself.[/dim]")
        return
    tree = Tree("delegations")
    for d in log.delegations:
        mark = "[green]ok[/green]" if d.ok else "[red]failed[/red]"
        node = tree.add(
            f"{d.agent}  {mark}  [dim]{d.stop_reason}, {d.steps} step(s), "
            f"{d.total_tokens:,} tokens, {d.duration_s:.1f}s[/dim]"
        )
        node.add(f"[dim]task:[/dim] {d.task[:150]}")
        node.add(f"[dim]tools:[/dim] {' -> '.join(d.tool_sequence) or '(none)'}")
        if d.note:
            node.add(f"[yellow]note:[/yellow] {d.note[:150]}")
    console.print()
    console.print(tree)


def _print_hidden_cost(parent_usage, log: DelegationLog, model: str) -> int:
    """The number every existing tool in this project gets wrong for a delegating run."""
    parent_tokens = parent_usage.prompt_tokens + parent_usage.completion_tokens
    sub_tokens = log.total_tokens
    total = parent_tokens + sub_tokens

    table = Table(title="token accounting")
    table.add_column("", width=34)
    table.add_column("tokens", justify="right", width=10)
    table.add_column("cost", justify="right", width=10)
    table.add_row(
        "parent (what Trajectory.usage sees)",
        f"{parent_tokens:,}",
        f"${cost_usd(model, parent_usage.prompt_tokens, parent_usage.completion_tokens):.5f}",
    )
    table.add_row(
        "sub-agents (invisible to it)",
        f"{sub_tokens:,}",
        f"${cost_usd(model, log.usage.prompt_tokens, log.usage.completion_tokens):.5f}",
    )
    table.add_row(
        "[bold]actual total[/bold]",
        f"[bold]{total:,}[/bold]",
        f"[bold]${cost_usd(model, parent_usage.prompt_tokens + log.usage.prompt_tokens, parent_usage.completion_tokens + log.usage.completion_tokens):.5f}[/bold]",
    )
    console.print()
    console.print(table)

    if sub_tokens:
        console.print(
            Panel(
                f"[bold]{log.hidden_share(parent_tokens):.0%} of this run's tokens are "
                f"invisible to lesson 3's `Trajectory`.[/bold]\n\n"
                f"A sub-agent's model calls happen inside `registry.dispatch()`, which "
                f"is not a place `Trajectory` looks — it was designed before sub-agents "
                f"existed and counts only the steps it can see. So lesson 7's per-case "
                f"token counts and lesson 8's `CostReport` both under-report a "
                f"delegating agent by this much.\n\n"
                f"This is the fifth silent measurement bug in this project and the "
                f"first that was predicted rather than discovered. It is why "
                f"`DelegationLog` exists, and why this table shows two numbers instead "
                f"of the one you would naturally report.",
                style="yellow",
            )
        )
    return 0


# ---------------------------------------------------------------------------
def pipeline(client, topic: str, mode: str, revise: bool) -> int:
    base = build_registry()
    stages = research_write_critique(revise=revise)

    console.print(Panel(f"{topic}\n\n[dim]mode: {mode}[/dim]", title="topic", style="blue"))

    def on_stage(result) -> None:
        mark = "[green]ok[/green]" if result.ok else "[red]failed[/red]"
        console.print(
            f"[dim]{result.agent:16}[/dim] {mark}  {result.stop_reason}, "
            f"{result.steps} step(s), {result.total_tokens:,} tokens"
            + (f", {result.messages_in} messages in" if result.messages_in else "")
        )

    run = run_pipeline(client, topic, base, stages, mode=mode, on_stage=on_stage)

    console.print(Rule("stages"))
    for result in run.stages:
        console.print(f"\n[bold cyan]{result.agent}[/bold cyan]")
        console.print(result.output or "[red](nothing)[/red]")

    console.print(Rule("final"))
    console.print(run.final_answer or "[red](none — pipeline stopped early)[/red]")

    approved = run.critic_approved
    console.print(
        f"\n[dim]critic: {'approved' if approved else 'raised problems' if approved is False else 'did not run'}"
        f"  |  {run.total_tokens:,} tokens  |  {run.duration_s:.1f}s[/dim]"
    )
    if run.stopped_early:
        console.print("[red]stopped early: a required stage did not finish.[/red]")

    console.print(
        Panel(
            "Handoff, not delegation. No model chose this sequence — it is fixed in "
            "code, which makes it cheaper and more predictable and unable to adapt "
            "when a stage returns something unexpected.\n\n"
            "The rule that falls out: **if you know the sequence in advance, a "
            "pipeline beats a delegating agent**, because otherwise you are paying a "
            "model to make a decision you already made. Delegation earns its cost only "
            "when the route depends on what is found along the way.\n\n"
            "Try `--mode shared` to see what carrying the whole message list costs, "
            "and `--compare` to measure the two against each other.",
            style="cyan",
        )
    )
    return 0


def compare_modes(client, topic: str) -> int:
    """relay versus shared, on the same topic."""
    base = build_registry()
    runs = {}
    for mode in ("relay", "shared"):
        console.print(f"\n[dim]running {mode}...[/dim]")
        runs[mode] = run_pipeline(
            client, topic, base, research_write_critique(), mode=mode
        )

    table = Table(title="relay vs shared")
    table.add_column("metric", width=26)
    table.add_column("relay", justify="right", width=12)
    table.add_column("shared", justify="right", width=12)
    table.add_column("change", justify="right", width=12)

    a, b = runs["relay"], runs["shared"]

    def row(label, x, y, fmt="{:,}"):
        delta = f"{(y - x) / x:+.0%}" if x else "n/a"
        table.add_row(label, fmt.format(x), fmt.format(y), delta)

    row("total tokens", a.total_tokens, b.total_tokens)
    row("prompt tokens", a.usage.prompt_tokens, b.usage.prompt_tokens)
    row("completion tokens", a.usage.completion_tokens, b.usage.completion_tokens)
    row("duration (s)", a.duration_s, b.duration_s, "{:.1f}")
    table.add_row(
        "critic verdict",
        str(a.critic_approved),
        str(b.critic_approved),
        "",
    )
    console.print()
    console.print(table)

    console.print(
        Panel(
            "`shared` hands the whole message list forward, so a later stage sees every "
            "tool result directly rather than a summary of it. That is the only way a "
            "critic can tell a quotation from a paraphrase — and it pays lesson 3's "
            "growth across the pipeline instead of within one agent.\n\n"
            "`relay` passes only the previous output, which means every fact a later "
            "stage needs has to be plumbed to it by name. The failure mode is quiet: a "
            "stage works from less than you assumed and nothing reports it.",
            style="cyan",
        )
    )
    return 0


# ---------------------------------------------------------------------------
def show_recursion_guard(client) -> int:
    """Demonstrate the depth cap without spending a token."""
    base = build_registry()
    log = DelegationLog()

    console.print(Rule("the recursion guard"))
    for depth in (0, 1, 2):
        budget = DelegationBudget(depth=depth, max_depth=1)
        registry = build_team_registry(base, TEAM, client, log, budget=budget)
        delegation_tools = [n for n in registry.names if n.startswith("ask_")]
        console.print(
            f"depth {depth}: {len(registry)} tools, delegation tools: "
            f"{delegation_tools or '[dim]none[/dim]'}"
        )

    budget = DelegationBudget(max_calls=2)
    budget.calls = 2
    registry = build_team_registry(base, TEAM, client, log, budget=budget)
    from llmkit import ToolCall

    execution = registry.dispatch(ToolCall(id="x", name="ask_critic", arguments={"task": "hi"}))
    console.print(f"\nwith the call budget used up: [yellow]{execution.result}[/yellow]")
    console.print(f"[dim]failure_kind: {execution.failure_kind}[/dim]")

    console.print(
        Panel(
            "Two limits, stopping two different failures.\n\n"
            "**Depth** stops recursion, and is enforced by withholding the tools rather "
            "than refusing the call. A sub-agent holding delegation tools can call "
            "itself or a peer that calls back, and lesson 3's step cap will not catch "
            "it: from the parent's view that is still one tool call that happens to "
            "take a while.\n\n"
            "**Breadth** stops a coordinator calling six specialists per step, each "
            "costing a full agent run. Depth alone would not notice.\n\n"
            "The budget refusal is a `ToolError`, so the parent reads it as an "
            "observation and answers with what it has. Lesson 2's rule: a tool that "
            "cannot do its job returns a string, it does not raise.",
            style="cyan",
        )
    )
    return 0


# ---------------------------------------------------------------------------
def probe(client, case_id: str, repeats: int, pause: float) -> int:
    """One eval case, solo versus delegating. Lesson 9's discipline applied here.

    A full suite run through a delegating agent costs two to three times the usual
    33,000 tokens. Lesson 9's most decisive findings came from probing single cases at
    a twentieth of that, so this is the command to reach for first.
    """
    import time

    from loop import run_agent
    from scorers import score_case

    case = by_id(case_id)
    base = build_registry()

    console.print(Panel(case.question, title=f"{case_id}  ({case.category.value})", style="blue"))
    console.print(f"[dim]why this case exists: {case.why}[/dim]\n")

    table = Table(title=f"solo vs team on {client.config.model}")
    table.add_column("architecture", width=12)
    table.add_column("run", justify="right", width=4)
    table.add_column("naive", width=7)
    table.add_column("effective", width=10)
    table.add_column("tokens", justify="right", width=8)
    table.add_column("tools (effective)", overflow="fold")
    artifacts: list[str] = []

    for label in ("solo", "team"):
        for index in range(repeats):
            log = DelegationLog()
            if label == "solo":
                registry, prompt = base, None
            else:
                registry = build_team_registry(base, TEAM, client, log)
                prompt = COORDINATOR_PROMPT

            try:
                trajectory = run_agent(
                    client, case.question, registry,
                    max_steps=case.max_steps, system_prompt=prompt,
                )
            except Exception as exc:  # noqa: BLE001
                # A probe that dies halfway tells you nothing about the runs it did
                # finish, and the most likely cause is a rate limit -- which is a fact
                # about the quota, not about the architecture. Lesson 7's harness
                # learned this and records errors as results; this had not, and hit a
                # 429 on its fourth case.
                table.add_row(
                    label if index == 0 else "",
                    str(index + 1),
                    "[yellow]error[/yellow]",
                    "-",
                    "-",
                    f"[yellow]{type(exc).__name__}: {str(exc)[:70]}[/yellow]",
                )
                console.print(table)
                console.print(
                    "\n[yellow]Stopped early.[/yellow] Results above are real; the rest "
                    "did not run."
                )
                return 1

            effective = log.effective_tool_sequence(trajectory.tool_sequence)

            class _Naive:
                """Scored the way lesson 7 would, seeing only the parent's tools."""

                final_answer = trajectory.final_answer
                tool_sequence = trajectory.tool_sequence
                succeeded = trajectory.succeeded

            class _Effective:
                """Scored with delegations expanded into the tools actually used."""

                final_answer = trajectory.final_answer
                tool_sequence = effective
                succeeded = trajectory.succeeded

            naive_passed, _ = score_case(case, _Naive)
            eff_passed, _ = score_case(case, _Effective)
            if naive_passed != eff_passed:
                artifacts.append(f"{label} run {index + 1}")

            total = (
                trajectory.usage.prompt_tokens
                + trajectory.usage.completion_tokens
                + log.total_tokens
            )
            table.add_row(
                label if index == 0 else "",
                str(index + 1),
                "[green]yes[/green]" if naive_passed else "[red]no[/red]",
                "[green]yes[/green]" if eff_passed else "[red]no[/red]",
                f"{total:,}",
                " -> ".join(effective) or "(none)",
            )
            if index < repeats - 1 or label == "solo":
                time.sleep(pause)

    console.print(table)
    console.print(
        Panel(
            "Two columns because one of them lies.\n\n"
            "**naive** scores the way lesson 7 does, from the parent's `tool_sequence`. "
            "**effective** expands each delegation into the tools the sub-agent really "
            "used. They differ whenever work was delegated, because `Trajectory` cannot "
            "see inside a tool call — the same blind spot that hides a sub-agent's "
            "tokens hides its tool use.\n\n"
            "That matters more than it sounds: 10 of the 16 eval cases assert "
            "`used_tools`, and an eleventh asserts `answered_without_tools`. Under "
            "delegation those checks measure the wrong thing, so a full suite run would "
            "report a regression that is largely an artifact of the instrument.\n\n"
            "Token counts are the honest total, parent plus sub-agents. Read off "
            "`Trajectory.usage` alone the team looks *cheaper*."
            + (
                f"\n\n[bold]Measurement artifact on: {', '.join(artifacts)}[/bold] — the "
                f"naive and effective verdicts disagree, so the naive one is wrong."
                if artifacts
                else ""
            ),
            style="cyan",
        )
    )
    return 0


# ---------------------------------------------------------------------------
class _TeamExperiment:
    """Lesson 9's experiment shape, declared here rather than in lesson 9.

    Lesson 10 does not extend lesson 9's `Config` — that would make an earlier lesson
    depend on a later one. Instead it uses lesson 7's `run_eval` (which already takes a
    registry as an argument) and lesson 9's `compare`/`decide`/`Changelog` directly,
    which is a better demonstration anyway: the iteration machinery is reusable by any
    caller, not just by the CLI it shipped with.
    """

    id = "multi_agent_team"
    variable = "architecture: a delegating coordinator with three specialists"
    hypothesis = (
        "A delegating agent will score WORSE and cost MORE on this dataset. The 16 "
        "cases are mostly single-tool questions with no branch that depends on what is "
        "found, so delegation adds a model call, a place to lose information in the "
        "restatement, and a whole agent run per sub-task. Predicted before running: "
        "the point of the experiment is to have the evidence for when NOT to reach for "
        "a second agent."
    )
    predicts_fixed: list[str] = []
    #: Named before the run. Each for a stated reason, which is what makes a wrong
    #: prediction informative: arith_precision needs an exact six-decimal string to
    #: survive being restated by a sub-agent and summarised back; files_quote_definition
    #: needs a verbatim phrase to survive the same round trip; no_tool_definition is a
    #: restraint case and delegation is one more tool to be tempted by.
    predicts_broken = ["arith_precision", "files_quote_definition", "no_tool_definition"]
    predicts_no_change = False
    baseline_run = "baseline"
    run_name = "exp_multi_agent_team"


def evaluate(client, pause: float, force: bool) -> int:
    experiment = _TeamExperiment()
    try:
        baseline = EvalRun.load(experiment.baseline_run)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    estimate = 33_000 * 2.5
    console.print(
        Panel(
            f"[bold]{experiment.id}[/bold]\n\n"
            f"hypothesis: {experiment.hypothesis}\n\n"
            f"variable:   {experiment.variable}\n"
            f"predicts:   broken={experiment.predicts_broken}\n"
            f"cost:       ~{estimate:,.0f} tokens (a delegating run costs roughly the "
            f"sum of its agents)",
            style="bold blue",
        )
    )
    if not force:
        console.print(
            "[yellow]This is the most expensive command in the project.[/yellow] "
            "Probe a couple of cases first (`--probe arith_precision`), then pass "
            "--force to run the full suite."
        )
        return 0

    log = DelegationLog()
    registry = build_team_registry(build_registry(), TEAM, client, log)
    naive_sequences: dict[str, list[str]] = {}

    def correct(trajectory, execution):
        """Repair the record before it is scored.

        Without this the suite reports a regression that is mostly an artifact: the
        coordinator's tool_sequence reads ["ask_calculator"], so used_tools(["calculate"])
        fails on a correct answer. Ten of these sixteen cases assert used_tools. The
        naive sequence is kept alongside so the artifact stays visible rather than being
        quietly papered over.
        """
        naive_sequences[trajectory.question[:60]] = list(execution.tool_sequence)
        from dataclasses import replace as _replace

        return _replace(
            execution,
            tool_sequence=log.effective_tool_sequence(execution.tool_sequence),
        )

    def report(case, result, from_cache) -> None:
        mark = "[green]y[/green]" if result.passed else "[red]n[/red]"
        if result.error:
            mark = "[yellow]![/yellow]"
        tag = " [dim](cached)[/dim]" if from_cache else ""
        console.print(f"  {mark} {case.id:28}{tag}")

    run, cache_hits = run_eval(
        client,
        registry,
        experiment.run_name,
        cases=CASES,
        system_prompt=COORDINATOR_PROMPT,
        pause_between=pause,
        on_progress=report,
        # The registry differs from the baseline's, and lesson 7's cache key cannot see
        # that -- exactly the hole lesson 9 found and fixed. Without this the whole
        # experiment would replay the solo agent's cached runs and report "no change".
        variant_key="team:" + ",".join(s.name for s in TEAM),
        extra_config={
            "experiment": experiment.id,
            "architecture": "delegating coordinator",
            "sub_agents": [s.name for s in TEAM],
            # Recorded because CaseResult cannot see it. Without this line the saved run
            # under-reports its own cost and every later comparison inherits the error.
            "sub_agent_tokens": log.total_tokens,
            "sub_agent_calls": log.calls,
            "scoring": "effective tool sequence (delegations expanded)",
        },
        correct_execution=correct,
    )
    run.config["naive_tool_sequences"] = naive_sequences
    run.save()

    comparison = compare(baseline, run)
    judgement = decide(
        comparison,
        load_noise(),
        unpredicted_breaks=[
            c for c in comparison.broken if c not in experiment.predicts_broken
        ],
    )

    changelog = Changelog.load()
    attempt = build_attempt(experiment, comparison, judgement)
    changelog.append(attempt)
    changelog.save()

    console.print(Rule("result"))
    console.print(
        f"solo {baseline.passed}/{baseline.total}  ->  team {run.passed}/{run.total}"
        f"   (net {comparison.net:+d})"
    )
    if comparison.fixed:
        console.print(f"[green]fixed: {', '.join(comparison.fixed)}[/green]")
    if comparison.broken:
        console.print(f"[red]broke: {', '.join(comparison.broken)}[/red]")

    colour = {Decision.KEEP: "green", Decision.REVERT: "red"}.get(judgement.decision, "yellow")
    console.print(f"\n[bold {colour}]{judgement.decision.value.upper()}[/bold {colour}]")
    for reason in judgement.reasons:
        console.print(f"  - {reason}")
    for caveat in judgement.caveats:
        console.print(f"  [yellow]still unknown:[/yellow] {caveat}")

    console.print(
        f"\nprediction: {'[green]hit[/green]' if attempt.prediction_hit else '[red]miss[/red]'}"
    )
    console.print(f"  predicted broken: {experiment.predicts_broken}")
    console.print(f"  actual broken:    {comparison.broken or '[]'}")

    parent_tokens = run.total_tokens
    console.print(
        Panel(
            f"Cost, stated honestly: the run file records {parent_tokens:,} tokens, and "
            f"the sub-agents spent a further {log.total_tokens:,} across {log.calls} "
            f"calls that `CaseResult` cannot see. The real total is "
            f"{parent_tokens + log.total_tokens:,}.\n\n"
            f"So every comparison in the table above understates the team's cost, and "
            f"`--cost-compare` in lesson 8 would too. Recorded in the run's config so "
            f"at least it is not lost.",
            style="yellow",
        )
    )
    console.print(f"[dim]appended to lesson 9's attempts.json ({cache_hits} cached)[/dim]")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-agent: delegation, handoff, and whether they are worth it."
    )
    parser.add_argument("--team", action="store_true", help="Show the wiring. Free.")
    parser.add_argument("--ask", metavar="QUESTION", help="Run the delegating coordinator.")
    parser.add_argument("--router", action="store_true",
                        help="With --ask: strip the coordinator's own tools.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--pipeline", metavar="TOPIC", help="Run research/write/critique.")
    parser.add_argument("--mode", choices=("relay", "shared"), default="relay")
    parser.add_argument("--revise", action="store_true", help="Add a revision stage.")
    parser.add_argument("--compare", metavar="TOPIC", help="relay vs shared on one topic.")
    parser.add_argument("--recursion", action="store_true", help="Show the depth guard. Free.")
    parser.add_argument("--probe", metavar="CASE_ID", help="One eval case, solo vs team.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats for --probe.")
    parser.add_argument("--evaluate", action="store_true", help="The full 16-case comparison.")
    parser.add_argument("--force", action="store_true", help="Confirm the expensive run.")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--model", help="Override the model.")
    args = parser.parse_args()

    if args.team and not args.model:
        return show_team()

    try:
        client = get_client(model=args.model or EVAL_MODEL)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if args.team:
        return show_team()
    if args.recursion:
        return show_recursion_guard(client)
    if args.ask:
        return ask(client, args.ask, args.max_steps, keep_own=not args.router)
    if args.pipeline:
        return pipeline(client, args.pipeline, args.mode, args.revise)
    if args.compare:
        return compare_modes(client, args.compare)
    if args.probe:
        return probe(client, args.probe, max(args.repeats, 1), args.pause)
    if args.evaluate:
        return evaluate(client, args.pause, args.force)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
