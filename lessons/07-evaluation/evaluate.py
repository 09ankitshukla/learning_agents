"""Lesson 7 deliverable: an eval harness with a scorecard.

    uv run lessons/07-evaluation/evaluate.py --list
    uv run lessons/07-evaluation/evaluate.py --run baseline
    uv run lessons/07-evaluation/evaluate.py --show baseline
    uv run lessons/07-evaluation/evaluate.py --run strict --prompt strict
    uv run lessons/07-evaluation/evaluate.py --compare baseline strict
    uv run lessons/07-evaluation/evaluate.py --case arith_large_product

Cost note: a full 12-case run is roughly 40,000 tokens against a 200,000/day
allowance. Results are cached per case, so re-running the same configuration is
free. Use `--model openai/gpt-oss-20b` to spend a different quota.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llmkit import ConfigError, console, get_client

_LESSONS = Path(__file__).resolve().parents[1]
for _folder in ("02-tool-calling", "03-agent-loop"):
    _path = str(_LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from toolset import build_registry  # noqa: E402

from dataset import CASES, by_id, summary  # noqa: E402
from harness import (  # noqa: E402
    EvalRun,
    clear_cache,
    compare,
    run_eval,
    wilson_interval,
)

# Prompt variants, so the harness has something real to compare.
PROMPTS: dict[str, str | None] = {
    "default": None,  # lesson 3's DEFAULT_SYSTEM_PROMPT
    "strict": (
        "You solve problems using tools.\n"
        "Always use a tool for arithmetic, times, currency conversion and file "
        "access. Never compute or recall a value you could look up exactly.\n"
        "If no tool can answer the question, say plainly that you cannot and stop. "
        "Never guess a number you have not obtained from a tool.\n"
        "When you have what you need, answer concisely."
    ),
    "terse": (
        "Answer using tools where useful. Be brief."
    ),
}


# ---------------------------------------------------------------------------
def print_scorecard(run: EvalRun, cache_hits: int = 0) -> None:
    console.print(Rule(f"Scorecard: {run.name}"))

    table = Table(show_lines=False)
    table.add_column("case", width=26)
    table.add_column("cat", width=11)
    table.add_column("ok", width=4)
    table.add_column("tools used", width=30, overflow="fold")
    table.add_column("steps", width=6, justify="right")
    table.add_column("tokens", width=8, justify="right")

    for result in run.results:
        mark = "[green]y[/green]" if result.passed else "[red]n[/red]"
        if result.error:
            mark = "[yellow]![/yellow]"
        table.add_row(
            result.case_id,
            result.category,
            mark,
            " -> ".join(result.tool_sequence) or "[dim]none[/dim]",
            str(result.steps),
            f"{result.total_tokens:,}",
        )
    console.print(table)

    # -- aggregate metrics ------------------------------------------------
    low, high = wilson_interval(run.passed, run.total)
    metrics = Table(show_header=False, box=None)
    metrics.add_column("metric", style="cyan", width=24)
    metrics.add_column("value")
    metrics.add_row("task success", f"{run.passed}/{run.total}  ({run.success_rate:.0%})")
    metrics.add_row("95% interval", f"{low:.0%} to {high:.0%}")
    metrics.add_row("tool-choice accuracy", f"{run.tool_choice_accuracy:.0%}")
    metrics.add_row("mean steps", f"{run.mean_steps:.1f}")
    metrics.add_row("total tokens", f"{run.total_tokens:,}")
    metrics.add_row("model", f"{run.provider}:{run.model}")
    if cache_hits:
        metrics.add_row("from cache", f"{cache_hits}/{run.total} cases (no tokens spent)")
    console.print(metrics)

    # -- per category -----------------------------------------------------
    cats = Table(title="by category")
    cats.add_column("category", width=12)
    cats.add_column("passed", width=10)
    for category, (passed, total) in sorted(run.by_category().items()):
        style = "green" if passed == total else ("red" if passed == 0 else "yellow")
        cats.add_row(category, f"[{style}]{passed}/{total}[/{style}]")
    console.print(cats)
    console.print(
        "[dim]Aggregates hide structure. An agent can look 75% accurate overall while "
        "failing every case in one category, which is a completely different problem "
        "from failing a scattered quarter of everything.[/dim]"
    )

    # -- failures ---------------------------------------------------------
    failures = [r for r in run.results if not r.passed]
    if failures:
        console.print(Rule("Failures", style="red"))
        for result in failures:
            console.print(f"[red]{result.case_id}[/red]  ({result.stop_reason})")
            if result.error:
                console.print(f"  [yellow]error: {result.error}[/yellow]")
            for score in result.scores:
                if not score["passed"]:
                    console.print(f"  [red]x[/red] {score['name']}: {score['detail']}")
            if result.answer:
                console.print(f"  [dim]answer: {result.answer[:160]}[/dim]")
            console.print()

    console.print(
        Panel(
            f"[bold]{run.passed}/{run.total} is not a precise number.[/bold] The 95% interval is "
            f"{low:.0%} to {high:.0%} -- with {run.total} cases, one case is "
            f"{1 / run.total:.0%}. This suite can tell a working agent from a broken "
            f"one. It cannot tell 70% from 80%.\n\n"
            "Read the failures rather than the headline. That is where the "
            "information is, and it is what tells you whether a change helped.",
            title="how to read this",
            style="yellow",
        )
    )


def print_comparison(comparison) -> None:
    baseline, candidate = comparison.baseline, comparison.candidate
    console.print(Rule(f"{baseline.name} -> {candidate.name}"))

    for warning in comparison.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if comparison.warnings:
        console.print()

    table = Table()
    table.add_column("metric", width=22)
    table.add_column(baseline.name, width=16, justify="right")
    table.add_column(candidate.name, width=16, justify="right")
    table.add_column("change", width=12, justify="right")

    def row(label, a, b, fmt="{:.0%}"):
        delta = b - a
        arrow = "same" if abs(delta) < 1e-9 else ("up" if delta > 0 else "down")
        table.add_row(label, fmt.format(a), fmt.format(b), f"{arrow} {fmt.format(abs(delta))}")

    row("task success", baseline.success_rate, candidate.success_rate)
    row("tool-choice accuracy", baseline.tool_choice_accuracy, candidate.tool_choice_accuracy)
    row("mean steps", baseline.mean_steps, candidate.mean_steps, "{:.1f}")
    row("total tokens", baseline.total_tokens, candidate.total_tokens, "{:,.0f}")
    console.print(table)

    console.print(f"\n[bold]verdict: {comparison.verdict}[/bold]  (net {comparison.net:+d} cases)")

    if comparison.fixed:
        console.print(f"\n[green]fixed ({len(comparison.fixed)})[/green]")
        for case_id in comparison.fixed:
            console.print(f"  + {case_id}")
    if comparison.broken:
        console.print(f"\n[red]broken ({len(comparison.broken)})[/red]")
        for case_id in comparison.broken:
            result = candidate.result_for(case_id)
            console.print(f"  - {case_id}: {', '.join(result.failed_scorers)}")
    if comparison.still_failing:
        console.print(f"\n[dim]still failing ({len(comparison.still_failing)}): "
                      f"{', '.join(comparison.still_failing)}[/dim]")

    console.print(
        Panel(
            "The per-case diff is the point, not the average.\n\n"
            "An aggregate can rise while specific cases regress, and the regressions "
            "are usually what you care about -- especially the security and "
            "fabrication cases, where a loss is not offset by two wins elsewhere. A "
            "harness that only printed '68% -> 74%' would hide exactly that.\n\n"
            "Treat a net change of one case as noise. Look at *which* cases moved.",
            style="cyan",
        )
    )


# ---------------------------------------------------------------------------
def make_progress_printer():
    def report(case, result, from_cache) -> None:
        mark = "[green]y[/green]" if result.passed else "[red]n[/red]"
        if result.error:
            mark = "[yellow]![/yellow]"
        tag = " [dim](cached)[/dim]" if from_cache else ""
        console.print(f"  {mark} {case.id:28}{tag}")
        if result.error:
            console.print(f"      [yellow]{result.error[:120]}[/yellow]")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the agent against a labelled dataset.")
    parser.add_argument("--run", metavar="NAME", help="Run the suite and save it under NAME.")
    parser.add_argument("--show", metavar="NAME", help="Print a saved run's scorecard.")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--list", action="store_true", help="List dataset and saved runs.")
    parser.add_argument("--case", metavar="ID", help="Run a single case (useful while iterating).")
    parser.add_argument(
        "--prompt", choices=sorted(PROMPTS), default="default", help="System prompt variant."
    )
    parser.add_argument("--model", help="Override the model for this run.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached case results.")
    parser.add_argument("--clear-cache", action="store_true", help="Delete the case cache.")
    parser.add_argument(
        "--pause",
        type=float,
        default=3.0,
        help="Seconds between live cases, to stay under the per-minute token ceiling.",
    )
    args = parser.parse_args()

    if args.clear_cache:
        console.print(f"[dim]cleared {clear_cache()} cached case result(s)[/dim]")
        if not (args.run or args.case):
            return 0

    if args.list:
        table = Table(title=f"dataset: {summary()}")
        table.add_column("id", style="cyan", width=26)
        table.add_column("category", width=11)
        table.add_column("why it exists", overflow="fold")
        for case in CASES:
            table.add_row(case.id, case.category.value, case.why)
        console.print(table)
        console.print(f"\nsaved runs: {EvalRun.list_runs() or '(none yet)'}")
        return 0

    if args.show:
        try:
            print_scorecard(EvalRun.load(args.show))
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        return 0

    if args.compare:
        try:
            baseline = EvalRun.load(args.compare[0])
            candidate = EvalRun.load(args.compare[1])
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        print_comparison(compare(baseline, candidate))
        return 0

    # Everything below needs a model.
    try:
        client = get_client(model=args.model) if args.model else get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    registry = build_registry()
    system_prompt = PROMPTS[args.prompt]

    console.print(
        Panel.fit(
            f"{client.config.describe()}  |  prompt: {args.prompt}  |  {summary()}",
            style="bold cyan",
        )
    )

    if args.case:
        from harness import run_case

        try:
            case = by_id(args.case)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1

        console.print(Panel(case.question, title=case.id, style="blue"))
        result, cached = run_case(
            client,
            registry,
            case,
            system_prompt=system_prompt,
            use_cache=not args.no_cache,
        )
        console.print(f"\npassed: {'[green]yes[/green]' if result.passed else '[red]no[/red]'}"
                      f"{' [dim](cached)[/dim]' if cached else ''}")
        console.print(f"tools: {' -> '.join(result.tool_sequence) or '(none)'}")
        console.print(f"stop_reason: {result.stop_reason}, steps: {result.steps}")
        for score in result.scores:
            mark = "[green]y[/green]" if score["passed"] else "[red]n[/red]"
            console.print(f"  {mark} {score['name']}: {score['detail']}")
        if result.answer:
            console.print(Panel(result.answer[:800], title="answer", style="dim"))
        return 0

    if not args.run:
        parser.print_help()
        return 0

    console.print(f"[dim]running {len(CASES)} cases (cached results reused)...[/dim]")
    run, cache_hits = run_eval(
        client,
        registry,
        name=args.run,
        system_prompt=system_prompt,
        use_cache=not args.no_cache,
        pause_between=args.pause,
        on_progress=make_progress_printer(),
    )
    run.config["prompt_variant"] = args.prompt
    path = run.save()

    console.print()
    print_scorecard(run, cache_hits)
    console.print(f"\n[dim]saved to {path}[/dim]")
    if run.errors:
        console.print(
            f"[yellow]{len(run.errors)} case(s) errored (likely rate limits). "
            f"Re-run the same command -- cached successes will not be re-spent.[/yellow]"
        )
    console.print(
        f"[dim]compare with: uv run lessons/07-evaluation/evaluate.py "
        f"--compare {args.run} OTHER_RUN[/dim]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
