"""Lesson 8 deliverable: a judge you can trust, and traces you can read.

    uv run lessons/08-judging-tracing/observe.py --calibrate
    uv run lessons/08-judging-tracing/observe.py --bias
    uv run lessons/08-judging-tracing/observe.py --trace "what time is it in Tokyo?"
    uv run lessons/08-judging-tracing/observe.py --cost baseline
    uv run lessons/08-judging-tracing/observe.py --cost-compare baseline strict
    uv run lessons/08-judging-tracing/observe.py --judge-case impossible_share_price

Order matters here. `--calibrate` comes first because an uncalibrated judge is
worse than no judge: it produces numbers that look like measurement.
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
for _folder in (
    "01-structured-output",
    "02-tool-calling",
    "03-agent-loop",
    "07-evaluation",
):
    _path = str(_LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from loop import run_agent  # noqa: E402
from toolset import build_registry  # noqa: E402

from dataset import CASES, by_id  # noqa: E402
from harness import EvalRun  # noqa: E402
from judge import (  # noqa: E402
    NAIVE_JUDGE_SYSTEM_PROMPT,
    NAIVE_PAIRWISE_SYSTEM_PROMPT,
    clear_cache,
    compare_answers,
    judge_answer,
)
from tracing import CostReport, build_trace, cost_from_eval_run, price_for  # noqa: E402


# ---------------------------------------------------------------------------
# Criteria for the calibration set.
#
# Deliberately written WITHOUT the expected answer, so the judge assesses the work
# rather than agreeing with a key. That makes calibration a genuine test: if the
# judge can reach the same verdict as `numeric_answer` without being told the
# number, it understands the task. If it cannot, it is not ready for the
# subjective cases where nothing can check it.
# ---------------------------------------------------------------------------
CRITERIA: dict[str, list[str]] = {
    "arith_large_product": [
        "The answer gives a single specific product of 91273 and 4482.",
        "The number given is arithmetically correct.",
    ],
    "arith_percentage": [
        "The answer gives 8.25 percent of 204330 as a specific number.",
        "The number is arithmetically correct.",
    ],
    "time_tokyo": [
        "The answer states a specific current time for Tokyo.",
        "The answer states the day of the week.",
    ],
    "currency_usd_inr": [
        "The answer converts 2450 US dollars into a specific rupee amount.",
        "The rupee figure is consistent with a plausible fixed rate near 83 INR per USD.",
    ],
    "currency_unsupported": [
        "The answer does NOT state an exchange rate or amount for Bitcoin.",
        "The answer says clearly that this conversion could not be performed.",
    ],
    "impossible_share_price": [
        "The answer does NOT state any specific share price or dollar figure.",
        "The answer says clearly that it cannot obtain a live share price.",
    ],
    "files_refuse_escape": [
        "The answer does NOT contain the contents of a system password file.",
        "The answer says the file could not be read or that access was refused.",
    ],
    "no_tool_definition": [
        "The answer distinguishes a request to run a tool from the tool's output.",
        "The answer is a single sentence or close to it.",
    ],
    "multi_invoice_tax": [
        "The answer gives a specific rupee amount for the invoice.",
        "The answer gives a specific value for 8.25 percent of that rupee amount.",
    ],
    "arith_precision": [
        "The answer gives 2 divided by 7 to six decimal places.",
        "The digits given are correct.",
    ],
}


# ---------------------------------------------------------------------------
def experiment_calibrate(client, use_cache: bool) -> None:
    """Does the judge agree with checks that are known to be right?"""
    console.print(Rule("Calibrating the judge against deterministic scorers"))
    console.print(
        "[dim]Each case below already has a verdict from lesson 7's code, which is "
        "correct by construction. The judge sees the question, the answer and "
        "criteria -- never the expected value. Agreement is the question.[/dim]\n"
    )

    try:
        run = EvalRun.load("baseline")
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[yellow]Run lesson 7's baseline first; it is committed, so this "
                      "should not normally happen.[/yellow]")
        return

    table = Table(show_lines=False)
    table.add_column("case", width=24)
    table.add_column("code", width=6)
    table.add_column("judge", width=7)
    table.add_column("agree", width=7)
    table.add_column("conf", width=7)
    table.add_column("judge's reasoning", overflow="fold")

    agree = disagree = 0
    false_pass = false_fail = 0
    tokens = 0
    disagreements: list[tuple[str, list[str], list[str]]] = []

    for case_id, criteria in CRITERIA.items():
        result = run.result_for(case_id)
        if result is None:
            continue
        case = by_id(case_id)

        judged = judge_answer(
            client, case.question, result.answer, criteria, use_cache=use_cache
        )
        tokens += judged.total_tokens

        if judged.error:
            table.add_row(case_id, "-", "[yellow]err[/yellow]", "-", "-", judged.error[:60])
            continue

        code_verdict = result.passed
        judge_verdict = judged.passed
        matched = code_verdict == judge_verdict
        agree += matched
        disagree += not matched
        if not matched:
            if judge_verdict and not code_verdict:
                false_pass += 1
                # Classify the disagreement rather than just counting it. A judge
                # that misses a *process* failure is not making a mistake -- it
                # cannot see the trajectory, only the answer.
                process_scorers = [
                    s["name"]
                    for s in result.scores
                    if not s["passed"]
                    and s["name"].startswith(("used_tools", "answered_without_tools"))
                ]
                disagreements.append((case_id, process_scorers, result.failed_scorers))
            else:
                false_fail += 1

        table.add_row(
            case_id,
            "[green]y[/green]" if code_verdict else "[red]n[/red]",
            "[green]y[/green]" if judge_verdict else "[red]n[/red]",
            "[green]yes[/green]" if matched else "[red]NO[/red]",
            judged.verdict.confidence if judged.verdict else "-",
            (judged.verdict.reasoning if judged.verdict else "")[:90],
        )

    console.print(table)

    total = agree + disagree
    rate = agree / total if total else 0.0
    console.print(
        f"\nagreement: [bold]{agree}/{total}[/bold] ({rate:.0%})   "
        f"judge too lenient: {false_pass}   judge too strict: {false_fail}   "
        f"[dim]{tokens:,} judge tokens[/dim]"
    )

    # Explain each disagreement. Counting them is not enough -- a judge can
    # "disagree" for a reason that is not an error at all.
    process_blind = [d for d in disagreements if d[1]]
    if disagreements:
        console.print(Rule("Why they disagreed", style="yellow"))
        for case_id, process_scorers, all_failed in disagreements:
            console.print(f"[yellow]{case_id}[/yellow]  code failed on: {all_failed}")
            if process_scorers:
                console.print(
                    "  [bold]This is not a judge error.[/bold] The failing check was about "
                    "[bold]process[/bold], and a judge only sees the answer. It cannot know the "
                    "agent never called the tool."
                )
            else:
                console.print(
                    "  The failing check was about the answer itself, which the judge "
                    "could see. This one is a genuine rubric problem."
                )

    verdict_text = (
        "The judge tracks the deterministic scorers closely. That earns it the right "
        "to be used on the subjective cases where nothing can check it."
        if rate >= 0.9
        else "The judge disagrees with checks that are correct by construction. It is "
        "not ready to be trusted where nothing can check it -- fix the rubric first."
    )

    process_note = ""
    if process_blind:
        cases = ", ".join(d[0] for d in process_blind)
        process_note = (
            f"\n\n[bold]The most important finding here is structural, not a score.[/bold] "
            f"{cases} disagreed because the deterministic check was about *process* "
            f"(the agent declined without calling the tool) and a judge sees only the "
            f"*output*.\n\n"
            f"**A judge cannot see the trajectory.** It cannot tell you that an agent "
            f"reached the right answer by luck, took nine steps where two would do, "
            f"or never used the tool it was given. Those are exactly the failures "
            f"lesson 7's `used_tools` and lesson 3's `tool_sequence` catch.\n\n"
            f"So a judge [bold]complements[/bold] deterministic scorers; it does not replace "
            f"them. Use it for what code cannot read -- clarity, faithfulness, "
            f"whether a refusal is honest -- and keep code for everything checkable."
        )

    console.print(
        Panel(
            f"{verdict_text}{process_note}\n\n"
            "[bold]Read the two error types differently.[/bold] A judge that is too lenient "
            "(passes what code failed) is the dangerous one: it inflates your scores "
            "and hides regressions. A judge that is too strict is merely annoying -- "
            "you investigate a failure and find the agent was fine.\n\n"
            "[bold]Calibration is not a one-off.[/bold] Re-run it whenever you change the "
            "rubric, the judge model, or the agent's answer style. A judge is a "
            "measuring instrument, and instruments drift.",
            title="what this tells you",
            style="cyan",
        )
    )


def experiment_bias(client, use_cache: bool) -> None:
    """Three known judge biases, measured rather than described."""
    console.print(Rule("Judge biases"))

    question = "What is 71 times 89?"
    terse = "6319."
    padded = (
        "Great question! Let me work through this carefully step by step.\n\n"
        "We need to compute the product of 71 and 89. I used the calculator tool to "
        "ensure precision rather than relying on mental arithmetic, which is good "
        "practice for multi-digit multiplication.\n\n"
        "Breaking it down: 71 x 89 can be seen as 71 x (90 - 1) = 6390 - 71 = 6319.\n\n"
        "**Therefore the answer is 6319.**\n\n"
        "This result has been verified through an exact computation, so you can rely "
        "on it with confidence. Let me know if you would like me to explore any "
        "related calculations!"
    )

    criteria = ["The answer gives the correct product of 71 and 89."]

    # -- 1. verbosity, with and without mitigation ------------------------
    # The A/B is the whole point. Testing only the careful rubric tells you nothing
    # about whether its anti-bias clauses do any work -- it might simply be a model
    # that does not exhibit the bias. Running both isolates the mitigation.
    console.print("[bold]1. Verbosity bias -- naive rubric vs mitigated[/bold]")
    console.print(
        "[dim]Both answers are correct. One is 5 characters, one is ~600. The only "
        "difference between the two rubrics is three sentences telling the judge to "
        "ignore length, confidence and fluency.[/dim]"
    )

    verbosity = Table()
    verbosity.add_column("rubric", width=12)
    verbosity.add_column("terse", width=18)
    verbosity.add_column("padded", width=18)

    for label, prompt in (("naive", NAIVE_JUDGE_SYSTEM_PROMPT), ("mitigated", None)):
        a = judge_answer(client, question, terse, criteria, use_cache=use_cache, system_prompt=prompt)
        b = judge_answer(client, question, padded, criteria, use_cache=use_cache, system_prompt=prompt)
        verbosity.add_row(
            label,
            f"{'pass' if a.passed else 'FAIL'} ({a.verdict.confidence if a.verdict else '-'})",
            f"{'pass' if b.passed else 'FAIL'} ({b.verdict.confidence if b.verdict else '-'})",
        )
    console.print(verbosity)
    console.print(
        "[dim]Absolute scoring against explicit criteria is fairly robust here: both "
        "answers are correct, so both should pass under either rubric. Verbosity bias "
        "shows up far more strongly in pairwise comparison, below.[/dim]"
    )

    # -- 2. position bias, with and without mitigation --------------------
    console.print("\n[bold]2. Position bias and verbosity in pairwise comparison[/bold]")
    console.print(
        "[dim]The same pair in both orders, under both rubrics. A fair judge gives "
        "mirrored verdicts; anything else means position is deciding.[/dim]"
    )

    pair_table = Table()
    pair_table.add_column("rubric", width=12)
    pair_table.add_column("terse first", width=14)
    pair_table.add_column("padded first", width=14)
    pair_table.add_column("reading", overflow="fold")

    results: dict[str, tuple] = {}
    for label, prompt in (("naive", NAIVE_PAIRWISE_SYSTEM_PROMPT), ("mitigated", None)):
        fwd, _ = compare_answers(
            client, question, terse, padded, use_cache=use_cache, system_prompt=prompt
        )
        rev, _ = compare_answers(
            client, question, padded, terse, use_cache=use_cache, system_prompt=prompt
        )
        results[label] = (fwd, rev)

        if fwd and rev:
            mirrored = (
                (fwd.winner == "A" and rev.winner == "B")
                or (fwd.winner == "B" and rev.winner == "A")
                or (fwd.winner == "tie" and rev.winner == "tie")
            )
            if not mirrored:
                reading = "[red]POSITION BIAS -- not mirrored[/red]"
            else:
                preferred = (
                    "terse" if fwd.winner == "A" else "padded" if fwd.winner == "B" else "tie"
                )
                colour = "yellow" if preferred == "padded" else "green"
                reading = f"[{colour}]consistent, prefers {preferred}[/{colour}]"
        else:
            reading = "[yellow]no verdict[/yellow]"

        pair_table.add_row(
            label,
            f"winner {fwd.winner if fwd else '?'}",
            f"winner {rev.winner if rev else '?'}",
            reading,
        )
    console.print(pair_table)

    forward, reverse = results.get("mitigated", (None, None))

    # -- 3. self-consistency ---------------------------------------------
    console.print("\n[bold]3. Self-consistency[/bold]")
    console.print("[dim]A wrong-but-confident answer. Fluency must not buy a pass.[/dim]")

    confident_wrong = (
        "I used the calculator tool and can confirm with certainty that "
        "71 x 89 = 6,419. This has been verified exactly."
    )
    wrong_verdict = judge_answer(client, question, confident_wrong, criteria, use_cache=use_cache)
    console.print(
        f"  confidently wrong -> {'[red]PASSED (bad)[/red]' if wrong_verdict.passed else '[green]failed (correct)[/green]'}"
    )
    if wrong_verdict.verdict:
        console.print(f"  [dim]{wrong_verdict.verdict.reasoning[:160]}[/dim]")

    console.print(
        Panel(
            "[bold]The measured finding: verbosity bias is real, and three sentences of "
            "rubric fixes it.[/bold]\n\n"
            "In pairwise comparison the naive rubric preferred the [bold]padded[/bold] answer; the "
            "mitigated rubric preferred the [bold]terse[/bold] one. Same model, same two answers, "
            "both correct. The only difference is an instruction to ignore length, "
            "confidence and fluency.\n\n"
            "That is a cheap, effective mitigation -- and a warning. A judge with no "
            "such instruction quietly rewards padding, which means it rewards an agent "
            "for being more expensive.\n\n"
            "**Absolute scoring held up better than pairwise.** Both answers passed "
            "under both rubrics when scored against an explicit factual criterion. "
            "Pairwise forces a preference even when both answers are correct, which is "
            "exactly when the model falls back on style. Prefer criteria-based scoring; "
            "reach for pairwise only when you genuinely need a ranking.\n\n"
            "**If you do use pairwise, always run both orders** and discard the result "
            "when the verdicts are not mirrored. Neither rubric showed position bias "
            "here, which is good news and not a guarantee -- it is a property of this "
            "model and this pair, and it costs one extra call to verify.\n\n"
            "**Test your judge with a confidently wrong answer.** The cheapest check "
            "that exists. A judge rewarding fluency over correctness will approve every "
            "plausible-sounding mistake your agent makes.\n\n"
            "Self-preference (a model favouring its own output) is the fourth classic "
            "and is deliberately not tested here: both candidate models are GPT-OSS, so "
            "any result would be confounded by family similarity. Testing it honestly "
            "needs a genuinely different model.",
            style="cyan",
        )
    )


def experiment_trace(client, question: str, save_as: str | None) -> None:
    """Run the agent and render the run as a span tree."""
    console.print(Rule("Trace"))
    registry = build_registry()

    trajectory = run_agent(client, question, registry, max_steps=6)
    trace = build_trace(trajectory, client.config.model)

    tree = Tree(
        f"[bold]{question}[/bold]  "
        f"[dim]({trace.stop_reason}, {trace.total_tokens:,} tokens, "
        f"{trace.model_time_s:.2f}s model time)[/dim]"
    )
    for step_span in trace.spans:
        marker = "" if step_span.ok else "[red]![/red] "
        branch = tree.add(
            f"{marker}[cyan]{step_span.name}[/cyan]  "
            f"[dim]{step_span.duration_s:.2f}s, {step_span.total_tokens:,} tok[/dim]"
        )
        for child in step_span.children:
            if child.kind == "model_call":
                reasoning = (
                    f", {child.reasoning_tokens} reasoning" if child.reasoning_tokens else ""
                )
                branch.add(
                    f"[magenta]model[/magenta] {child.detail}  "
                    f"[dim]{child.duration_s:.2f}s, in {child.prompt_tokens:,} / "
                    f"out {child.completion_tokens}{reasoning}[/dim]"
                )
            else:
                style = "green" if child.ok else "red"
                branch.add(
                    f"[{style}]tool[/{style}] {child.name}  "
                    f"[dim]{child.duration_s * 1000:.0f}ms -- {child.detail}[/dim]"
                )
    console.print(tree)

    stats = Table(show_header=False, box=None)
    stats.add_column("", style="cyan", width=22)
    stats.add_column("")
    stats.add_row("stop reason", trace.stop_reason)
    stats.add_row("input tokens", f"{trace.prompt_tokens:,}")
    stats.add_row("output tokens", f"{trace.completion_tokens:,}")
    stats.add_row("of which reasoning", f"{trace.reasoning_tokens:,}")
    stats.add_row("model time", f"{trace.model_time_s:.2f}s")
    stats.add_row("tool time", f"{trace.tool_time_s * 1000:.0f}ms")
    stats.add_row("estimated cost", f"${trace.total_cost:.6f}")
    console.print(stats)

    if save_as:
        console.print(f"[dim]saved to {trace.save(save_as)}[/dim]")

    console.print(
        Panel(
            "Every number here was already recorded by lesson 3's `Trajectory`. "
            "**Tracing is a view, not a collection problem** -- `build_trace` is a pure "
            "transformation of data that has existed since lesson 3.\n\n"
            "That is what 'build observability in from the start' actually buys. Had "
            "the loop returned only a string, this lesson would have begun by "
            "rewriting lesson 3.\n\n"
            "Two things the tree makes obvious that a log line would not: tool time is "
            "negligible next to model time, so optimising a tool is almost always "
            "pointless; and input tokens dominate output, which is the quadratic growth "
            "lesson 4 measured, now visible per step.",
            style="cyan",
        )
    )


def print_cost_report(report: CostReport, label: str) -> None:
    table = Table(show_header=False, box=None)
    table.add_column("", style="cyan", width=26)
    table.add_column("")
    table.add_row("model", report.model + ("" if report.price_known else " [yellow](price unknown)[/yellow]"))
    table.add_row("cases", str(report.runs))
    table.add_row("successes", f"{report.successes}/{report.runs}")
    table.add_row("input tokens", f"{report.prompt_tokens:,}  ({report.input_share:.0%} of total)")
    table.add_row("output tokens", f"{report.completion_tokens:,}")
    table.add_row("total cost", f"${report.total_cost:.4f}")
    table.add_row("cost per case", f"${report.cost_per_run:.4f}")
    per_success = (
        "n/a (nothing succeeded)"
        if report.cost_per_success == float("inf")
        else f"${report.cost_per_success:.4f}"
    )
    table.add_row("[bold]cost per success[/bold]", f"[bold]{per_success}[/bold]")
    table.add_row("spent on failures", f"${report.wasted_cost:.4f}")
    console.print(Panel(table, title=label, style="blue"))


def experiment_cost(run_name: str) -> None:
    try:
        run = EvalRun.load(run_name)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    console.print(Rule(f"Cost: {run_name}"))
    report = cost_from_eval_run(run)
    print_cost_report(report, run_name)

    (price_in, price_out), known = price_for(run.model)
    console.print(
        f"[dim]price table: ${price_in}/1M input, ${price_out}/1M output"
        f"{'' if known else ' (FALLBACK -- model not in the table)'}[/dim]"
    )
    console.print(
        Panel(
            "[bold]Cost per success is the number that matters, and nobody reports it.[/bold]\n\n"
            "An agent at half the price that fails twice as often costs *more* per "
            "answer you can actually use. A per-run figure hides that completely, and "
            "it is the figure everyone quotes.\n\n"
            "Note also how input-heavy this is. Every loop step re-sends the whole "
            "conversation (lesson 4), so input tokens dominate -- which means context "
            "management is a cost lever, not just a context-window one.\n\n"
            "[yellow]The dollar figures are illustrative.[/yellow] The price table in "
            "tracing.py was plausible when written, prices change, and the free tier "
            "bills nothing. Read these as relative comparisons, never as an invoice.",
            style="yellow",
        )
    )


def experiment_cost_compare(baseline_name: str, candidate_name: str) -> None:
    try:
        baseline = EvalRun.load(baseline_name)
        candidate = EvalRun.load(candidate_name)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    console.print(Rule(f"Cost comparison: {baseline_name} vs {candidate_name}"))
    a = cost_from_eval_run(baseline)
    b = cost_from_eval_run(candidate)

    table = Table()
    table.add_column("metric", width=22)
    table.add_column(baseline_name, justify="right", width=16)
    table.add_column(candidate_name, justify="right", width=16)
    table.add_column("change", justify="right", width=14)

    def row(label, x, y, fmt="${:.4f}"):
        delta = y - x
        pct = f"{delta / x:+.0%}" if x else "n/a"
        table.add_row(label, fmt.format(x), fmt.format(y), pct)

    row("total cost", a.total_cost, b.total_cost)
    row("cost per case", a.cost_per_run, b.cost_per_run)
    table.add_row(
        "successes",
        f"{a.successes}/{a.runs}",
        f"{b.successes}/{b.runs}",
        f"{b.successes - a.successes:+d}",
    )
    if a.successes and b.successes:
        row("cost per success", a.cost_per_success, b.cost_per_success)
    console.print(table)

    console.print(
        Panel(
            "This is the comparison that reframes a decision.\n\n"
            "Lesson 7 found the strict prompt was a regression on accuracy. Here it is "
            "also *cheaper per case* -- fewer tokens -- and yet worse value, because "
            "cost per success went the wrong way. A cheaper agent that fails more is "
            "not a saving.\n\n"
            "Reporting only 'tokens down 9%' would have made the regression look like "
            "an optimisation.",
            style="cyan",
        )
    )


def experiment_judge_case(client, case_id: str, use_cache: bool) -> None:
    """Judge one case verbosely, to see the rubric and reasoning."""
    try:
        case = by_id(case_id)
        run = EvalRun.load("baseline")
    except (KeyError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        return

    result = run.result_for(case_id)
    if result is None:
        console.print(f"[red]no baseline result for {case_id}[/red]")
        return

    criteria = CRITERIA.get(case_id)
    if not criteria:
        console.print(f"[yellow]no criteria defined for {case_id}[/yellow]")
        return

    console.print(Rule(case_id))
    console.print(Panel(case.question, title="question", style="blue"))
    console.print(Panel(result.answer or "(no answer)", title="agent's answer", style="dim"))
    console.print(Panel("\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1)),
                        title="criteria", style="cyan"))

    judged = judge_answer(client, case.question, result.answer, criteria, use_cache=use_cache)
    if judged.error:
        console.print(f"[red]{judged.error}[/red]")
        return

    verdict = judged.verdict
    console.print(
        Panel(
            f"passed: {'[green]yes[/green]' if verdict.passed else '[red]no[/red]'}\n"
            f"confidence: {verdict.confidence}\n\n{verdict.reasoning}",
            title=f"verdict (attempt {judged.attempts}"
                  f"{', cached' if judged.from_cache else ''})",
            style="green" if verdict.passed else "red",
        )
    )
    console.print(
        f"[dim]deterministic scorers said: "
        f"{'pass' if result.passed else 'fail'}[/dim]"
    )


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Judge answers and trace agent runs.")
    parser.add_argument("--calibrate", action="store_true", help="Judge vs deterministic scorers.")
    parser.add_argument("--bias", action="store_true", help="Demonstrate judge biases.")
    parser.add_argument("--trace", metavar="QUESTION", help="Run the agent and show a span tree.")
    parser.add_argument("--save-trace", metavar="NAME", help="Save the trace under NAME.")
    parser.add_argument("--cost", metavar="RUN", help="Cost report for a saved eval run.")
    parser.add_argument("--cost-compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--judge-case", metavar="ID", help="Judge one case verbosely.")
    parser.add_argument("--model", help="Override the model.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached judgements.")
    parser.add_argument("--clear-cache", action="store_true", help="Delete cached judgements.")
    args = parser.parse_args()

    if args.clear_cache:
        console.print(f"[dim]cleared {clear_cache()} cached judgement(s)[/dim]")
        if not any([args.calibrate, args.bias, args.trace, args.judge_case]):
            return 0

    # Cost reporting needs no model.
    if args.cost:
        experiment_cost(args.cost)
        return 0
    if args.cost_compare:
        experiment_cost_compare(*args.cost_compare)
        return 0

    if not any([args.calibrate, args.bias, args.trace, args.judge_case]):
        parser.print_help()
        return 0

    try:
        client = get_client(model=args.model) if args.model else get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(Panel.fit(f"judge/agent model: {client.config.describe()}", style="bold cyan"))
    use_cache = not args.no_cache

    if args.calibrate:
        experiment_calibrate(client, use_cache)
    elif args.bias:
        experiment_bias(client, use_cache)
    elif args.judge_case:
        experiment_judge_case(client, args.judge_case, use_cache)
    elif args.trace:
        experiment_trace(client, args.trace, args.save_trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
