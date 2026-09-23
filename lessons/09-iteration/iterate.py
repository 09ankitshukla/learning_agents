"""Lesson 9 deliverable: change one thing, measure it, keep or revert.

    uv run lessons/09-iteration/iterate.py --list
    uv run lessons/09-iteration/iterate.py --try documented_refusal     # free
    uv run lessons/09-iteration/iterate.py --noise --repeats 2          # ~33k tokens
    uv run lessons/09-iteration/iterate.py --try verify_conversion      # ~33k tokens
    uv run lessons/09-iteration/iterate.py --log
    uv run lessons/09-iteration/iterate.py --scorecard
    uv run lessons/09-iteration/iterate.py --replay                     # free

Read `--list` first. It shows every experiment, its prediction, and what it will
cost, and it refuses to run anything that changes two variables at once.

Cost note: a full 16-case run is ~33,000 tokens against a 200,000/day allowance, so
this lesson is three or four experiments per day, not thirty. That constraint is the
lesson as much as the code is -- it forces you to decide which hypothesis is worth
testing, which is a better habit than being able to test all of them.
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
for _folder in ("01-structured-output", "02-tool-calling", "03-agent-loop",
                "07-evaluation", "08-judging-tracing", "09-iteration"):
    _path = str(_LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate import print_comparison, print_scorecard  # noqa: E402
from harness import EvalRun, compare, run_eval  # noqa: E402

from experiments import BASELINE, BASELINE_RUN, EXPERIMENTS, Status, by_id  # noqa: E402
from iteration import (  # noqa: E402
    Changelog,
    Decision,
    build_attempt,
    check_predictions,
    decide,
    load_noise,
    measure_noise,
    save_noise,
)

#: 16 cases at ~2,100 tokens each. Printed rather than hidden, because an experiment
#: you cannot afford is a different decision from one you can.
ESTIMATED_TOKENS_PER_RUN = 33_000

#: Seconds between live cases. Much larger than lesson 7's default of 3, because the
#: per-minute ceiling is 8,000 tokens and 16 cases at ~2,100 each will sail past it
#: at 3-second spacing. A 413 mid-suite costs more time than the pauses do.
DEFAULT_PAUSE = 18.0


# ---------------------------------------------------------------------------
def print_experiments() -> None:
    baseline = _try_load(BASELINE_RUN)
    changelog = Changelog.load()

    table = Table(title=f"experiments (baseline: {BASELINE.describe()})")
    table.add_column("id", style="cyan", width=18)
    table.add_column("variable", width=32, overflow="fold")
    table.add_column("predicts", width=22, overflow="fold")
    table.add_column("cost", width=6, justify="right")
    table.add_column("state", width=10)

    for experiment in EXPERIMENTS:
        problems = experiment.validate()
        if baseline is not None:
            problems += check_predictions(experiment, baseline)

        if experiment.predicts_no_change:
            predicts = "[dim]no change (control)[/dim]"
        else:
            parts = []
            if experiment.predicts_fixed:
                parts.append("[green]+" + ", +".join(experiment.predicts_fixed) + "[/green]")
            if experiment.predicts_broken:
                parts.append("[red]-" + ", -".join(experiment.predicts_broken) + "[/red]")
            predicts = "\n".join(parts)

        attempts = changelog.for_experiment(experiment.id)
        if attempts:
            last = attempts[-1]
            colour = {"keep": "green", "revert": "red"}.get(last.decision, "yellow")
            state = f"[{colour}]{last.decision}[/{colour}]"
            if last.forced:
                state += " [dim](forced)[/dim]"
        elif experiment.status is Status.REJECTED_UNRUN:
            state = "[red]rejected[/red]"
        elif problems and not experiment.requires_force:
            state = "[red]invalid[/red]"
        elif problems:
            state = "[yellow]needs --force[/yellow]"
        else:
            state = "[dim]not run[/dim]"

        table.add_row(
            experiment.id,
            experiment.variable,
            predicts,
            "free" if experiment.free else f"~{ESTIMATED_TOKENS_PER_RUN // 1000}k",
            state,
        )
    console.print(table)

    for experiment in EXPERIMENTS:
        if experiment.status is Status.REJECTED_UNRUN and experiment.rejected_because:
            console.print(
                f"\n[red]rejected without a full run:[/red] [cyan]{experiment.id}[/cyan]\n"
                f"  {experiment.rejected_because}"
            )

    for experiment in EXPERIMENTS:
        problems = experiment.validate()
        if baseline is not None:
            problems += check_predictions(experiment, baseline)
        if problems and experiment.requires_force:
            console.print(
                f"\n[yellow]needs --force:[/yellow] {problems[0]}"
            )
            continue
        for problem in problems:
            console.print(f"[red]invalid:[/red] {problem}")

    noise = load_noise()
    if noise:
        console.print(
            f"\nnoise floor: {len(noise.flipped)}/{noise.cases} cases flipped across "
            f"{noise.repeats} identical runs -> a net change must reach "
            f"[bold]{noise.min_detectable} case(s)[/bold] to be believable."
        )
        if noise.flipped:
            console.print(f"[dim]flipped: {', '.join(noise.flipped)}[/dim]")
        if noise.provisional:
            console.print(
                f"[yellow]provisional:[/yellow] only {noise.repeats} repeats. A case "
                f"failing one run in seven looks stable at this sample size -- that "
                f"happened here, so the threshold is held at "
                f"{noise.min_detectable} rather than the measured "
                f"{len(noise.flipped) + 1}. Use --recheck on any case a decision rests on."
            )
    else:
        console.print(
            "\n[yellow]No noise floor measured yet.[/yellow] Until you run "
            "`--noise`, every decision assumes a 2-case threshold, which is a guess. "
            "Measuring it costs one full run."
        )

    console.print(
        Panel(
            "The columns to read first are [bold]variable[/bold] and "
            "[bold]predicts[/bold], not [bold]state[/bold].\n\n"
            "One variable per experiment is enforced, not suggested: `validate()` "
            "refuses anything that differs from the baseline in more than one field. "
            "Change the prompt and the step cap together and a win tells you nothing "
            "you can ship.\n\n"
            "The prediction is written before the run and never edited. That is what "
            "makes `--scorecard` possible, and the hit rate it reports is usually a "
            "surprise.",
            style="cyan",
        )
    )


# ---------------------------------------------------------------------------
def run_noise(client, repeats: int, pause: float) -> int:
    """Re-run the baseline configuration with the cache OFF and diff the repeats.

    The first repeat is lesson 7's committed `baseline`, so N repeats cost N-1 runs.
    """
    baseline = _try_load(BASELINE_RUN)
    if baseline is None:
        console.print(f"[red]No `{BASELINE_RUN}` run to compare against.[/red]")
        return 1

    console.print(
        Panel.fit(
            f"Measuring the noise floor: {repeats - 1} uncached re-run(s) of the "
            f"baseline configuration, ~{ESTIMATED_TOKENS_PER_RUN * (repeats - 1):,} tokens.\n"
            f"The cache is off, so nothing is read from it and nothing is written to it "
            f"-- the committed baseline's cached executions are untouched.",
            style="yellow",
        )
    )

    runs = [baseline]
    for index in range(1, repeats):
        name = f"noise_{index}"
        console.print(f"\n[dim]repeat {index + 1}/{repeats} -> {name}[/dim]")
        run, _ = run_eval(
            client,
            BASELINE.build_registry(),
            name,
            cases=BASELINE.build_cases(),
            system_prompt=BASELINE.system_prompt,
            use_cache=False,
            pause_between=pause,
            on_progress=_progress(),
            variant_key=BASELINE.residual_key,
            extra_config={"purpose": "noise floor repeat"},
        )
        run.save()
        runs.append(run)

    floor = measure_noise(runs)
    save_noise(floor)

    console.print(Rule("noise floor"))
    table = Table()
    table.add_column("metric", width=26)
    table.add_column("value", justify="right")
    table.add_row("repeats", str(floor.repeats))
    table.add_row("cases", str(floor.cases))
    table.add_row("flipped", f"{len(floor.flipped)} ({floor.flip_rate:.0%})")
    table.add_row("min detectable change", f"{floor.min_detectable} case(s)")
    table.add_row("", f"{floor.detectable_rate:.0%} of the suite")
    if floor.provisional:
        table.add_row("[yellow]provisional[/yellow]", "[yellow]fewer than 3 repeats[/yellow]")
    console.print(table)
    if floor.flipped:
        console.print(f"\n[yellow]flipped:[/yellow] {', '.join(floor.flipped)}")
    for run in runs:
        console.print(f"[dim]{run.name}: {run.passed}/{run.total}[/dim]")

    console.print(
        Panel(
            "Temperature 0 is not determinism. The same question, the same model, the "
            "same prompt, and the verdict can still move -- floating-point reduction "
            "order varies with server batching, and a reasoning model's hidden "
            "deliberation amplifies it.\n\n"
            f"A net change below {floor.min_detectable} case(s) is not distinguishable "
            f"from running the same thing twice. Note the tension this exposes: caching "
            f"is what makes comparisons reproducible, and it is exactly what hides "
            f"variance. One mechanism cannot give you both.\n\n"
            + (
                "This measurement is PROVISIONAL and you should not trust a clean "
                "result from it. Two repeats agreeing is weak evidence: a case that "
                "fails one run in seven will look perfectly stable, which is exactly "
                "what happened in this project -- 0/16 flips at R=2, then a case found "
                "flaking at roughly 14%. Run at least 3 repeats, and use --recheck on "
                "any single case a decision hangs on."
                if floor.provisional
                else "Three or more repeats, so the floor is usable. Still prefer "
                "--recheck on any single case a decision turns on: a suite-level rate "
                "does not tell you about the one case in front of you."
            ),
            style="cyan",
        )
    )
    return 0


# ---------------------------------------------------------------------------
def run_experiment(client, experiment_id: str, pause: float, force: bool) -> int:
    experiment = by_id(experiment_id)
    baseline = _try_load(experiment.baseline_run)
    if baseline is None:
        console.print(f"[red]No `{experiment.baseline_run}` run to compare against.[/red]")
        return 1

    problems = experiment.validate() + check_predictions(experiment, baseline)
    overridden = False
    if problems:
        for problem in problems:
            console.print(f"[red]invalid:[/red] {problem}")
        if not force:
            console.print(
                "\n[dim]Fix the experiment, or pass --force to run it anyway and have "
                "the changelog record that you did.[/dim]"
            )
            return 1
        overridden = True
        console.print(
            "\n[yellow]--force: running anyway. The changelog will record that the "
            "one-variable rule was overridden here.[/yellow]"
        )

    console.print(
        Panel(
            f"[bold]{experiment.id}[/bold]\n\n"
            f"hypothesis: {experiment.hypothesis}\n\n"
            f"variable:   {experiment.variable}\n"
            f"config:     {experiment.config.describe()}\n"
            f"predicts:   "
            + (
                "no change (control)"
                if experiment.predicts_no_change
                else f"fixed={experiment.predicts_fixed or '[]'} "
                f"broken={experiment.predicts_broken or '[]'}"
            )
            + f"\ncost:       {'free (re-scores cached executions)' if experiment.free else f'~{ESTIMATED_TOKENS_PER_RUN:,} tokens'}",
            style="bold blue",
        )
    )

    config = experiment.config
    run, cache_hits = run_eval(
        client,
        config.build_registry(),
        experiment.run_name,
        cases=config.build_cases(),
        system_prompt=config.system_prompt,
        pause_between=pause,
        on_progress=_progress(),
        variant_key=config.residual_key,
        extra_config={
            "experiment": experiment.id,
            "variable": experiment.variable,
            "config": config.describe(),
        },
    )
    run.save()
    print_scorecard(run, cache_hits)

    comparison = compare(baseline, run)
    print_comparison(comparison)

    noise = load_noise()
    judgement = decide(
        comparison,
        noise,
        unpredicted_breaks=[
            case_id for case_id in comparison.broken
            if case_id not in experiment.predicts_broken
        ],
    )

    changelog = Changelog.load()
    attempt = build_attempt(experiment, comparison, judgement, forced=overridden)
    changelog.append(attempt)
    changelog.save()

    _print_judgement(attempt, judgement)
    return 0


def recheck_case(client, case_id: str, experiment_id: str, repeats: int, pause: float) -> int:
    """Run one case repeatedly under one experiment's config, cache off.

    This exists because of a specific mistake. `verify_first` fixed the case it was
    aimed at and broke one nobody predicted, so the rule reverted it -- and the
    "regression" turned out to reproduce about one run in seven. A whole suite is too
    expensive to repeat, but a single case is not: five repeats of one case cost about
    23,000 tokens against 33,000 for a full run, and they answer a question the full
    run cannot.

    The lesson generalises past this project. A per-case flake rate is the thing you
    need when a result hangs on one case, and a suite-level noise floor built from two
    repeats will not give it to you.
    """
    from dataset import by_id as case_by_id
    from harness import run_case

    # `--experiment baseline` is not a special case bolted on for convenience. A flake
    # rate under your change tells you nothing about whether your change caused it;
    # only the same measurement on the control does. Making the control awkward to
    # reach would be the wrong default.
    if experiment_id == "baseline":
        config, label = BASELINE, "baseline (control)"
    else:
        experiment = by_id(experiment_id)
        config, label = experiment.config, experiment.id

    case = next((c for c in config.build_cases() if c.id == case_id), None)
    if case is None:
        console.print(f"[red]No case {case_id!r} in this dataset.[/red]")
        return 1
    case_by_id(case_id)  # raises with a helpful message if the id is simply wrong

    console.print(
        Panel.fit(
            f"rechecking [bold]{case_id}[/bold] under [bold]{label}[/bold]\n"
            f"{repeats} uncached repeats, ~{repeats * 4_600:,} tokens",
            style="yellow",
        )
    )

    passes = 0
    stop_reasons: list[str] = []
    for index in range(repeats):
        result, _ = run_case(
            client,
            config.build_registry(),
            case,
            system_prompt=config.system_prompt,
            use_cache=False,
            variant_key=config.residual_key,
        )
        passes += int(result.passed)
        stop_reasons.append(result.stop_reason)
        mark = "[green]pass[/green]" if result.passed else "[red]fail[/red]"
        console.print(
            f"  {index + 1}/{repeats}  {mark}  stop={result.stop_reason}  "
            f"tools={'->'.join(result.tool_sequence) or '(none)'}"
        )
        for score in result.scores:
            if not score["passed"]:
                console.print(f"        [red]{score['name']}: {score['detail'][:110]}[/red]")
        if index < repeats - 1:
            import time

            time.sleep(pause)

    rate = passes / repeats
    console.print(f"\n[bold]{passes}/{repeats} passed ({rate:.0%})[/bold]")
    console.print(f"[dim]stop reasons: {', '.join(sorted(set(stop_reasons)))}[/dim]")

    if 0 < passes < repeats:
        verdict = (
            f"FLAKY. This case is not a stable signal, so any decision that turned on it "
            f"was decided by chance. Either fix the flake or take the case out of the "
            f"suite -- leaving it in means every future comparison carries a coin flip."
        )
    elif passes == repeats:
        verdict = (
            "Passed every repeat. If this case failed in a run, that failure was the "
            "flake, and a decision made on it should be revisited."
        )
    else:
        verdict = (
            "Failed every repeat. That is a real regression, not noise, and the revert "
            "stands."
        )
    if experiment_id != "baseline":
        verdict += (
            f"\n\nThis does not yet say your change caused it. Run the same check on the "
            f"control before claiming that:\n  --recheck {case_id} --experiment baseline "
            f"--repeats {repeats}"
        )
    console.print(Panel(verdict, style="cyan"))
    return 0


def _print_judgement(attempt, judgement) -> None:
    colour = {Decision.KEEP: "green", Decision.REVERT: "red"}.get(judgement.decision, "yellow")
    console.print(Rule("decision"))
    console.print(f"[bold {colour}]{judgement.decision.value.upper()}[/bold {colour}]")
    for reason in judgement.reasons:
        console.print(f"  - {reason}")
    for caveat in judgement.caveats:
        console.print(f"  [yellow]still unknown:[/yellow] {caveat}")
    if judgement.recheck:
        console.print(
            f"\n[yellow]next:[/yellow] uv run lessons/09-iteration/iterate.py "
            f"--recheck {judgement.recheck[0]} --experiment {attempt.experiment_id} --repeats 5"
        )

    console.print(f"\nprediction: {'[green]hit[/green]' if attempt.prediction_hit else '[red]miss[/red]'}")
    if attempt.predicted_no_change:
        console.print("  predicted: no change (control)")
    else:
        console.print(
            f"  predicted: fixed={attempt.predicted_fixed or '[]'} "
            f"broken={attempt.predicted_broken or '[]'}"
        )
    console.print(
        f"  actual:    fixed={attempt.actual_fixed or '[]'} "
        f"broken={attempt.actual_broken or '[]'}"
    )
    if attempt.surprises:
        console.print(f"  [yellow]unforeseen: {', '.join(attempt.surprises)}[/yellow]")

    console.print("\n[dim]appended to attempts.json, decision included.[/dim]")


# ---------------------------------------------------------------------------
def print_log() -> int:
    changelog = Changelog.load()
    if not changelog.attempts:
        console.print("[dim]No attempts recorded yet.[/dim]")
        return 0

    console.print(Rule(f"changelog: {len(changelog.attempts)} attempt(s)"))
    for attempt in changelog.attempts:
        colour = {"keep": "green", "revert": "red"}.get(attempt.decision, "yellow")
        console.print(
            f"\n[bold]{attempt.experiment_id}[/bold]  "
            f"[{colour}]{attempt.decision}[/{colour}]  "
            f"[dim]{attempt.created_at}[/dim]"
        )
        console.print(f"  variable:   {attempt.variable}")
        if attempt.forced:
            console.print("  [yellow]forced:     the one-variable rule was overridden[/yellow]")
        console.print(
            f"  score:      {attempt.baseline_passed}/{attempt.total_cases} -> "
            f"{attempt.candidate_passed}/{attempt.total_cases}  (net {attempt.net:+d})"
        )
        if attempt.actual_fixed:
            console.print(f"  [green]fixed:      {', '.join(attempt.actual_fixed)}[/green]")
        if attempt.actual_broken:
            console.print(f"  [red]broke:      {', '.join(attempt.actual_broken)}[/red]")
        console.print(
            f"  prediction: {'hit' if attempt.prediction_hit else 'MISS'}"
            + (f"  [yellow]unforeseen {', '.join(attempt.surprises)}[/yellow]"
               if attempt.surprises else "")
        )
        if attempt.cost_per_success_before and attempt.cost_per_success_after:
            ratio = attempt.cost_per_success_after / attempt.cost_per_success_before
            console.print(f"  cost/success: {ratio:.2f}x")
        for reason in attempt.reasons:
            console.print(f"  [dim]> {reason}[/dim]")

    console.print(
        Panel(
            "This file is append-only and keeps the rejections on purpose.\n\n"
            "A log of only the changes that shipped is worse than no log: six months "
            "on, someone proposes an idea that was already measured and lost, and "
            "nothing in the repo contradicts them. The rejected attempts are the part "
            "that saves time.\n\n"
            f"Spent so far: {changelog.tokens_spent:,} tokens across "
            f"{len(changelog.attempts)} attempt(s), of which "
            f"{len(changelog.kept)} were kept.",
            style="cyan",
        )
    )
    return 0


def print_prediction_scorecard() -> int:
    changelog = Changelog.load()
    if not changelog.attempts:
        console.print("[dim]No attempts recorded yet.[/dim]")
        return 0

    table = Table(title="were the predictions right?")
    table.add_column("experiment", style="cyan", width=20)
    table.add_column("predicted", width=30, overflow="fold")
    table.add_column("actual", width=30, overflow="fold")
    table.add_column("", width=6)

    for attempt in changelog.attempts:
        predicted = (
            "no change"
            if attempt.predicted_no_change
            else " ".join(
                [f"+{c}" for c in attempt.predicted_fixed]
                + [f"-{c}" for c in attempt.predicted_broken]
            )
            or "nothing"
        )
        actual = (
            " ".join(
                [f"+{c}" for c in attempt.actual_fixed] + [f"-{c}" for c in attempt.actual_broken]
            )
            or "no change"
        )
        table.add_row(
            attempt.experiment_id,
            predicted,
            actual,
            "[green]hit[/green]" if attempt.prediction_hit else "[red]miss[/red]",
        )
    console.print(table)
    console.print(
        f"\n[bold]{changelog.predictions_hit}/{changelog.predictions_made} predictions "
        f"correct ({changelog.hit_rate:.0%})[/bold]"
    )
    console.print(
        Panel(
            "This is the number that justifies the whole lesson.\n\n"
            "If a hypothesis about a prompt were reliable, you could reason your way to "
            "a better agent and skip the measuring. The hit rate is the evidence for or "
            "against that, on your own agent, in your own words -- and it is graded "
            "strictly: predicting the win but missing a regression counts as a miss, "
            "because in production the regression is the part that matters.\n\n"
            "A low rate is not embarrassing. It is the reason the harness exists.",
            style="cyan",
        )
    )
    return 0


def replay() -> int:
    """Re-derive every decision from the saved runs, spending nothing.

    The changelog stores what was decided; this recomputes what *would* be decided
    now. They diverge whenever a rule or the noise floor changes, and seeing that
    divergence is the point: a decision is a function of evidence and a rule, and
    both are allowed to improve. Storing only the verdict would hide that.
    """
    changelog = Changelog.load()
    if not changelog.attempts:
        console.print("[dim]No attempts recorded yet.[/dim]")
        return 0

    noise = load_noise()
    table = Table(title="stored decision vs the rule as it stands today")
    table.add_column("experiment", style="cyan", width=20)
    table.add_column("stored", width=14)
    table.add_column("now", width=14)
    table.add_column("note", overflow="fold")

    changed = 0
    for attempt in changelog.attempts:
        try:
            baseline = EvalRun.load(attempt.baseline)
            candidate = EvalRun.load(attempt.candidate)
        except FileNotFoundError:
            table.add_row(attempt.experiment_id, attempt.decision, "[dim]?[/dim]",
                          "run files missing; cannot re-derive")
            continue
        judgement = decide(
            compare(baseline, candidate),
            noise,
            # Re-derived from the stored prediction, not from today's registry. The
            # prediction is history and must not be quietly updated to match the
            # outcome -- that would turn the scorecard into a formality.
            unpredicted_breaks=[
                c for c in attempt.actual_broken if c not in attempt.predicted_broken
            ],
        )
        now = judgement.decision.value
        same = now == attempt.decision
        changed += int(not same)
        table.add_row(
            attempt.experiment_id,
            attempt.decision,
            now if same else f"[yellow]{now}[/yellow]",
            judgement.reasons[0] if judgement.reasons else "",
        )
    console.print(table)
    console.print(
        f"\n{changed} of {len(changelog.attempts)} decision(s) would differ today."
        + (
            "  [dim]Nothing has changed, which is what you want to see until a rule or "
            "the noise floor moves.[/dim]"
            if not changed
            else ""
        )
    )
    return 0


# ---------------------------------------------------------------------------
def _try_load(name: str) -> EvalRun | None:
    try:
        return EvalRun.load(name)
    except FileNotFoundError:
        return None


def _progress():
    def report(case, result, from_cache) -> None:
        mark = "[green]y[/green]" if result.passed else "[red]n[/red]"
        if result.error:
            mark = "[yellow]![/yellow]"
        tag = " [dim](cached)[/dim]" if from_cache else ""
        console.print(f"  {mark} {case.id:28}{tag}")
        if result.error:
            console.print(f"      [yellow]{result.error[:140]}[/yellow]")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iterate on the agent: one variable, measured, kept or reverted."
    )
    parser.add_argument("--list", action="store_true", help="Experiments, predictions and cost.")
    parser.add_argument("--try", dest="try_id", metavar="ID", help="Run one experiment.")
    parser.add_argument("--noise", action="store_true", help="Measure the noise floor.")
    parser.add_argument("--repeats", type=int, default=2, help="Repeats for --noise (incl. baseline).")
    parser.add_argument("--log", action="store_true", help="The changelog, rejections included.")
    parser.add_argument("--scorecard", action="store_true", help="Prediction hit rate.")
    parser.add_argument("--replay", action="store_true", help="Re-derive decisions, free.")
    parser.add_argument("--recheck", metavar="CASE_ID",
                        help="Repeat one case under --experiment to test whether a break is real.")
    parser.add_argument("--experiment", metavar="ID", help="Which experiment's config to recheck under.")
    parser.add_argument("--force", action="store_true", help="Run an invalid experiment anyway.")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE,
                        help="Seconds between live cases (per-minute token ceiling).")
    parser.add_argument("--model", help="Override the model. Changes what you are comparing.")
    args = parser.parse_args()

    if args.list:
        print_experiments()
        return 0
    if args.log:
        return print_log()
    if args.scorecard:
        return print_prediction_scorecard()
    if args.replay:
        return replay()

    if args.recheck and not args.experiment:
        console.print("[red]--recheck also needs --experiment ID.[/red]")
        return 1

    if not (args.try_id or args.noise or args.recheck):
        parser.print_help()
        return 0

    model = args.model or BASELINE.model
    try:
        client = get_client(model=model)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    if model != BASELINE.model:
        console.print(
            f"[yellow]warning:[/yellow] running on {model}, but the baseline was "
            f"{BASELINE.model}. Any difference you see is the model, not your variable."
        )

    if args.recheck:
        return recheck_case(client, args.recheck, args.experiment, max(args.repeats, 2), args.pause)
    if args.noise:
        if args.repeats < 2:
            console.print("[red]--repeats must be at least 2.[/red]")
            return 1
        if args.repeats < 3:
            console.print(
                "[yellow]warning:[/yellow] two repeats cannot establish a noise floor. "
                "A case failing one run in seven looks perfectly stable at R=2, which "
                "happened in this project. The result will be marked provisional."
            )
        return run_noise(client, args.repeats, args.pause)
    return run_experiment(client, args.try_id, args.pause, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
