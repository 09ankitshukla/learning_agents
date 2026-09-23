"""The iteration loop: noise floor, decision rule, and an append-only changelog.

Three ideas, in the order they matter.

**Measure the noise before you measure the change.** Run the identical
configuration twice and some cases will flip anyway. That flip count is the floor on
what your suite can detect: if two cases move on their own, a two-case "improvement"
is indistinguishable from doing nothing. Almost every eval suite skips this step and
then reports single-case wins as progress.

**The decision must be a rule, not a feeling.** `decide()` takes a comparison and
returns keep/revert/inconclusive with its reasoning. Writing it down in advance is
what stops you re-reading a result until it says what you hoped. One rule overrides
the arithmetic: a regression in a safety case is never bought off by wins elsewhere.

**Record the attempts you rejected.** `Changelog` is append-only and keeps the
failures. It is the only artifact here that gets more valuable over time, because it
is what stops the same idea being tried three times.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

_LESSONS = Path(__file__).resolve().parents[1]
for _folder in ("07-evaluation", "08-judging-tracing"):
    _path = str(_LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from harness import Comparison, EvalRun, wilson_interval  # noqa: E402
from tracing import cost_from_eval_run  # noqa: E402

CHANGELOG_PATH = Path(__file__).parent / "attempts.json"

#: Categories where a regression is not tradeable. These are the fabrication and
#: sandbox-escape cases: an agent that starts inventing share prices has not got
#: "slightly worse on one case", it has acquired a different and worse failure mode.
#: Averaging that against two arithmetic wins is how you ship something harmful.
PROTECTED_CATEGORIES = frozenset({"impossible"})

#: A keep is downgraded if the accuracy gain costs more than this much extra per
#: successful answer. 2.0 means "doubling the price of a right answer needs an
#: argument, not just a green number".
COST_PER_SUCCESS_CEILING = 2.0


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
@dataclass
class NoiseFloor:
    """How much the suite moves when nothing changes."""

    run_names: list[str]
    #: Cases that did not agree across every repeat.
    flipped: list[str] = field(default_factory=list)
    #: case id -> how many repeats it passed in.
    pass_counts: dict[str, int] = field(default_factory=dict)
    cases: int = 0

    @property
    def repeats(self) -> int:
        return len(self.run_names)

    @property
    def flip_rate(self) -> float:
        return len(self.flipped) / self.cases if self.cases else 0.0

    @property
    def provisional(self) -> bool:
        """True when too few repeats were run to trust the measurement.

        Two repeats sound like enough and are not. A case that fails one time in
        seven has a good chance of looking perfectly stable across two runs, so a
        clean result at R=2 is weak evidence of stability rather than evidence of
        determinism. This project measured 0/16 flips at R=2 and then found a case
        that flaked at roughly 1 in 7 -- inside the same afternoon.
        """
        return self.repeats < 3

    @property
    def min_detectable(self) -> int:
        """Smallest net case change worth believing.

        If N cases flip with no change at all, a net movement of N could be entirely
        noise, so you need N+1 before claiming anything. Never 0: even with no
        observed flips, concluding the suite is deterministic is a much stronger claim
        than the evidence supports.

        And when the measurement is provisional, the measured value is not used as-is.
        Returning 1 from two clean repeats is how a flake gets promoted to a finding,
        which is exactly what happened here before this floor was added.
        """
        measured = len(self.flipped) + 1
        return max(measured, 2) if self.provisional else measured

    @property
    def detectable_rate(self) -> float:
        return self.min_detectable / self.cases if self.cases else 0.0

    def to_json(self) -> dict:
        return asdict(self) | {
            "repeats": self.repeats,
            "min_detectable": self.min_detectable,
        }


def measure_noise(runs: list[EvalRun]) -> NoiseFloor:
    """Compare repeats of the same configuration and report what moved.

    Caching and variance measurement are in direct tension, and it is worth being
    explicit about it. Lesson 7 caches executions so that comparisons are
    reproducible -- a re-run of the baseline gives identical numbers. That is exactly
    what you want when comparing two configurations, and exactly what makes variance
    invisible. So the repeats here must be produced with the cache OFF, and they cost
    full price. The same mechanism cannot give you both reproducibility and variance.
    """
    if len(runs) < 2:
        raise ValueError("Need at least two runs of the same configuration.")

    case_ids = [r.case_id for r in runs[0].results]
    floor = NoiseFloor(run_names=[r.name for r in runs], cases=len(case_ids))
    for case_id in case_ids:
        verdicts = []
        for run in runs:
            result = run.result_for(case_id)
            if result is None:
                break
            verdicts.append(result.passed)
        if len(verdicts) != len(runs):
            continue
        floor.pass_counts[case_id] = sum(verdicts)
        if 0 < sum(verdicts) < len(verdicts):
            floor.flipped.append(case_id)
    return floor


def load_noise() -> NoiseFloor | None:
    """The saved noise floor, if one has been measured."""
    path = Path(__file__).parent / "noise.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return NoiseFloor(
        run_names=raw["run_names"],
        flipped=raw.get("flipped", []),
        pass_counts=raw.get("pass_counts", {}),
        cases=raw.get("cases", 0),
    )


def save_noise(floor: NoiseFloor) -> Path:
    path = Path(__file__).parent / "noise.json"
    path.write_text(json.dumps(floor.to_json(), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
class Decision(str, Enum):
    KEEP = "keep"
    REVERT = "revert"
    #: The most useful of the three, and the one teams refuse to say. It means the
    #: suite cannot resolve this change -- not that the change is bad. The follow-up
    #: is a bigger dataset, not a bigger opinion.
    INCONCLUSIVE = "inconclusive"


@dataclass
class Judgement:
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    cost_before: float = 0.0
    cost_after: float = 0.0
    #: Cases the rule wants re-tested before it will commit. Populated when a result
    #: turns on one or two case movements nobody predicted, because at that size a
    #: flake and a regression look identical and only a re-run separates them.
    recheck: list[str] = field(default_factory=list)
    #: Separate from `reasons` because these do not change the decision. They are the
    #: things that remain unknown after it, and they are what an honest write-up
    #: reports alongside the verdict.
    caveats: list[str] = field(default_factory=list)

    @property
    def cost_ratio(self) -> float:
        if not self.cost_before:
            return 0.0
        return self.cost_after / self.cost_before


def decide(
    comparison: Comparison,
    noise: NoiseFloor | None = None,
    *,
    protected: frozenset[str] = PROTECTED_CATEGORIES,
    unpredicted_breaks: list[str] | None = None,
) -> Judgement:
    """Apply the rule. Order matters, and the first rule is not about the score.

    1. A regression in a protected category reverts, whatever the net. Not
       negotiable and not averaged.
    2. A net loss reverts. Nothing was gained.
    3. A trade that nets zero on the strength of an *unpredicted* break asks for a
       re-test rather than reverting. Added after this rule reverted a genuine fix on
       the strength of a regression that turned out to flake about one run in seven.
       Re-testing one case is cheap; throwing away a working change is not.
    4. No movement at all reverts. There is nothing to carry.
    5. A net gain smaller than the noise floor is inconclusive. This is the rule that
       costs discipline, because it refuses wins you want to bank.
    6. Otherwise keep -- unless the win more than doubled the cost per successful
       answer, which needs a decision rather than a default.
    """
    judgement = Judgement(decision=Decision.INCONCLUSIVE)

    report_before = cost_from_eval_run(comparison.baseline)
    report_after = cost_from_eval_run(comparison.candidate)
    judgement.cost_before = report_before.cost_per_success
    judgement.cost_after = report_after.cost_per_success

    # 1. Safety first, and not as a slogan.
    protected_breaks = [
        case_id
        for case_id in comparison.broken
        if (r := comparison.candidate.result_for(case_id)) and r.category in protected
    ]
    if protected_breaks:
        judgement.decision = Decision.REVERT
        judgement.reasons.append(
            f"broke protected case(s) {protected_breaks} in category "
            f"{'/'.join(sorted(protected))}. A fabrication or sandbox regression is not "
            f"offset by gains elsewhere, so the net score is not consulted."
        )
        return judgement

    net = comparison.net
    surprises = [c for c in comparison.broken if c in (unpredicted_breaks or [])]

    # 2. No gain.
    if net < 0:
        judgement.decision = Decision.REVERT
        judgement.reasons.append(f"net {net:+d} cases: strictly worse.")
        return judgement

    # 3. A tie that hangs on a break nobody saw coming.
    if net == 0 and comparison.fixed and surprises:
        judgement.decision = Decision.INCONCLUSIVE
        judgement.recheck = list(surprises)
        judgement.reasons.append(
            f"net zero, but the whole result turns on {surprises}, which nobody "
            f"predicted. At one case a flake and a regression are indistinguishable, and "
            f"this rule has already reverted a genuine fix on the strength of a failure "
            f"that reproduced about one run in seven. Re-test before deciding: "
            f"`--recheck {surprises[0]} --experiment ...` costs one case, not one suite."
        )
        return judgement

    if net == 0:
        judgement.decision = Decision.REVERT
        if comparison.fixed and comparison.broken:
            judgement.reasons.append(
                f"net zero, but not nothing: traded {comparison.broken} for "
                f"{comparison.fixed}, and the break was predicted, so it is a real "
                f"trade rather than possible noise. Revert unless you value those cases "
                f"differently, in which case say so explicitly."
            )
        else:
            judgement.reasons.append("no case changed. Nothing gained; do not carry it.")
        return judgement

    # 3. Is the gain bigger than the noise?
    floor = noise.min_detectable if noise else 2
    source = "measured noise floor" if noise else "default assumption (no noise floor measured)"
    if net < floor:
        judgement.decision = Decision.INCONCLUSIVE
        judgement.reasons.append(
            f"net {net:+d} case(s) is below the {floor}-case threshold from the {source}. "
            f"The change may well be real; this suite cannot show it. Grow the dataset "
            f"rather than re-running until it looks better."
        )
    else:
        judgement.decision = Decision.KEEP
        judgement.reasons.append(
            f"net {net:+d} cases, at or above the {floor}-case threshold, with no "
            f"protected regression."
        )

    # 4. Cost. Applied to a keep only; an inconclusive result is already parked.
    if judgement.decision is Decision.KEEP and judgement.cost_ratio > COST_PER_SUCCESS_CEILING:
        judgement.decision = Decision.INCONCLUSIVE
        judgement.reasons.append(
            f"cost per successful answer rose {judgement.cost_ratio:.1f}x "
            f"(${judgement.cost_before:.5f} -> ${judgement.cost_after:.5f}). The accuracy "
            f"gain is real and you are paying a lot for it; that is a product decision, "
            f"not an automatic keep."
        )
    elif judgement.cost_before and judgement.cost_after:
        judgement.reasons.append(
            f"cost per success {judgement.cost_ratio:.2f}x "
            f"(${judgement.cost_before:.5f} -> ${judgement.cost_after:.5f})."
        )

    judgement.caveats += sample_size_caveats(comparison)
    return judgement


def sample_size_caveats(comparison: Comparison) -> list[str]:
    """What the noise floor cannot tell you, however cleanly it came out.

    Measuring zero flips across repeats answers one question -- "is this change
    reproducible?" -- and people immediately mistake it for the other one: "does this
    change generalise?" They are different, and only the first is about variance.

    A one-case gain that reproduces perfectly is still one case out of sixteen. The
    Wilson intervals for 15/16 and 16/16 overlap heavily, so the suite genuinely
    cannot distinguish the two underlying success rates. That is a sample-size limit
    and no amount of re-running fixes it; only more cases do.

    Reported as a caveat rather than enforced as a gate, because the two failure modes
    are asymmetric. Gating on overlapping intervals at n=16 would block every change
    this suite could ever measure, which is how a team ends up ignoring its own
    harness.
    """
    before, after = comparison.baseline, comparison.candidate
    low_a, high_a = wilson_interval(before.passed, before.total)
    low_b, high_b = wilson_interval(after.passed, after.total)
    caveats = [
        f"95% intervals: {before.name} {low_a:.0%}-{high_a:.0%}, "
        f"{after.name} {low_b:.0%}-{high_b:.0%}."
    ]
    if low_b <= high_a and low_a <= high_b:
        caveats.append(
            f"Those intervals overlap, so with {after.total} cases the suite cannot show "
            f"that this change generalises -- only that it reproduced here. Growing the "
            f"dataset is the fix; re-running is not."
        )
    return caveats


# ---------------------------------------------------------------------------
# Prediction coherence
# ---------------------------------------------------------------------------
def check_predictions(experiment, baseline: EvalRun) -> list[str]:
    """Catch predictions that cannot come true, before spending anything.

    Predicting that an already-failing case will "break", or that an already-passing
    case will be "fixed", is a sign the prediction was written without looking at the
    baseline. Which is the whole problem: step zero of iteration is reading the
    current failures, and it is the step people skip in favour of theorising.
    """
    problems: list[str] = []
    for case_id in experiment.predicts_fixed:
        result = baseline.result_for(case_id)
        if result is None:
            problems.append(f"predicts_fixed names {case_id!r}, which is not in the baseline.")
        elif result.passed:
            problems.append(
                f"predicts_fixed names {case_id!r}, which already passes. Did you read "
                f"the baseline before predicting?"
            )
    for case_id in experiment.predicts_broken:
        result = baseline.result_for(case_id)
        if result is None:
            problems.append(f"predicts_broken names {case_id!r}, which is not in the baseline.")
        elif not result.passed:
            problems.append(
                f"predicts_broken names {case_id!r}, which is already failing, so it "
                f"cannot break."
            )
    return problems


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------
@dataclass
class Attempt:
    """One measured attempt, kept whether or not it worked."""

    experiment_id: str
    created_at: str
    variable: str
    hypothesis: str
    baseline: str
    candidate: str
    predicted_fixed: list[str]
    predicted_broken: list[str]
    predicted_no_change: bool
    actual_fixed: list[str]
    actual_broken: list[str]
    baseline_passed: int
    candidate_passed: int
    total_cases: int
    decision: str
    reasons: list[str]
    cost_per_success_before: float
    cost_per_success_after: float
    tokens_spent: int
    caveats: list[str] = field(default_factory=list)
    #: True when the one-variable rule was overridden. Recorded because the log has to
    #: be readable by someone who was not there, and "we broke our own rule here" is
    #: the single most useful thing such a reader can know about an entry.
    forced: bool = False
    #: Cases the decision is waiting on. An open question, recorded as one.
    recheck: list[str] = field(default_factory=list)

    # -- prediction accuracy --------------------------------------------
    @property
    def prediction_hit(self) -> bool:
        """Exact match on both lists, order-insensitive.

        Deliberately strict. "I said it would help and it did, though it also broke
        something I did not foresee" is not a correct prediction, and grading it as
        one is how you keep believing your intuition is good.
        """
        if self.predicted_no_change:
            return not self.actual_fixed and not self.actual_broken
        if not self.predicted_fixed and not self.predicted_broken:
            # No prediction at all cannot be a correct prediction. Found by a test:
            # two empty lists compared equal to two empty outcomes and scored a hit,
            # which would have inflated the scorecard with attempts that never bet on
            # anything. `validate()` forbids this, but --force can still get one in,
            # and the number this feeds is the whole point of the lesson.
            return False
        return set(self.predicted_fixed) == set(self.actual_fixed) and set(
            self.predicted_broken
        ) == set(self.actual_broken)

    @property
    def surprises(self) -> list[str]:
        """Case movements nobody called. The expensive kind."""
        unforeseen_breaks = sorted(set(self.actual_broken) - set(self.predicted_broken))
        unforeseen_fixes = sorted(set(self.actual_fixed) - set(self.predicted_fixed))
        out = [f"-{c}" for c in unforeseen_breaks] + [f"+{c}" for c in unforeseen_fixes]
        return out

    @property
    def net(self) -> int:
        return len(self.actual_fixed) - len(self.actual_broken)


@dataclass
class Changelog:
    attempts: list[Attempt] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = CHANGELOG_PATH) -> Changelog:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(attempts=[Attempt(**a) for a in raw.get("attempts", [])])

    def save(self, path: Path = CHANGELOG_PATH) -> Path:
        path.write_text(
            json.dumps({"attempts": [asdict(a) for a in self.attempts]}, indent=2),
            encoding="utf-8",
        )
        return path

    def append(self, attempt: Attempt) -> None:
        """Append-only, including re-runs of the same experiment.

        Overwriting an earlier attempt for the same experiment id would destroy the
        most interesting record in the file: the same change measured twice with
        different outcomes, which is the clearest possible evidence about noise.
        """
        self.attempts.append(attempt)

    # -- the scorecard --------------------------------------------------
    @property
    def predictions_made(self) -> int:
        return len(self.attempts)

    @property
    def predictions_hit(self) -> int:
        return sum(1 for a in self.attempts if a.prediction_hit)

    @property
    def hit_rate(self) -> float:
        return self.predictions_hit / self.predictions_made if self.attempts else 0.0

    @property
    def kept(self) -> list[Attempt]:
        return [a for a in self.attempts if a.decision == Decision.KEEP.value]

    @property
    def tokens_spent(self) -> int:
        return sum(a.tokens_spent for a in self.attempts)

    def for_experiment(self, experiment_id: str) -> list[Attempt]:
        return [a for a in self.attempts if a.experiment_id == experiment_id]


def build_attempt(
    experiment, comparison: Comparison, judgement: Judgement, *, forced: bool = False
) -> Attempt:
    return Attempt(
        experiment_id=experiment.id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        variable=experiment.variable,
        hypothesis=experiment.hypothesis,
        baseline=comparison.baseline.name,
        candidate=comparison.candidate.name,
        predicted_fixed=list(experiment.predicts_fixed),
        predicted_broken=list(experiment.predicts_broken),
        predicted_no_change=experiment.predicts_no_change,
        actual_fixed=list(comparison.fixed),
        actual_broken=list(comparison.broken),
        baseline_passed=comparison.baseline.passed,
        candidate_passed=comparison.candidate.passed,
        total_cases=comparison.candidate.total,
        decision=judgement.decision.value,
        reasons=list(judgement.reasons),
        cost_per_success_before=judgement.cost_before,
        cost_per_success_after=judgement.cost_after,
        tokens_spent=comparison.candidate.tokens_actually_spent,
        caveats=list(judgement.caveats),
        forced=forced,
        recheck=list(judgement.recheck),
    )
