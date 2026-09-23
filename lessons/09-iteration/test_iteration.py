"""Tests for the iteration machinery. Offline, free, and pointed at the rules.

The interesting tests here are not the happy paths. They are the ones that pin the
two bugs lesson 9 found in lesson 7's harness -- a cache key blind to a variable, and
a dataset fingerprint blind to a scorer swap -- and the decision rules, which encode
judgements you would otherwise have to remember to apply.

A decision rule is exactly the sort of thing that rots: someone relaxes the safety
override to unblock a release and nothing notices. `test_protected_regression_...`
notices.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from dataset import CASES, by_id
from harness import CaseResult, EvalRun, cache_key, compare
from scorers import mentions, numeric_answer, scorer_label, used_tools

from experiments import (
    BASELINE,
    EXPERIMENTS,
    Config,
    Experiment,
    TOOL_VARIANTS,
    build_variant_registry,
    cited_tool_limits,
)
from iteration import (
    Attempt,
    Changelog,
    Decision,
    NoiseFloor,
    build_attempt,
    check_predictions,
    decide,
    measure_noise,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _result(case_id: str, passed: bool, category: str = "arithmetic", tokens: int = 2000):
    return CaseResult(
        case_id=case_id,
        category=category,
        passed=passed,
        answer="answer",
        tool_sequence=["calculate"],
        stop_reason="completed",
        steps=2,
        prompt_tokens=tokens,
        completion_tokens=tokens // 4,
        latency_s=0.5,
        scores=[],
    )


def _run(name: str, outcomes: dict[str, bool], categories: dict[str, str] | None = None,
         tokens: int = 2000) -> EvalRun:
    categories = categories or {}
    return EvalRun(
        name=name,
        model="openai/gpt-oss-20b",
        provider="groq",
        max_steps=6,
        created_at="2026-01-01T00:00:00+00:00",
        config={"dataset_version": "v2:abc", "variant_key": ""},
        results=[
            _result(cid, passed, categories.get(cid, "arithmetic"), tokens)
            for cid, passed in outcomes.items()
        ],
    )


class _Traj:
    def __init__(self, answer, tools):
        self.final_answer = answer
        self.tool_sequence = tools

    @property
    def succeeded(self):
        return True


# ---------------------------------------------------------------------------
# Config and the one-variable rule
# ---------------------------------------------------------------------------
def test_baseline_differs_from_itself_in_nothing():
    assert BASELINE.diff(BASELINE) == []


def test_diff_names_the_single_changed_field():
    assert replace(BASELINE, prompt="terse").diff(BASELINE) == ["prompt"]
    assert replace(BASELINE, step_bonus=2).diff(BASELINE) == ["step_bonus"]


def test_every_registered_experiment_is_valid_or_declares_that_it_is_not():
    """The registry is documentation, and documentation that quietly fails its own
    rule is worse than none. Exactly one entry breaks the rule, it says so, and it
    still cannot run without --force."""
    for experiment in EXPERIMENTS:
        if experiment.requires_force:
            assert experiment.validate() != [], (
                f"{experiment.id} claims to need --force but validates cleanly"
            )
        else:
            assert experiment.validate() == [], experiment.id


def test_every_registered_experiment_changes_exactly_one_variable():
    for experiment in EXPERIMENTS:
        expected = 1 if not experiment.requires_force else 2
        assert len(experiment.config.diff(BASELINE)) == expected, experiment.id


def test_every_rejected_experiment_says_why():
    """A registry entry marked rejected with no reason is worse than a deleted one: it
    looks settled while telling the next reader nothing, so the idea gets retried."""
    from experiments import Status

    for experiment in EXPERIMENTS:
        if experiment.status is Status.REJECTED_UNRUN:
            assert experiment.rejected_because, experiment.id
            assert len(experiment.rejected_because) > 80, experiment.id


def test_requires_force_is_a_label_not_a_permission():
    """A rule you can opt out of by setting a field is not a rule. The flag documents
    intent; validate() still says no and --force is still required."""
    marked = Experiment(
        id="marked", hypothesis="h", variable="two things",
        config=replace(BASELINE, prompt="terse", step_bonus=2),
        predicts_fixed=["currency_unsupported"], requires_force=True,
    )
    assert marked.validate() != []


def test_two_variable_experiment_is_rejected():
    bad = Experiment(
        id="two_things",
        hypothesis="h",
        variable="prompt and steps",
        config=replace(BASELINE, prompt="terse", step_bonus=2),
        predicts_fixed=["currency_unsupported"],
    )
    problems = bad.validate()
    assert problems
    assert "2 variables" in problems[0]


def test_experiment_with_no_prediction_is_rejected():
    bad = Experiment(
        id="no_bet", hypothesis="h", variable="prompt",
        config=replace(BASELINE, prompt="terse"),
    )
    assert any("predicts nothing" in p for p in bad.validate())


def test_control_must_declare_that_it_predicts_nothing():
    control = Experiment(
        id="control", hypothesis="h", variable="steps",
        config=replace(BASELINE, step_bonus=2), predicts_no_change=True,
    )
    assert control.validate() == []


def test_a_config_identical_to_baseline_cannot_be_an_experiment():
    same = Experiment(id="same", hypothesis="h", variable="none",
                      config=BASELINE, predicts_no_change=True)
    assert any("changes nothing" in p for p in same.validate())


# ---------------------------------------------------------------------------
# The cache-key hole lesson 9 found
# ---------------------------------------------------------------------------
def test_tool_variant_produces_a_different_cache_key():
    """The bug: lesson 7's key covered question/model/steps/prompt, which was
    everything lesson 7 could change. Lesson 9 changes tool descriptions, so without
    a residual key a tool experiment is silently served the baseline's cached runs
    and reports 'no change' -- a measurement tool lying with total confidence."""
    case = by_id("currency_unsupported")
    variant = replace(BASELINE, tool_variant="hide_currency_list")

    base_key = cache_key(case, BASELINE.model, case.max_steps,
                         BASELINE.system_prompt, BASELINE.residual_key)
    variant_key = cache_key(case, variant.model, case.max_steps,
                            variant.system_prompt, variant.residual_key)
    assert base_key != variant_key


def test_every_tool_variant_has_a_distinct_residual_key():
    keys = {
        name: replace(BASELINE, tool_variant=name).residual_key for name in TOOL_VARIANTS
    }
    assert keys["as_built"] == ""
    assert len(set(keys.values())) == len(keys)


def test_baseline_residual_key_is_empty_so_older_cache_entries_survive():
    """Load bearing, not cosmetic. An empty residual key hashes to nothing, so every
    execution cached before lesson 9 existed stays addressable -- which is why
    reproducing lesson 7's baseline costs zero tokens."""
    case = by_id("arith_large_product")
    with_variant = cache_key(case, "m", 6, None, "")
    without_argument = cache_key(case, "m", 6, None)
    assert with_variant == without_argument


def test_dataset_variant_does_not_change_the_cache_key():
    """Scoring changes must not invalidate executions. This is lesson 7's rule --
    cache the execution, never the score -- and lesson 9 is the first caller that
    could break it."""
    case = by_id("currency_unsupported")
    variant = replace(BASELINE, dataset_variant="documented_refusal")
    assert variant.residual_key == BASELINE.residual_key == ""


# ---------------------------------------------------------------------------
# The fingerprint hole lesson 9 found
# ---------------------------------------------------------------------------
def test_scorer_label_includes_arguments():
    assert scorer_label(numeric_answer(31.0)) == "numeric_answer(31)"
    assert scorer_label(used_tools(["calculate"])) == "used_tools(['calculate'])"
    assert scorer_label(numeric_answer(31.0)) != scorer_label(numeric_answer(30.0))


def test_swapping_a_scorer_changes_the_dataset_fingerprint():
    """v1 hashed len(case.scorers), so replacing one scorer with another left the
    fingerprint identical and `compare()` reported apples to apples across two runs
    graded by different instruments."""
    from harness import _dataset_fingerprint

    original = [by_id("arith_large_product")]
    swapped = [replace(original[0], scorers=[mentions(["409"]), used_tools(["calculate"])])]
    assert len(original[0].scorers) == len(swapped[0].scorers)  # v1 saw no difference
    assert _dataset_fingerprint(original) != _dataset_fingerprint(swapped)


def test_fingerprint_is_tagged_with_its_algorithm_version():
    from harness import _dataset_fingerprint

    assert _dataset_fingerprint(CASES).startswith("v2:")


def test_comparison_distinguishes_algorithm_change_from_dataset_change():
    """'I cannot verify this' and 'this is wrong' are different claims, and a tool
    that says the second when it means the first sends you hunting for an edit that
    never happened."""
    old = _run("old", {"a": True})
    old.config["dataset_version"] = "18c51bea5b3c"  # untagged, i.e. v1
    new = _run("new", {"a": True})
    new.config["dataset_version"] = "v2:31cae8715cac"

    warnings = " ".join(compare(old, new).warnings)
    assert "different algorithm versions" in warnings
    assert "The dataset changed" not in warnings


def test_comparison_still_reports_a_real_dataset_change():
    old = _run("old", {"a": True})
    old.config["dataset_version"] = "v2:aaaaaaaaaaaa"
    new = _run("new", {"a": True})
    new.config["dataset_version"] = "v2:bbbbbbbbbbbb"
    assert any("dataset changed" in w for w in compare(old, new).warnings)


# ---------------------------------------------------------------------------
# Tool variants
# ---------------------------------------------------------------------------
def test_variant_registry_changes_only_the_description():
    base = build_variant_registry("as_built")
    variant = build_variant_registry("hide_currency_list")

    assert variant.names == base.names
    base_specs = {s.name: s for s in base.specs}
    variant_specs = {s.name: s for s in variant.specs}
    assert variant_specs["convert_currency"].description != base_specs["convert_currency"].description
    # Same schema and same function: that is what makes it one variable.
    assert variant_specs["convert_currency"].parameters == base_specs["convert_currency"].parameters
    for name in base.names:
        if name != "convert_currency":
            assert variant_specs[name].description == base_specs[name].description


def test_variant_registry_does_not_mutate_the_shipped_tools():
    """A variant that edited the real ToolSpec in place would silently change every
    later lesson in the same process, including the baseline it is compared against."""
    build_variant_registry("hide_currency_list")
    fresh = build_variant_registry("as_built")
    description = {s.name: s.description for s in fresh.specs}["convert_currency"]
    assert "USD, EUR, GBP, INR, JPY, AUD, CAD" in description


def test_variant_registry_still_dispatches():
    variant = build_variant_registry("hide_currency_list")
    assert "convert_currency" in variant
    assert len(variant) == len(build_variant_registry("as_built"))


# ---------------------------------------------------------------------------
# The eval-change scorer
# ---------------------------------------------------------------------------
def test_cited_tool_limits_accepts_an_actual_tool_call():
    scorer = cited_tool_limits("convert_currency", ["supported currencies"])
    assert scorer(_Traj("whatever", ["convert_currency"])).passed


def test_cited_tool_limits_accepts_a_refusal_that_cites_the_documented_limits():
    scorer = cited_tool_limits("convert_currency", ["supported currencies"])
    result = scorer(_Traj("BTC is not in the supported currencies list.", []))
    assert result.passed


def test_cited_tool_limits_rejects_a_refusal_on_different_grounds():
    """The distinction the whole experiment rests on. 'I cannot look up the price of
    Bitcoin' is not the same claim as 'this tool does not accept BTC', and the second
    is the only one grounded in something the agent actually read."""
    scorer = cited_tool_limits("convert_currency", ["supported currencies"])
    result = scorer(_Traj("I don't have a way to look up the current price of Bitcoin.", []))
    assert not result.passed


def test_cited_tool_limits_rejects_an_empty_answer():
    scorer = cited_tool_limits("convert_currency", ["supported currencies"])
    assert not scorer(_Traj(None, [])).passed


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def test_measure_noise_finds_the_flipped_case():
    a = _run("r1", {"x": True, "y": True, "z": False})
    b = _run("r2", {"x": True, "y": False, "z": False})
    floor = measure_noise([a, b])
    assert floor.flipped == ["y"]
    assert floor.cases == 3
    assert floor.min_detectable == 2


def test_two_repeats_are_marked_provisional_and_do_not_license_a_floor_of_one():
    """The correction this project earned the hard way. 0/16 flips were measured
    across two repeats, which set the threshold to 1 case; hours later a case was
    found flaking at roughly 1 in 7, and the threshold of 1 had already reverted a
    genuine fix. Two clean repeats are weak evidence of stability, not evidence of
    determinism."""
    a = _run("r1", {"x": True, "y": False})
    floor = measure_noise([a, _run("r2", {"x": True, "y": False})])
    assert floor.flipped == []
    assert floor.provisional
    assert floor.min_detectable == 2  # not 1, despite measuring zero flips


def test_three_clean_repeats_are_trusted():
    outcomes = {"x": True, "y": False}
    floor = measure_noise([_run(f"r{i}", outcomes) for i in range(3)])
    assert not floor.provisional
    assert floor.min_detectable == 1


def test_measured_flips_are_never_reported_as_below_the_provisional_floor():
    a = _run("r1", {"x": True, "y": True, "z": True})
    b = _run("r2", {"x": False, "y": False, "z": True})
    floor = measure_noise([a, b])
    assert len(floor.flipped) == 2
    assert floor.min_detectable == 3  # measured beats the provisional minimum


def test_measure_noise_needs_two_runs():
    with pytest.raises(ValueError):
        measure_noise([_run("only", {"x": True})])


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------
def test_protected_regression_reverts_even_with_a_large_net_gain():
    """The rule that must never be quietly relaxed. Three arithmetic wins do not buy
    one fabrication regression: the agent has not got slightly worse, it has acquired
    a different and worse failure mode."""
    categories = {"impossible_share_price": "impossible"}
    before = _run("b", {"a": False, "b": False, "c": False, "impossible_share_price": True},
                  categories)
    after = _run("c", {"a": True, "b": True, "c": True, "impossible_share_price": False},
                 categories)
    judgement = decide(compare(before, after), NoiseFloor(run_names=["b", "c"], cases=4))
    assert judgement.decision is Decision.REVERT
    assert "protected" in judgement.reasons[0]


def test_net_zero_trade_with_a_predicted_break_reverts():
    before = _run("b", {"a": True, "b": False})
    after = _run("c", {"a": False, "b": True})
    judgement = decide(compare(before, after), None, unpredicted_breaks=[])
    assert judgement.decision is Decision.REVERT
    assert "traded" in judgement.reasons[0]
    assert judgement.recheck == []


def test_net_zero_trade_on_an_unpredicted_break_asks_for_a_recheck():
    """The rule change this project paid for. `verify_first` fixed the case it aimed
    at and broke one nobody predicted, so the rule reverted it -- and the regression
    reproduced about one run in seven. At one case, a flake and a regression look
    identical, and re-testing the case is far cheaper than discarding a working fix."""
    before = _run("b", {"a": True, "b": False})
    after = _run("c", {"a": False, "b": True})
    judgement = decide(compare(before, after), None, unpredicted_breaks=["a"])
    assert judgement.decision is Decision.INCONCLUSIVE
    assert judgement.recheck == ["a"]
    assert "--recheck" in judgement.reasons[0]


def test_a_protected_break_still_reverts_even_when_unpredicted():
    """The recheck rule must not become a loophole around the safety override."""
    categories = {"impossible_share_price": "impossible"}
    before = _run("b", {"a": False, "impossible_share_price": True}, categories)
    after = _run("c", {"a": True, "impossible_share_price": False}, categories)
    judgement = decide(
        compare(before, after), None, unpredicted_breaks=["impossible_share_price"]
    )
    assert judgement.decision is Decision.REVERT
    assert judgement.recheck == []


def test_no_movement_reverts():
    before = _run("b", {"a": True, "b": False})
    after = _run("c", {"a": True, "b": False})
    judgement = decide(compare(before, after), None)
    assert judgement.decision is Decision.REVERT
    assert "no case changed" in judgement.reasons[0]


def test_gain_below_the_noise_floor_is_inconclusive_not_a_win():
    """The rule that costs discipline: it refuses a win you would like to bank."""
    before = _run("b", {"a": False, "b": True, "c": True})
    after = _run("c", {"a": True, "b": True, "c": True})
    floor = NoiseFloor(run_names=["x", "y"], flipped=["b", "c"], cases=3)  # min_detectable 3
    judgement = decide(compare(before, after), floor)
    assert judgement.decision is Decision.INCONCLUSIVE
    assert "below the 3-case threshold" in judgement.reasons[0]


def test_gain_at_the_threshold_is_kept():
    before = _run("b", {"a": False, "b": False, "c": True})
    after = _run("c", {"a": True, "b": True, "c": True})
    floor = NoiseFloor(run_names=["x", "y"], flipped=["c"], cases=3)  # min_detectable 2
    judgement = decide(compare(before, after), floor)
    assert judgement.decision is Decision.KEEP


def test_a_win_that_more_than_doubles_cost_per_success_is_downgraded():
    before = _run("b", {"a": False, "b": False, "c": True}, tokens=1000)
    after = _run("c", {"a": True, "b": True, "c": True}, tokens=9000)
    floor = NoiseFloor(run_names=["x", "y"], flipped=[], cases=3)
    judgement = decide(compare(before, after), floor)
    assert judgement.decision is Decision.INCONCLUSIVE
    assert any("cost per successful answer rose" in r for r in judgement.reasons)


def test_a_win_exactly_at_the_cost_ceiling_is_still_kept():
    """Pins which side of the boundary the rule sits on. A threshold with an
    undocumented edge is a threshold someone will argue about later."""
    before = _run("b", {"a": False, "b": False, "c": True}, tokens=1000)
    after = _run("c", {"a": True, "b": True, "c": True}, tokens=6000)
    judgement = decide(compare(before, after), NoiseFloor(run_names=["x", "y"], cases=3))
    assert judgement.cost_ratio == pytest.approx(2.0)
    assert judgement.decision is Decision.KEEP


def test_a_reproducible_win_still_carries_a_sample_size_caveat():
    """The gap the measured noise floor exposed. Zero flips across repeats answers
    "is this reproducible?", and people immediately read it as "does this generalise?".
    With 16 cases the Wilson intervals overlap heavily, so the honest answer to the
    second question is still no -- reported as a caveat, not as a veto."""
    before = _run("b", {f"c{i}": True for i in range(15)} | {"c15": False})
    after = _run("c", {f"c{i}": True for i in range(16)})
    # Three repeats, so the floor is not provisional and min_detectable is 1. A +1
    # gain therefore clears the variance bar -- and still cannot be shown to generalise.
    floor = NoiseFloor(run_names=["x", "y", "z"], flipped=[], cases=16)
    judgement = decide(compare(before, after), floor)

    assert judgement.decision is Decision.KEEP
    assert any("intervals overlap" in c for c in judgement.caveats)
    assert not any("intervals overlap" in r for r in judgement.reasons)


def test_missing_noise_floor_uses_a_stated_default_rather_than_zero():
    before = _run("b", {"a": False, "b": True})
    after = _run("c", {"a": True, "b": True})
    judgement = decide(compare(before, after), None)
    assert judgement.decision is Decision.INCONCLUSIVE
    assert "no noise floor measured" in judgement.reasons[0]


# ---------------------------------------------------------------------------
# Prediction coherence and scoring
# ---------------------------------------------------------------------------
def test_predicting_a_fix_for_an_already_passing_case_is_flagged():
    baseline = _run("baseline", {"passing": True, "failing": False})
    experiment = Experiment(
        id="e", hypothesis="h", variable="prompt",
        config=replace(BASELINE, prompt="terse"), predicts_fixed=["passing"],
    )
    assert any("already passes" in p for p in check_predictions(experiment, baseline))


def test_predicting_a_break_for_an_already_failing_case_is_flagged():
    baseline = _run("baseline", {"passing": True, "failing": False})
    experiment = Experiment(
        id="e", hypothesis="h", variable="prompt",
        config=replace(BASELINE, prompt="terse"), predicts_broken=["failing"],
    )
    assert any("already failing" in p for p in check_predictions(experiment, baseline))


def test_predicting_an_unknown_case_is_flagged():
    baseline = _run("baseline", {"passing": True})
    experiment = Experiment(
        id="e", hypothesis="h", variable="prompt",
        config=replace(BASELINE, prompt="terse"), predicts_fixed=["nope"],
    )
    assert any("not in the baseline" in p for p in check_predictions(experiment, baseline))


def test_registry_predictions_are_coherent_with_the_committed_baseline():
    """Guards against a stale registry: if the baseline improves, a prediction that
    an already-fixed case will be fixed again is now incoherent and should be edited
    rather than left to quietly pass."""
    try:
        baseline = EvalRun.load("baseline")
    except FileNotFoundError:
        pytest.skip("no committed baseline run")
    for experiment in EXPERIMENTS:
        assert check_predictions(experiment, baseline) == [], experiment.id


def _attempt(**kwargs) -> Attempt:
    defaults = dict(
        experiment_id="e", created_at="2026-01-01T00:00:00+00:00", variable="prompt",
        hypothesis="h", baseline="baseline", candidate="exp_e",
        predicted_fixed=[], predicted_broken=[], predicted_no_change=False,
        actual_fixed=[], actual_broken=[], baseline_passed=15, candidate_passed=15,
        total_cases=16, decision="revert", reasons=[], cost_per_success_before=0.0,
        cost_per_success_after=0.0, tokens_spent=0,
    )
    return Attempt(**(defaults | kwargs))


def test_prediction_hit_requires_getting_the_regressions_right_too():
    """Graded strictly on purpose. 'I called the win but missed a regression' is how
    you keep believing your intuition is good."""
    half_right = _attempt(predicted_fixed=["a"], actual_fixed=["a"], actual_broken=["b"])
    assert not half_right.prediction_hit
    assert half_right.surprises == ["-b"]

    exact = _attempt(predicted_fixed=["a"], predicted_broken=["b"],
                     actual_fixed=["a"], actual_broken=["b"])
    assert exact.prediction_hit
    assert exact.surprises == []


def test_a_control_hits_only_when_nothing_moved():
    assert _attempt(predicted_no_change=True).prediction_hit
    assert not _attempt(predicted_no_change=True, actual_fixed=["a"]).prediction_hit


def test_an_attempt_that_predicted_nothing_is_never_a_hit():
    """Found by a test rather than by thinking. Two empty prediction lists compared
    equal to two empty outcomes, so an attempt that bet on nothing scored a hit and
    inflated the one number this lesson exists to produce."""
    assert not _attempt().prediction_hit


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------
def test_changelog_round_trips(tmp_path):
    log = Changelog()
    log.append(_attempt(experiment_id="one", decision="revert"))
    log.append(_attempt(experiment_id="two", decision="keep",
                        predicted_fixed=["a"], actual_fixed=["a"]))
    path = log.save(tmp_path / "attempts.json")

    reloaded = Changelog.load(path)
    assert [a.experiment_id for a in reloaded.attempts] == ["one", "two"]
    assert [a.decision for a in reloaded.kept] == ["keep"]
    assert reloaded.predictions_hit == 1
    assert reloaded.hit_rate == 0.5


def test_changelog_keeps_repeated_attempts_at_the_same_experiment(tmp_path):
    """Overwriting would destroy the most interesting record in the file: the same
    change measured twice with different outcomes, which is direct evidence of noise."""
    log = Changelog()
    log.append(_attempt(experiment_id="same", actual_fixed=["a"]))
    log.append(_attempt(experiment_id="same", actual_fixed=[]))
    assert len(log.for_experiment("same")) == 2


def test_build_attempt_records_the_prediction_and_the_outcome():
    experiment = Experiment(
        id="e", hypothesis="h", variable="prompt",
        config=replace(BASELINE, prompt="terse"), predicts_fixed=["a"],
    )
    before = _run("b", {"a": False, "b": True})
    after = _run("c", {"a": True, "b": True})
    comparison = compare(before, after)
    attempt = build_attempt(experiment, comparison, decide(comparison, None))

    assert attempt.predicted_fixed == ["a"]
    assert attempt.actual_fixed == ["a"]
    assert attempt.prediction_hit
    assert attempt.net == 1
    assert attempt.tokens_spent == after.tokens_actually_spent


def test_a_fully_cached_run_reports_zero_tokens_spent():
    """`total_tokens` is what the configuration costs cold and feeds cost per success.
    `tokens_actually_spent` is what producing this file cost. Reporting the first as the
    second made lesson 9's changelog claim 130,000 tokens for about half that."""
    run = _run("cached", {"a": True, "b": True})
    assert run.total_tokens > 0
    run.config["cached_cases"] = ["a", "b"]
    assert run.tokens_actually_spent == 0

    run.config["cached_cases"] = ["a"]
    assert run.tokens_actually_spent == run.result_for("b").total_tokens
