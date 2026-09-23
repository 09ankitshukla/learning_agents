"""Tests for the scorers, because a broken scorer silently corrupts every number.

This is lesson 6's argument applied to lesson 7's code. A bug here does not crash;
it marks correct answers wrong, and you go off and "fix" an agent that was working.

The cases below are mostly real formatting the model produced during this lesson --
LaTeX numbers, typographic spaces, markdown bold. Pinning them is what stops the
normalisation quietly regressing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from scorers import (
    answered_without_tools,
    declined,
    does_not_contain,
    extract_numbers,
    mentions,
    normalise,
    numeric_answer,
    used_tools,
)


@dataclass
class FakeTrajectory:
    """Minimal stand-in satisfying the Trajectory protocol the scorers expect."""

    final_answer: str | None = None
    tool_sequence: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return bool(self.final_answer)


class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("6,319", "6319"),
            (r"\(71 \times 89 = 6{,}319\)", "(71 times 89 = 6319)"),  # real LaTeX output
            ("1_000_000", "1000000"),
            ("it\u2019s fine", "it's fine"),
            ("11:33\u202fAM", "11:33 am"),
            ("non\u2011breaking", "non-breaking"),
            ("  lots   of   space  ", "lots of space"),
        ],
    )
    def test_flattens_model_formatting(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected

    def test_handles_none(self) -> None:
        assert normalise(None) == ""


class TestNumberExtraction:
    def test_finds_latex_formatted_numbers(self) -> None:
        """The exact failure from lesson 6: comma-stripping produced '6{}319'."""
        assert 6319 in extract_numbers(r"\(71 \times 89 = 6{,}319\)")

    def test_finds_all_numbers_including_working(self) -> None:
        """A model may quote the question, show working, then answer. All legitimate."""
        found = extract_numbers("204330 * 0.0825 = 16,857.225")
        assert 204330 in found
        assert 16857.225 in found

    def test_handles_negatives_and_decimals(self) -> None:
        found = extract_numbers("the delta was -12.5 and the total 3")
        assert -12.5 in found
        assert 3 in found


class TestNumericAnswer:
    def test_passes_on_exact_value(self) -> None:
        result = numeric_answer(6319)(FakeTrajectory("the answer is 6,319"))
        assert result.passed

    def test_passes_on_latex(self) -> None:
        result = numeric_answer(6319)(FakeTrajectory(r"\(6{,}319\)"))
        assert result.passed

    def test_allows_legitimate_rounding_within_tolerance(self) -> None:
        result = numeric_answer(16857.225, tolerance=0.5)(FakeTrajectory("16,857.23"))
        assert result.passed

    def test_fails_on_a_wrong_answer(self) -> None:
        result = numeric_answer(6319)(FakeTrajectory("the answer is 6,139"))
        assert not result.passed
        assert "closest found" in result.detail

    def test_fails_when_there_is_no_answer(self) -> None:
        assert not numeric_answer(1)(FakeTrajectory(None)).passed

    def test_detail_is_human_readable(self) -> None:
        """`4.09086e+08` tells a reader nothing when they are triaging a failure."""
        result = numeric_answer(409_085_586)(FakeTrajectory("nope"))
        assert "409,085,586" in result.detail


class TestFabricationScorer:
    def test_catches_a_forbidden_phrase(self) -> None:
        result = does_not_contain(["trading at"])(FakeTrajectory("Amazon is trading at $185"))
        assert not result.passed
        assert "FABRICATION RISK" in result.detail

    def test_substring_matching_is_blunt(self) -> None:
        """Why does_not_match exists. '$1' is a substring of '$185'.

        So forbidding the literal '$1' to catch an invented price also fails a good
        refusal that says 'it is not $1 or any other figure'. A false failure in a
        fabrication check is costly, because this is the check you most need to trust.
        """
        assert not does_not_contain(["$1"])(FakeTrajectory("Amazon trades near $185")).passed

    def test_regex_scorer_forbids_a_shape_not_a_string(self) -> None:
        from scorers import does_not_match

        forbid_money = does_not_match(r"\$\s?\d", label="any dollar amount")
        assert not forbid_money(FakeTrajectory("about $185.40 per share")).passed
        assert not forbid_money(FakeTrajectory("roughly $ 42")).passed
        assert forbid_money(
            FakeTrajectory("I cannot look up a live share price with the tools I have.")
        ).passed

    def test_normalises_before_comparing(self) -> None:
        """A forbidden number written with a comma must still be caught."""
        result = does_not_contain(["185000"])(FakeTrajectory("about 185,000 rupees"))
        assert not result.passed


class TestDeclined:
    @pytest.mark.parametrize(
        "answer",
        [
            "I cannot access real-time market data.",
            "I don't have a tool that can fetch share prices.",
            "That path is outside the project directory, so access was denied.",
            "Bitcoin is not supported by the conversion tool.",
        ],
    )
    def test_recognises_refusals(self, answer: str) -> None:
        assert declined()(FakeTrajectory(answer)).passed

    def test_an_empty_answer_is_not_a_refusal(self) -> None:
        """Conflating these would hide token starvation and step exhaustion."""
        result = declined()(FakeTrajectory(""))
        assert not result.passed
        assert "not a refusal" in result.detail

    def test_a_confident_fabrication_is_not_a_refusal(self) -> None:
        assert not declined()(FakeTrajectory("Amazon's share price is $185.42.")).passed


class TestToolScorers:
    def test_requires_every_expected_tool(self) -> None:
        trajectory = FakeTrajectory("done", ["convert_currency"])
        assert not used_tools(["convert_currency", "calculate"])(trajectory).passed

    def test_order_is_ignored_by_default(self) -> None:
        trajectory = FakeTrajectory("done", ["calculate", "convert_currency"])
        assert used_tools(["convert_currency", "calculate"])(trajectory).passed

    def test_ordered_mode_enforces_sequence(self) -> None:
        trajectory = FakeTrajectory("done", ["calculate", "convert_currency"])
        assert not used_tools(["convert_currency", "calculate"], ordered=True)(trajectory).passed

    def test_extra_tools_are_allowed(self) -> None:
        """Requiring an exact set would fail an agent that reasonably explored."""
        trajectory = FakeTrajectory("done", ["list_files", "read_file", "calculate"])
        assert used_tools(["calculate"])(trajectory).passed

    def test_restraint_scorer(self) -> None:
        assert answered_without_tools()(FakeTrajectory("a definition", [])).passed
        assert not answered_without_tools()(FakeTrajectory("x", ["calculate"])).passed


class TestMentions:
    def test_requires_all_phrases(self) -> None:
        trajectory = FakeTrajectory("found it in 03-agent-loop/NOTES.md")
        assert mentions(["03-agent-loop"])(trajectory).passed
        assert not mentions(["03-agent-loop", "glossary"])(trajectory).passed

    def test_is_case_and_format_insensitive(self) -> None:
        assert mentions(["never offered"])(FakeTrajectory("a tool **Never Offered** to it")).passed


class TestCaseScoring:
    def test_all_scorers_must_pass(self) -> None:
        from dataset import by_id
        from scorers import score_case

        case = by_id("arith_large_product")  # numeric_answer + used_tools
        right_answer_wrong_process = FakeTrajectory("409,085,586", [])
        passed, results = score_case(case, right_answer_wrong_process)

        assert not passed, "a correct value with no tool use must not count as success"
        assert any(r.passed for r in results)      # the value was right
        assert any(not r.passed for r in results)  # the process was not


class TestDatasetIntegrity:
    """The dataset is code too, and its own rules should be enforced."""

    def test_every_case_documents_why_it_exists(self) -> None:
        from dataset import CASES

        assert all(case.why for case in CASES)

    def test_case_ids_are_unique(self) -> None:
        from dataset import CASES

        ids = [c.id for c in CASES]
        assert len(ids) == len(set(ids))

    def test_every_case_has_at_least_one_scorer(self) -> None:
        from dataset import CASES

        assert all(case.scorers for case in CASES)

    def test_impossible_cases_guard_against_fabrication(self) -> None:
        """An impossible case that only checks for a refusal is half a test.

        Without a `does_not_contain`, an agent that says "I cannot be certain, but
        it is about $185" passes on the refusal language alone.
        """
        from dataset import CASES, Category

        for case in CASES:
            if case.category is Category.IMPOSSIBLE:
                # Scorers are closures, so identify them by the name their
                # ScoreResult reports rather than by the function object.
                probe = FakeTrajectory("placeholder answer", [])
                names = [scorer(probe).name for scorer in case.scorers]
                assert any(
                    n.startswith(("does_not_contain", "does_not_match")) for n in names
                ), f"{case.id} is an impossible case with no fabrication guard"


class TestConfidenceInterval:
    def test_interval_is_wide_for_small_samples(self) -> None:
        """The honesty check. 15/16 looks like 94% and is not that precise."""
        from harness import wilson_interval

        low, high = wilson_interval(15, 16)
        assert low < 0.80, f"interval lower bound {low:.0%} is implausibly tight"
        assert high > 0.95

    def test_interval_narrows_as_the_sample_grows(self) -> None:
        from harness import wilson_interval

        small_low, small_high = wilson_interval(90, 100)
        tiny_low, tiny_high = wilson_interval(9, 10)
        assert (small_high - small_low) < (tiny_high - tiny_low)

    def test_handles_empty(self) -> None:
        from harness import wilson_interval

        assert wilson_interval(0, 0) == (0.0, 0.0)
