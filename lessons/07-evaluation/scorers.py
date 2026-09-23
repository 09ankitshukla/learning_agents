"""Deterministic scorers: turning "is this answer good?" into a boolean.

Every scorer here is plain Python -- no model involved. That is deliberate and it
is the right order to build in: exhaust cheap, unambiguous checks before reaching
for a model to grade a model. Deterministic scorers are free, instant, reproducible
and never biased. Lesson 8 adds LLM-as-judge for the genuinely subjective cases,
and it will be a much smaller set than you expect.

The recurring difficulty is that models format answers freely. Lesson 6 taught this
painfully: a live test asserted `"6319" in answer` after stripping commas, and the
model wrote LaTeX `6{,}319`, which became `6{}319`. The arithmetic was perfect and
the assertion was wrong.

So `numeric_answer` normalises aggressively and extracts every number it can find,
rather than string-matching one formatting. **A scorer that is too strict measures
formatting instead of correctness, and you will not notice, because the failures
look real.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class ScoreResult:
    passed: bool
    #: What was checked and what was found. Shown for every failure, because a
    #: score without an explanation cannot be acted on.
    detail: str
    #: Name of the scorer, for per-scorer reporting.
    name: str = ""


class Trajectory(Protocol):
    """The subset of lesson 3's Trajectory that scorers may look at."""

    final_answer: str | None
    tool_sequence: list[str]

    @property
    def succeeded(self) -> bool: ...


Scorer = Callable[["Trajectory"], ScoreResult]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
#: Characters models use that break naive matching. Typographic quotes, LaTeX
#: grouping braces, non-breaking spaces, thousands separators.
_NOISE = str.maketrans(
    {
        ",": "",
        "_": "",
        "{": "",
        "}": "",
        "\\": "",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2009": " ",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
)


def normalise(text: str | None) -> str:
    """Flatten the formatting a model might apply, so checks test meaning.

    Removes thousands separators and LaTeX braces (so `6{,}319` becomes `6319`),
    folds typographic punctuation to ASCII, collapses whitespace, lowercases.
    """
    if not text:
        return ""
    cleaned = text.translate(_NOISE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().lower()


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numbers(text: str | None) -> list[float]:
    """Every number in the text, after normalisation.

    Extracting all of them and asking "is the right one present?" is far more
    robust than trying to locate *the* answer. A model may show its working, quote
    the question's figures, and give the result -- all legitimate.
    """
    out: list[float] = []
    for match in _NUMBER.finditer(normalise(text)):
        try:
            out.append(float(match.group()))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
def _pretty(value: float) -> str:
    """Readable number for a report. `4.09086e+08` tells a reader nothing.

    Scorer output is read by a human deciding whether a failure matters, so it is
    worth the four lines to print 409,085,586 instead.
    """
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def numeric_answer(expected: float, tolerance: float = 0.01) -> Scorer:
    """Pass if the expected number appears anywhere in the answer.

    Tolerance is absolute, and defaults to something tight. Rounding is legitimate
    (16,857.225 shown as 16,857.23) and should not fail; a wrong answer should.
    """

    def score(trajectory: Trajectory) -> ScoreResult:
        found = extract_numbers(trajectory.final_answer)
        hit = any(abs(value - expected) <= tolerance for value in found)
        # Show a few nearby numbers, not all of them -- a long answer yields dozens.
        nearest = [_pretty(v) for v in sorted(found, key=lambda v: abs(v - expected))[:3]]
        return ScoreResult(
            passed=hit,
            detail=(
                f"expected {_pretty(expected)} (+/-{_pretty(tolerance)}); "
                f"{'found it' if hit else f'closest found: {nearest or None}'}"
            ),
            name=f"numeric_answer({_pretty(expected)})",
        )

    return score


def mentions(required: list[str]) -> Scorer:
    """Pass if every required phrase appears, after normalisation.

    Use sparingly and for facts, not phrasing. Requiring a specific word is a
    fragile proxy for requiring a specific meaning, which is what lesson 8's judge
    is for.
    """

    def score(trajectory: Trajectory) -> ScoreResult:
        answer = normalise(trajectory.final_answer)
        missing = [phrase for phrase in required if normalise(phrase) not in answer]
        return ScoreResult(
            passed=not missing,
            detail="all phrases present" if not missing else f"missing: {missing}",
            name=f"mentions({required})",
        )

    return score


def does_not_contain(forbidden: list[str]) -> Scorer:
    """Pass if none of these appear. For catching fabrication.

    The most valuable scorer in the set, and the one usually missing. A case that
    only checks for the right answer cannot distinguish "declined correctly" from
    "invented something plausible". This is what catches a made-up share price or
    fabricated file contents.
    """

    def score(trajectory: Trajectory) -> ScoreResult:
        answer = normalise(trajectory.final_answer)
        found = [phrase for phrase in forbidden if normalise(phrase) in answer]
        return ScoreResult(
            passed=not found,
            detail="none of the forbidden strings present"
            if not found
            else f"FABRICATION RISK, found: {found}",
            name=f"does_not_contain({forbidden})",
        )

    return score


def does_not_match(pattern: str, *, label: str = "") -> Scorer:
    """Pass if a regex does not match. For forbidding a *shape* of answer.

    Added because substring matching is too blunt for fabrication checks. Forbidding
    the literal `"$1"` to catch an invented share price also fires on a perfectly
    good refusal that happens to say "it is not $1 or any other figure" -- and a
    false failure in a fabrication check is expensive, because it is the check you
    most need to trust.

    A pattern like `\\$\\s?\\d` forbids *any* dollar amount, which is what the case
    actually means.
    """
    compiled = re.compile(pattern, re.IGNORECASE)

    def score(trajectory: Trajectory) -> ScoreResult:
        answer = normalise(trajectory.final_answer)
        found = compiled.search(answer)
        return ScoreResult(
            passed=found is None,
            detail=(
                f"no match for {label or pattern!r}"
                if found is None
                else f"FABRICATION RISK, matched {label or pattern!r}: {found.group()!r}"
            ),
            name=f"does_not_match({label or pattern})",
        )

    return score


#: Phrases that indicate an agent recognised it could not complete a task.
#: Deliberately broad: the aim is to detect the *stance*, and a false negative here
#: (marking a genuine refusal as a failure) is worse than a false positive.
_REFUSAL_MARKERS = (
    "cannot", "can't", "cant", "unable", "not able", "no tool", "do not have",
    "don't have", "dont have", "no access", "not supported", "unsupported",
    "not available", "outside", "denied", "refused", "not permitted",
    "restricted", "i do not know", "i don't know", "real-time", "real time",
    "not possible", "failed", "error",
)


def declined() -> Scorer:
    """Pass if the agent said it could not do the task.

    Scoring a refusal with keywords is crude, and worth being honest about: an
    agent could refuse in words this misses, or could use the word "cannot" while
    still fabricating an answer. That is why the impossible cases pair `declined()`
    with `does_not_contain(...)` -- the second catches what the first cannot.

    Judging a refusal properly is a job for lesson 8's rubric-based judge.
    """

    def score(trajectory: Trajectory) -> ScoreResult:
        answer = normalise(trajectory.final_answer)
        if not answer:
            # No answer at all is not a refusal. It is a different failure, and
            # conflating them would hide token starvation and step exhaustion.
            return ScoreResult(False, "no answer produced (not a refusal)", "declined")
        hits = [m for m in _REFUSAL_MARKERS if m in answer]
        return ScoreResult(
            passed=bool(hits),
            detail=f"refusal signalled by {hits[:3]}" if hits else "no refusal language found",
            name="declined",
        )

    return score


def used_tools(expected: list[str], *, ordered: bool = False) -> Scorer:
    """Pass if the expected tools were used. Measures process, not outcome.

    Separate from success on purpose. An agent can reach the right answer the wrong
    way -- computing arithmetic in its head instead of calling `calculate` -- and
    that matters, because it is right by luck rather than by construction.

    `ordered=False` by default. Requiring exact order makes the scorer brittle for
    little gain; a genuinely order-dependent task is better expressed as two cases.
    """

    def score(trajectory: Trajectory) -> ScoreResult:
        actual = trajectory.tool_sequence
        if ordered:
            passed = _is_subsequence(expected, actual)
        else:
            passed = all(tool in actual for tool in expected)
        return ScoreResult(
            passed=passed,
            detail=f"expected {expected}, used {actual or '(none)'}",
            name=f"used_tools({expected})",
        )

    return score


def answered_without_tools() -> Scorer:
    """Pass if no tool was used. Measures restraint."""

    def score(trajectory: Trajectory) -> ScoreResult:
        actual = trajectory.tool_sequence
        return ScoreResult(
            passed=not actual,
            detail="no tools used" if not actual else f"unnecessarily used {actual}",
            name="answered_without_tools",
        )

    return score


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """True if `needle` appears in order within `haystack`, gaps allowed."""
    iterator = iter(haystack)
    return all(item in iterator for item in needle)


# ---------------------------------------------------------------------------
def score_case(case, trajectory: Trajectory) -> tuple[bool, list[ScoreResult]]:
    """Run every scorer for a case. Success requires all of them to pass.

    All-must-pass rather than a weighted average, because partial credit on a
    12-case set produces a number that looks precise and means very little. A case
    either did the job or it did not; the per-scorer detail explains which part
    failed.
    """
    results = [scorer(trajectory) for scorer in case.scorers]
    return all(r.passed for r in results), results
