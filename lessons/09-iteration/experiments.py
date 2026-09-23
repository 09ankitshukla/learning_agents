"""Experiments: a hypothesis, exactly one changed variable, and a prediction.

Lessons 7 and 8 built instruments. This file is about the *discipline* of using
them, and the discipline is almost entirely about what you write down before you
run anything.

**One variable.** `Config` enumerates everything lesson 9 can change, and
`Experiment.validate()` refuses to run if more than one field differs from the
baseline. That is not pedantry. Change the prompt and the step cap together, see the
score rise, and you have learned nothing you can act on -- you cannot ship half of
it, and you cannot explain the other half. The rule is enforced in code because it
is the rule everyone breaks when they are in a hurry.

**A prediction, recorded first.** `predicts_fixed` and `predicts_broken` are filled
in before the run and never edited afterwards. This turns "I had a feeling that
would work" into a number: `iterate.py --scorecard` reports how often the hypothesis
was right. Everyone believes they have good intuition about prompts. Almost nobody
has measured it.

**Rejected experiments stay here.** A registry of only the things that worked is a
trap: the same idea gets retried in six months because nothing records that it was
tried and lost. `status` and `rejected_because` keep the failures visible.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable

from llmkit import ToolRegistry
from llmkit.tools import Tool

_LESSONS = Path(__file__).resolve().parents[1]
for _folder in ("02-tool-calling", "03-agent-loop", "07-evaluation"):
    _path = str(_LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from toolset import build_registry  # noqa: E402

from dataset import CASES, EvalCase  # noqa: E402
from evaluate import PROMPTS as LESSON7_PROMPTS  # noqa: E402
from scorers import ScoreResult, Scorer, normalise  # noqa: E402

#: Lesson 7's committed runs were made on the small model, which has its own quota.
#: Every comparison here must use the same one or it is not a comparison.
EVAL_MODEL = "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
# Lesson 7's three are imported rather than copied, so "default" here is byte for
# byte the prompt behind the committed `baseline` run. Retyping it would be an
# invisible way to make every comparison in this lesson meaningless.
PROMPTS: dict[str, str | None] = dict(LESSON7_PROMPTS)

PROMPTS["verify_first"] = (
    "You solve problems using tools.\n"
    "Before telling the user you cannot do something, attempt the most relevant "
    "tool and read what it says. A tool's error message is evidence; your "
    "expectation is not.\n"
    "If the tool confirms it cannot help, say so plainly and quote the reason.\n"
    "When you have what you need, answer concisely."
)

PROMPTS["verify_conversion"] = (
    "You solve problems using tools.\n"
    "For any conversion, rate or unit lookup, call the tool before deciding whether "
    "it is possible. Do not assume from a tool's description which values it "
    "accepts; the tool will tell you, and its error is the evidence you report.\n"
    "For anything else, use your judgement about whether a tool is needed.\n"
    "When you have what you need, answer concisely."
)


# ---------------------------------------------------------------------------
# Tool-description variants
# ---------------------------------------------------------------------------
# A tool's description is the highest-leverage text in an agent (llmkit's ToolSpec
# says so), and it is almost never treated as a variable you can A/B. It is one
# here. Note this is the variable lesson 7's cache key could not see -- see
# `Config.residual_key`.
TOOL_VARIANTS: dict[str, dict[str, str]] = {
    "as_built": {},
    "hide_currency_list": {
        # The shipped description ends "Supported currencies: USD, EUR, GBP, INR,
        # JPY, AUD, CAD." Removing that list forces the agent to call the tool to
        # discover what it accepts. A diagnostic, not a proposal: see the experiment.
        "convert_currency": (
            "Convert an amount from one currency to another using a fixed rate table."
        ),
    },
    "loud_currency_list": {
        "convert_currency": (
            "Convert an amount from one currency to another using a fixed rate table. "
            "Supported currencies: USD, EUR, GBP, INR, JPY, AUD, CAD. Any other code, "
            "including cryptocurrencies, is rejected with an error naming the "
            "supported set. Call this tool to check rather than assuming."
        ),
    },
}


def build_variant_registry(variant: str) -> ToolRegistry:
    """The same six tools, with some descriptions swapped.

    Same functions, same schemas, same allowlist -- only the prose the model reads.
    Keeping the functions identical is what makes this a single-variable change.
    """
    overrides = TOOL_VARIANTS[variant]
    base = build_registry()
    if not overrides:
        return base
    return ToolRegistry(
        [
            Tool(replace(tool.spec, description=overrides[tool.spec.name]), tool.fn)
            if tool.spec.name in overrides
            else tool
            for tool in base.tools
        ]
    )


# ---------------------------------------------------------------------------
# Dataset variants: changing the *eval* is a legitimate move
# ---------------------------------------------------------------------------
def cited_tool_limits(tool: str, evidence: list[str]) -> Scorer:
    """Pass if the agent used the tool, OR refused citing the tool's stated limits.

    Written for the one experiment in this lesson that changes no agent code at all.

    The reasoning behind it: `currency_unsupported` demands `used_tools` so that a
    refusal is grounded in evidence rather than in a hunch. But `convert_currency`'s
    description already lists the seven currencies it accepts, so an agent that reads
    it carefully has evidence *without* calling. On that argument the case is unfair
    and the eval, not the agent, is what should change.

    This scorer encodes that argument honestly: refusing is acceptable only if the
    answer shows it came from the documented limits. "I cannot look up the price of
    Bitcoin" is a different claim, and a weaker one, and this will not accept it.
    """
    label = f"cited_tool_limits({tool})"

    def score(trajectory) -> ScoreResult:
        if tool in trajectory.tool_sequence:
            return ScoreResult(True, f"called {tool}", label)
        answer = normalise(trajectory.final_answer)
        if not answer:
            return ScoreResult(False, f"no answer and {tool} not called", label)
        found = [phrase for phrase in evidence if normalise(phrase) in answer]
        return ScoreResult(
            passed=bool(found),
            detail=(
                f"did not call {tool} but cited its limits via {found[:2]}"
                if found
                else f"did not call {tool} and cited none of {evidence}"
            ),
            name=label,
        )

    score.label = label
    return score


def _documented_refusal(cases: list[EvalCase]) -> list[EvalCase]:
    """Replace `currency_unsupported`'s tool requirement with the scorer above."""
    from scorers import scorer_label

    out: list[EvalCase] = []
    for case in cases:
        if case.id != "currency_unsupported":
            out.append(case)
            continue
        swapped = [
            cited_tool_limits(
                "convert_currency",
                [
                    "supported currencies",
                    "usd, eur",
                    "fixed rate table",
                    "rate table",
                    "not in the supported",
                    "only supports",
                ],
            )
            if scorer_label(scorer).startswith("used_tools")
            else scorer
            for scorer in case.scorers
        ]
        out.append(replace(case, scorers=swapped))
    return out


DATASET_VARIANTS: dict[str, Callable[[list[EvalCase]], list[EvalCase]]] = {
    "as_built": lambda cases: list(cases),
    "documented_refusal": _documented_refusal,
}


# ---------------------------------------------------------------------------
# Config: everything that can vary, and nothing that cannot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """The complete set of knobs. Deliberately small.

    A config object with twenty fields invites changing four of them at once. Five
    fields, each with a named effect, is a shape you can reason about -- and the
    `diff` below is what makes the one-variable rule checkable.
    """

    prompt: str = "default"
    model: str = EVAL_MODEL
    #: Added to every case's max_steps. Applied by rebuilding the case list, so
    #: lesson 7's cache key picks it up for free -- it already hashes case.max_steps.
    step_bonus: int = 0
    tool_variant: str = "as_built"
    #: Changes only how results are *scored*. Costs nothing: lesson 7 re-scores from
    #: cached executions, so an eval change is the one experiment that is always free.
    dataset_variant: str = "as_built"

    # -- derived --------------------------------------------------------
    @property
    def system_prompt(self) -> str | None:
        return PROMPTS[self.prompt]

    @property
    def residual_key(self) -> str:
        """What lesson 7's cache key cannot see, and therefore what we must tell it.

        Lesson 7 hashes the question, model, step cap and system prompt. Of the five
        fields here, four are already covered: `prompt` and `model` directly,
        `step_bonus` because it rewrites `case.max_steps`, and `dataset_variant`
        because it changes scoring only, which is recomputed every time by design.

        `tool_variant` is the one that is invisible to it, so it is the only thing
        in this key. Returning "" for the shipped tools is deliberate and load
        bearing: it keeps every execution cached before lesson 9 valid, so the
        baseline costs nothing to reproduce.
        """
        if self.tool_variant == "as_built":
            return ""
        payload = json.dumps(TOOL_VARIANTS[self.tool_variant], sort_keys=True)
        return "tools:" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    def diff(self, other: Config) -> list[str]:
        """Field names where these two configs differ."""
        return [
            name
            for name in ("prompt", "model", "step_bonus", "tool_variant", "dataset_variant")
            if getattr(self, name) != getattr(other, name)
        ]

    def build_registry(self) -> ToolRegistry:
        return build_variant_registry(self.tool_variant)

    def build_cases(self, base: list[EvalCase] | None = None) -> list[EvalCase]:
        cases = DATASET_VARIANTS[self.dataset_variant](base or CASES)
        if self.step_bonus:
            cases = [replace(c, max_steps=c.max_steps + self.step_bonus) for c in cases]
        return cases

    def describe(self) -> str:
        return (
            f"prompt={self.prompt} model={self.model} step_bonus={self.step_bonus:+d} "
            f"tools={self.tool_variant} dataset={self.dataset_variant}"
        )


#: The configuration behind lesson 7's committed `baseline` run. Everything is
#: measured against this, and `Experiment.validate` measures distance from it.
BASELINE = Config()
BASELINE_RUN = "baseline"


# ---------------------------------------------------------------------------
class Status(str, Enum):
    PROPOSED = "proposed"          # written down, not yet run
    RUN = "run"                    # measured; see the changelog for the outcome
    REJECTED_UNRUN = "rejected"    # dropped before spending tokens, with a reason


@dataclass(frozen=True)
class Experiment:
    id: str
    #: What you believe and *why you believe it*. The "why" is what makes a null
    #: result informative -- without it, "it did not work" teaches nothing.
    hypothesis: str
    #: The single variable, named in prose. Checked against `config.diff(BASELINE)`.
    variable: str
    config: Config
    #: Filled in BEFORE running. Never edited afterwards. This is the bet.
    predicts_fixed: list[str] = field(default_factory=list)
    predicts_broken: list[str] = field(default_factory=list)
    #: Set only for controls. Predicting nothing must be a deliberate statement, not
    #: something you get by leaving two lists empty and moving on.
    predicts_no_change: bool = False
    #: Declares up front that this experiment knowingly breaks the one-variable rule.
    #: `validate()` still rejects it -- the flag documents intent, it does not grant
    #: permission, and `--force` is still required. The distinction matters: a rule you
    #: can opt out of by setting a field is not a rule.
    requires_force: bool = False
    status: Status = Status.PROPOSED
    rejected_because: str | None = None
    #: Roughly what a full 16-case run costs cold, so the cost is visible at the
    #: point of deciding whether to run it.
    baseline_run: str = BASELINE_RUN

    @property
    def run_name(self) -> str:
        return f"exp_{self.id}"

    @property
    def free(self) -> bool:
        """True when this experiment changes scoring only, so the cache covers it."""
        return self.config.diff(BASELINE) == ["dataset_variant"]

    def validate(self) -> list[str]:
        """Complaints that should stop the run. Empty list means go."""
        problems: list[str] = []
        changed = self.config.diff(BASELINE)
        if not changed:
            problems.append(
                f"{self.id} changes nothing relative to the baseline, so it can only "
                f"measure noise. If that is what you want, use --noise."
            )
        elif len(changed) > 1:
            problems.append(
                f"{self.id} changes {len(changed)} variables at once ({', '.join(changed)}). "
                f"A result would not tell you which one moved the score, and you cannot "
                f"ship half a change. Split it into {len(changed)} experiments."
            )
        if not (self.predicts_fixed or self.predicts_broken or self.predicts_no_change):
            problems.append(
                f"{self.id} predicts nothing. An experiment without a prediction cannot "
                f"be wrong, which is the same as not being an experiment. Name the cases "
                f"you expect to move, or set predicts_no_change=True if it is a control."
            )
        if self.predicts_no_change and (self.predicts_fixed or self.predicts_broken):
            problems.append(
                f"{self.id} sets predicts_no_change but also names cases. Pick one."
            )
        overlap = set(self.predicts_fixed) & set(self.predicts_broken)
        if overlap:
            problems.append(f"{self.id} predicts {sorted(overlap)} both fixed and broken.")
        return problems


# ---------------------------------------------------------------------------
# The registry. Read top to bottom as a lab notebook.
# ---------------------------------------------------------------------------
EXPERIMENTS: list[Experiment] = [
    # -- the target: currency_unsupported, the one open failure from lesson 7 -----
    # The agent refuses "convert 100 USD to Bitcoin" *without calling the tool*. The
    # refusal is correct; the process is not, because the same reasoning would refuse
    # a currency the tool does handle. Lesson 7 left it failing on purpose.
    Experiment(
        id="verify_first",
        hypothesis=(
            "The agent refuses without checking because nothing tells it that a "
            "tool's error is better evidence than its own expectation. Say so "
            "explicitly and it will attempt the call, read the rejection, and refuse "
            "with grounds."
        ),
        variable="system prompt: adds a general 'attempt the tool before refusing' rule",
        config=replace(BASELINE, prompt="verify_first"),
        predicts_fixed=["currency_unsupported"],
        # Predicted, not discovered. A blanket instruction to reach for a tool before
        # refusing is also an instruction to reach for a tool, and `no_tool_definition`
        # exists precisely to catch over-eagerness.
        #
        # Measured: right about the win, right that something would break, wrong about
        # what. `no_tool_definition` held. `files_quote_definition` broke instead, with
        # stop_reason=phantom_tool -- the model asked for a tool that was never offered,
        # which is lesson 2's finding re-triggered by a prompt change three lessons
        # later. And it flakes: 8/10 passing under this prompt, 4/4 under the control.
        predicts_broken=["no_tool_definition"],
        status=Status.RUN,
    ),
    Experiment(
        id="verify_conversion",
        hypothesis=(
            "Same fix as verify_first, scoped to conversions and lookups. If the "
            "broad version works but costs an unnecessary tool call elsewhere, the "
            "narrow version should keep the win without the collateral damage."
        ),
        variable="system prompt: the same rule, restricted to conversion/lookup tools",
        config=replace(BASELINE, prompt="verify_conversion"),
        predicts_fixed=["currency_unsupported"],
        predicts_broken=[],
        # Measured: no metric change, and the suite was wrong to say so. The agent still
        # did not call the tool, but its refusal moved from "I don't have a way to look
        # up the current price of Bitcoin" to "the conversion tool only supports USD,
        # EUR, GBP, INR, JPY, AUD, CAD". That is a real improvement in grounding and no
        # scorer in the dataset could see it. Also 830 tokens cheaper than the baseline.
        status=Status.RUN,
    ),
    # -- a diagnostic, not a proposal ---------------------------------------------
    Experiment(
        id="hide_currency_list",
        hypothesis=(
            "The tool description already lists the seven supported currencies, so "
            "the agent can see BTC is absent and skips the call. Remove the list and "
            "it has to ask. Run this to test the DIAGNOSIS, not as a fix: a tool "
            "whose description hides what it accepts is a worse tool, and shipping it "
            "to make an eval pass is optimising the metric."
        ),
        variable="tool description: drops the supported-currency list from convert_currency",
        config=replace(BASELINE, tool_variant="hide_currency_list"),
        predicts_fixed=["currency_unsupported"],
        predicts_broken=[],
        status=Status.REJECTED_UNRUN,
        rejected_because=(
            "Refuted by a single-case probe for about 5,000 tokens instead of 33,000 "
            "for a suite run: `--recheck currency_unsupported --experiment "
            "hide_currency_list --repeats 3` gave 0/3, tools=(none) every time. With the "
            "supported-currency list removed entirely the agent STILL refuses without "
            "calling, so the description was never the cause -- it is declining on a "
            "prior belief about Bitcoin, not on anything it read. The diagnosis everyone "
            "reaches for first is simply wrong, and a three-repeat probe of one case "
            "killed it for a sixth of the price."
        ),
    ),
    Experiment(
        id="loud_currency_list",
        hypothesis=(
            "The inverse of hide_currency_list, and the version you could actually "
            "ship: keep the list and add a sentence telling the agent to check rather "
            "than assume. If description wording is the lever, this should move the "
            "case without making the tool less honest."
        ),
        variable="tool description: keeps the list, adds 'call this tool to check'",
        config=replace(BASELINE, tool_variant="loud_currency_list"),
        predicts_fixed=["currency_unsupported"],
        predicts_broken=[],
        status=Status.REJECTED_UNRUN,
        rejected_because=(
            "Also 0/3 on the same probe. Adding 'call this tool to check rather than "
            "assume' to the tool description changed nothing, while the *same sentence "
            "in the system prompt* (verify_first) did make the agent call the tool. So "
            "for this model, instruction in the system prompt carries and instruction in "
            "a tool description does not -- a distinction worth knowing before spending "
            "an afternoon rewording tool descriptions."
        ),
    ),
    # -- change the eval, not the agent. Costs nothing. ---------------------------
    Experiment(
        id="documented_refusal",
        hypothesis=(
            "The case may be wrong rather than the agent. convert_currency documents "
            "its seven currencies, so refusing without calling can be a well-grounded "
            "decision. Accept a refusal that cites those documented limits, and "
            "require a tool call otherwise."
        ),
        variable="dataset: swaps used_tools for cited_tool_limits on one case",
        config=replace(BASELINE, dataset_variant="documented_refusal"),
        predicts_fixed=["currency_unsupported"],
        predicts_broken=[],
        # Measured: no change. The prediction assumed the refusal was grounded in the
        # tool's documented currency list; reading the recorded answer showed it was
        # grounded in "I don't have a way to look up the current price of Bitcoin",
        # which is a different and weaker claim, and `cited_tool_limits` correctly
        # refuses to accept it.
        #
        # That refutation was available for free BEFORE the run, by reading one answer
        # already sitting in baseline.json. Reading the failure is step zero of
        # iteration, and it is the step skipped in favour of theorising.
        status=Status.RUN,
    ),
    # -- deliberately two variables, and it has to be forced ----------------------
    # Added after verify_conversion and documented_refusal each measured as "no
    # change". Reading the answers showed why: the prompt did move the refusal from
    # "I cannot look up the Bitcoin price" to "the tool only supports USD, EUR, GBP,
    # INR, JPY, AUD, CAD", and the new scorer was built to accept exactly that -- but
    # neither half is visible alone, because the scorer that moved was in the other
    # experiment.
    #
    # `validate()` rejects this, correctly, and `--force` records in the changelog
    # that the rule was overridden. It earns its place because it costs nothing (both
    # halves are already cached) and because a confirmed diagnosis is worth more than
    # a clean rule book. What it cannot tell you is which half did the work, which is
    # precisely the cost of running it.
    Experiment(
        id="combined_verify_and_rescore",
        hypothesis=(
            "verify_conversion and documented_refusal are each individually invisible "
            "to the suite and jointly sufficient: the prompt makes the refusal cite the "
            "documented currency list, and the rescored case accepts a refusal on those "
            "grounds. Run to confirm the diagnosis, not to ship."
        ),
        variable="TWO: the verify_conversion prompt AND the documented_refusal scorer",
        config=replace(BASELINE, prompt="verify_conversion", dataset_variant="documented_refusal"),
        predicts_fixed=["currency_unsupported"],
        predicts_broken=[],
        requires_force=True,
        # Measured: 16/16, the only 100% in the project, for zero tokens -- both halves
        # were already cached. Reproducible 4/4 on a recheck. Cost per success actually
        # fell to 0.89x, because the prompt is shorter and one more case succeeds.
        status=Status.RUN,
    ),
    # -- cost, not accuracy -------------------------------------------------------
    Experiment(
        id="terse_prompt",
        hypothesis=(
            "Lesson 7's `terse` prompt was defined and never run. Most of the default "
            "prompt is instruction the model arguably does not need. If accuracy holds, "
            "the shorter prompt is strictly better, because the system prompt is re-sent "
            "on every step of every case."
        ),
        variable="system prompt: lesson 7's unused one-line `terse` variant",
        config=replace(BASELINE, prompt="terse"),
        predicts_fixed=[],
        # An honest "I expect no accuracy gain" still has to name what is at risk, or
        # `validate()` rejects it. These two lean hardest on being told not to guess.
        # Note what cannot go here: `currency_unsupported` is already failing, so it
        # cannot break. Predicting a break for an already-failing case is a category
        # error, and `check_predictions` flags it.
        predicts_broken=["impossible_share_price", "files_refuse_escape"],
        # Not run: the day's 200,000 tokens went on the experiments above, and the four
        # rechecks that made them interpretable. Left as the honest state of things
        # rather than deleted -- "we ran out of budget before testing this" is a real
        # and common reason, and a registry that hides it pretends the work was
        # exhaustive.
    ),
    # -- a deliberate null experiment ---------------------------------------------
    Experiment(
        id="more_steps",
        hypothesis=(
            "Nothing is step-limited: no case in the baseline stopped with "
            "stop_reason max_steps. So +2 steps should change nothing at all, and is "
            "here as a control -- if it DOES move the score, the movement is noise and "
            "the whole suite's resolving power is worse than advertised."
        ),
        variable="step cap: +2 on every case",
        config=replace(BASELINE, step_bonus=2),
        predicts_no_change=True,
        # Not run, and worth saying why rather than leaving it looking forgotten. A
        # control that costs a full 33,000-token suite run is a luxury; the budget went
        # on rechecks instead, which answered a sharper question about the same thing.
        # If you have the tokens, run this before trusting any single-case result.
    ),
]


def by_id(experiment_id: str) -> Experiment:
    for experiment in EXPERIMENTS:
        if experiment.id == experiment_id:
            return experiment
    known = ", ".join(e.id for e in EXPERIMENTS)
    raise KeyError(f"No experiment {experiment_id!r}. Known: {known}")
