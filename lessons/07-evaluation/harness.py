"""Running an eval suite, caching the results, and comparing two runs.

Three design decisions worth understanding before the code.

**Caching is not an optimisation, it is a correctness feature.** A 12-case suite
costs roughly 40,000 tokens against a 200,000/day allowance, so an uncached harness
is one you run three times and then stop running. Worse, without caching you cannot
compare two configurations fairly: re-running the baseline gets you *different*
baseline numbers, and you end up attributing model variance to your change. Cached
runs are reproducible runs.

**A run is a durable artifact, not a printout.** Results are saved as JSON with the
configuration that produced them. That is what makes "did this change help?"
answerable a week later, and it is the seed of the tracing work in lesson 8.

**Comparison reports what broke, not just the average.** An aggregate score can
rise while specific cases regress, and the regressions are usually what you care
about. A harness that only prints "68% -> 74%" actively hides that.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset import CASES, Category, EvalCase
from scorers import ScoreResult, score_case

RUNS_DIR = Path(__file__).parent / "runs"


# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    answer: str | None
    tool_sequence: list[str]
    stop_reason: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    scores: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def failed_scorers(self) -> list[str]:
        return [s["name"] for s in self.scores if not s["passed"]]


@dataclass
class EvalRun:
    """One complete pass over the dataset, plus the configuration behind it."""

    name: str
    model: str
    provider: str
    max_steps: int
    created_at: str
    results: list[CaseResult] = field(default_factory=list)
    #: Anything else that could change the outcome. Recorded so a comparison can
    #: warn when two runs differ in more than the one thing you meant to change.
    config: dict[str, Any] = field(default_factory=dict)

    # -- aggregate metrics ----------------------------------------------
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def tool_choice_accuracy(self) -> float:
        """Fraction of cases where every expected tool was actually used.

        Reported separately from success because they diverge in informative ways.
        High success with low tool accuracy means the agent is right by luck.
        """
        scored = [
            r
            for r in self.results
            if any(s["name"].startswith("used_tools") or s["name"] == "answered_without_tools"
                   for s in r.scores)
        ]
        if not scored:
            return 0.0
        correct = sum(
            1
            for r in scored
            if all(
                s["passed"]
                for s in r.scores
                if s["name"].startswith("used_tools") or s["name"] == "answered_without_tools"
            )
        )
        return correct / len(scored)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.results)

    @property
    def mean_steps(self) -> float:
        return sum(r.steps for r in self.results) / self.total if self.total else 0.0

    @property
    def errors(self) -> list[CaseResult]:
        return [r for r in self.results if r.error]

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for result in self.results:
            done, total = out.get(result.category, (0, 0))
            out[result.category] = (done + int(result.passed), total + 1)
        return out

    def result_for(self, case_id: str) -> CaseResult | None:
        return next((r for r in self.results if r.case_id == case_id), None)

    # -- persistence ----------------------------------------------------
    @property
    def path(self) -> Path:
        return RUNS_DIR / f"{self.name}.json"

    def save(self) -> Path:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model": self.model,
            "provider": self.provider,
            "max_steps": self.max_steps,
            "created_at": self.created_at,
            "config": self.config,
            "results": [asdict(r) for r in self.results],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, name: str) -> EvalRun:
        path = RUNS_DIR / f"{name}.json"
        if not path.exists():
            available = sorted(p.stem for p in RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []
            raise FileNotFoundError(
                f"No run named {name!r}.\n"
                f"Available: {available or '(none yet)'}\n"
                f"Create one with: uv run lessons/07-evaluation/evaluate.py --run {name}"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            model=raw["model"],
            provider=raw["provider"],
            max_steps=raw.get("max_steps", 6),
            created_at=raw.get("created_at", ""),
            config=raw.get("config", {}),
            results=[CaseResult(**r) for r in raw.get("results", [])],
        )

    @classmethod
    def list_runs(cls) -> list[str]:
        if not RUNS_DIR.exists():
            return []
        return sorted(p.stem for p in RUNS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Per-case cache
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class Execution:
    """What the agent DID, separated from how we judge it.

    This split is the most important design decision in the file, and it was got
    wrong first. The original cache stored the fully *scored* `CaseResult`, so
    adding a scorer to an existing case had no effect: the cached verdict was
    replayed and the new check never ran. The suite reported a pass for a case that
    should have failed, which is the worst possible failure in a measurement tool --
    it was confidently wrong and silent about it.

    Cache the expensive, stable thing (the agent run) and recompute the cheap,
    volatile thing (the score) every time. Scoring is microseconds; an agent run is
    thousands of tokens. So changing a scorer re-scores the whole suite for free,
    and only a changed *question* costs anything.

    It also satisfies the `Trajectory` protocol the scorers expect, so a replayed
    execution is scored by exactly the same code as a live one.
    """

    final_answer: str | None
    tool_sequence: list[str]
    stop_reason: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == "completed" and bool(self.final_answer)


def cache_key(case: EvalCase, model: str, max_steps: int, system_prompt: str | None) -> str:
    """Identity of an agent *execution* -- deliberately not of its score.

    Includes everything that changes what the agent does: the question, the model,
    the step cap, the system prompt. Notably absent: the scorers, because they do
    not affect the run and must not invalidate it.

    A cache keyed on the case id alone would be worse than no cache, serving results
    from a different configuration while you believed you compared one thing.
    """
    digest = hashlib.sha256()
    for part in (case.id, case.question, model, str(max_steps), system_prompt or ""):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:20]


def _cached_execution(key: str) -> Execution | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return Execution(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        # A cache that fails to parse is simply a miss. Never let a stale schema
        # break the suite.
        return None


def _store_execution(key: str, execution: Execution) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(asdict(execution), indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_case(
    client,
    registry,
    case: EvalCase,
    *,
    system_prompt: str | None = None,
    use_cache: bool = True,
    on_progress=None,
) -> tuple[CaseResult, bool]:
    """Execute one case and score it. Returns (result, came_from_cache)."""
    from loop import run_agent

    key = cache_key(case, client.config.model, case.max_steps, system_prompt)
    from_cache = False
    execution: Execution | None = None

    if use_cache:
        execution = _cached_execution(key)
        from_cache = execution is not None

    if execution is None:
        started = time.perf_counter()
        try:
            trajectory = run_agent(
                client,
                case.question,
                registry,
                max_steps=case.max_steps,
                system_prompt=system_prompt,
            )
            execution = Execution(
                final_answer=trajectory.final_answer,
                tool_sequence=trajectory.tool_sequence,
                stop_reason=trajectory.stop_reason.value,
                steps=len(trajectory.steps),
                prompt_tokens=trajectory.usage.prompt_tokens,
                completion_tokens=trajectory.usage.completion_tokens,
                latency_s=trajectory.usage.latency_s,
            )
            if use_cache:
                _store_execution(key, execution)
        except Exception as exc:  # noqa: BLE001
            # An error is a result, not a crash. A suite that dies on case 3 of 16
            # tells you nothing about cases 4 to 16, and rate limits make this
            # common. Deliberately NOT cached -- a rate limit is not a finding about
            # the agent, and caching it would poison every later run.
            execution = Execution(
                final_answer=None,
                tool_sequence=[],
                stop_reason="error",
                steps=0,
                prompt_tokens=0,
                completion_tokens=0,
                latency_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )

    # Scoring always runs fresh, cached execution or not. See Execution's docstring.
    if execution.error:
        passed, scores = False, []
    else:
        passed, scores = score_case(case, execution)

    result = CaseResult(
        case_id=case.id,
        category=case.category.value,
        passed=passed,
        answer=execution.final_answer,
        tool_sequence=execution.tool_sequence,
        stop_reason=execution.stop_reason,
        steps=execution.steps,
        prompt_tokens=execution.prompt_tokens,
        completion_tokens=execution.completion_tokens,
        latency_s=execution.latency_s,
        scores=[_score_to_dict(s) for s in scores],
        error=execution.error,
    )
    if on_progress:
        on_progress(case, result, from_cache)
    return result, from_cache


def _score_to_dict(score: ScoreResult) -> dict[str, Any]:
    return {"name": score.name, "passed": score.passed, "detail": score.detail}


def run_eval(
    client,
    registry,
    name: str,
    *,
    cases: list[EvalCase] | None = None,
    system_prompt: str | None = None,
    use_cache: bool = True,
    pause_between: float = 0.0,
    on_progress=None,
) -> tuple[EvalRun, int]:
    """Run the suite. Returns (run, number_served_from_cache)."""
    selected = cases if cases is not None else CASES
    run = EvalRun(
        name=name,
        model=client.config.model,
        provider=client.config.provider,
        max_steps=max((c.max_steps for c in selected), default=6),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config={
            "system_prompt": system_prompt or "(lesson 3 default)",
            "cases": len(selected),
            "dataset_version": _dataset_fingerprint(selected),
        },
    )

    cache_hits = 0
    for index, case in enumerate(selected):
        result, from_cache = run_case(
            client,
            registry,
            case,
            system_prompt=system_prompt,
            use_cache=use_cache,
            on_progress=on_progress,
        )
        run.results.append(result)
        cache_hits += int(from_cache)
        # Pause only when we actually called the API, and not after the last case.
        if not from_cache and pause_between and index < len(selected) - 1:
            time.sleep(pause_between)

    return run, cache_hits


def _dataset_fingerprint(cases: list[EvalCase]) -> str:
    """So a comparison can tell you the dataset changed underneath it.

    Comparing two runs scored against different datasets is a silent way to reach a
    wrong conclusion, and it happens as soon as you add a case mid-session.
    """
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda c: c.id):
        digest.update(case.id.encode())
        digest.update(case.question.encode())
        digest.update(str(len(case.scorers)).encode())
    return digest.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
@dataclass
class Comparison:
    baseline: EvalRun
    candidate: EvalRun
    fixed: list[str] = field(default_factory=list)      # was failing, now passing
    broken: list[str] = field(default_factory=list)     # was passing, now failing
    still_failing: list[str] = field(default_factory=list)
    unchanged_pass: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def net(self) -> int:
        return len(self.fixed) - len(self.broken)

    @property
    def verdict(self) -> str:
        """Deliberately cautious language.

        With a dozen cases, a net gain of one is noise. Saying "better" on that
        basis is how a project convinces itself of an improvement it never made.
        """
        if self.broken and not self.fixed:
            return "REGRESSION"
        if self.fixed and not self.broken:
            return "improvement" if self.net > 1 else "improvement (within noise)"
        if not self.fixed and not self.broken:
            return "no change"
        return "mixed"


def compare(baseline: EvalRun, candidate: EvalRun) -> Comparison:
    """Diff two runs case by case.

    The per-case diff is the point. An average tells you the direction; only the
    case list tells you whether you traded three wins for two losses in areas you
    care about differently.
    """
    result = Comparison(baseline=baseline, candidate=candidate)

    base_fp = baseline.config.get("dataset_version")
    cand_fp = candidate.config.get("dataset_version")
    if base_fp and cand_fp and base_fp != cand_fp:
        result.warnings.append(
            "The dataset changed between these runs, so the comparison is not "
            "apples to apples. Re-run the baseline."
        )
    if baseline.model != candidate.model:
        result.warnings.append(
            f"Different models ({baseline.model} vs {candidate.model}). Fine if that "
            f"is what you are testing; misleading if you meant to vary the prompt."
        )
    if baseline.config.get("system_prompt") != candidate.config.get("system_prompt"):
        result.warnings.append("System prompts differ between these runs.")

    for case_result in candidate.results:
        before = baseline.result_for(case_result.case_id)
        if before is None:
            result.warnings.append(f"{case_result.case_id} is new; not in the baseline.")
            continue
        if before.passed and not case_result.passed:
            result.broken.append(case_result.case_id)
        elif not before.passed and case_result.passed:
            result.fixed.append(case_result.case_id)
        elif case_result.passed:
            result.unchanged_pass.append(case_result.case_id)
        else:
            result.still_failing.append(case_result.case_id)

    return result


# ---------------------------------------------------------------------------
def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a success rate.

    Included because a bare "9/12 = 75%" invites false confidence. The Wilson
    interval for 9/12 runs roughly 47% to 91%, which is the honest summary: this
    suite can tell a good agent from a broken one and cannot tell 70% from 80%.

    Printing the interval is the cheapest available defence against over-reading
    your own numbers.
    """
    if total == 0:
        return (0.0, 0.0)
    phat = passed / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    margin = (
        z * ((phat * (1 - phat) / total + z**2 / (4 * total**2)) ** 0.5)
    ) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def clear_cache() -> int:
    """Remove cached case results. Returns how many were deleted."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for path in files:
        path.unlink()
    return len(files)


def category_of(name: str) -> Category | None:
    try:
        return Category(name)
    except ValueError:
        return None
