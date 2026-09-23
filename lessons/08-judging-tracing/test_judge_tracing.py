"""Offline tests for the judge and the tracing layer.

Notice what makes these possible: lesson 6's `ScriptedClient`. A judge is a model
call, so testing it would normally mean spending tokens and accepting
non-determinism. With a scripted double you can hand the judge malformed JSON,
an empty reply, or a verdict with a missing field, and assert it behaves.

That is the payoff for building the doubles two lessons ago, and it is the reason a
judge -- the component most likely to silently corrupt your metrics -- can be tested
at all.
"""

from __future__ import annotations

import pytest
from fakes import ScriptedClient, text_response, tool_response
from llmkit import ToolRegistry, ToolSpec

from judge import (
    JUDGE_SYSTEM_PROMPT,
    NAIVE_JUDGE_SYSTEM_PROMPT,
    Verdict,
    build_prompt,
    judge_answer,
)
from tracing import (
    FALLBACK_PRICE,
    CostReport,
    build_trace,
    cost_usd,
    price_for,
)


# ---------------------------------------------------------------------------
class TestVerdictParsing:
    """The judge must survive every way a model mangles JSON."""

    def test_parses_a_clean_verdict(self) -> None:
        client = ScriptedClient(
            [text_response('{"reasoning": "correct product", "passed": true, "confidence": "high"}')]
        )
        result = judge_answer(client, "q", "6319", ["correct"], use_cache=False)
        assert result.passed
        assert result.verdict.confidence == "high"
        assert result.attempts == 1

    def test_parses_a_fenced_verdict(self) -> None:
        client = ScriptedClient(
            [text_response('```json\n{"reasoning": "ok", "passed": true, "confidence": "low"}\n```')]
        )
        assert judge_answer(client, "q", "a", ["c"], use_cache=False).passed

    def test_repairs_invalid_json(self) -> None:
        """Lesson 1's repair loop, doing its job for the judge."""
        client = ScriptedClient(
            [
                text_response("Here you go: not json at all"),
                text_response('{"reasoning": "fine", "passed": false, "confidence": "medium"}'),
            ]
        )
        result = judge_answer(client, "q", "a", ["c"], use_cache=False)
        assert result.verdict is not None
        assert result.attempts == 2
        assert not result.passed

    def test_repairs_a_missing_field(self) -> None:
        client = ScriptedClient(
            [
                text_response('{"reasoning": "no verdict key"}'),
                text_response('{"reasoning": "ok", "passed": true, "confidence": "high"}'),
            ]
        )
        assert judge_answer(client, "q", "a", ["c"], use_cache=False).passed

    def test_rejects_an_invalid_confidence_value(self) -> None:
        """The Literal type is doing real work: 'very high' is not a valid level."""
        client = ScriptedClient(
            [
                text_response('{"reasoning": "x", "passed": true, "confidence": "very high"}'),
                text_response('{"reasoning": "x", "passed": true, "confidence": "high"}'),
            ]
        )
        result = judge_answer(client, "q", "a", ["c"], use_cache=False)
        assert result.attempts == 2
        assert result.verdict.confidence == "high"

    def test_gives_up_after_max_attempts(self) -> None:
        client = ScriptedClient([text_response("nope")] * 3)
        result = judge_answer(client, "q", "a", ["c"], max_attempts=3, use_cache=False)
        assert result.verdict is None
        assert result.error
        assert not result.passed

    def test_a_failed_judge_never_passes(self) -> None:
        """The critical safety property.

        Defaulting an unparseable verdict to True would silently inflate every score
        in the suite. Failing closed at least shows up as a visible problem.
        """
        client = ScriptedClient([text_response("")] * 3)
        assert judge_answer(client, "q", "a", ["c"], use_cache=False).passed is False

    def test_empty_answer_is_still_judged(self) -> None:
        """An agent producing no answer must be judgeable, not a crash."""
        client = ScriptedClient(
            [text_response('{"reasoning": "no answer", "passed": false, "confidence": "high"}')]
        )
        result = judge_answer(client, "q", None, ["c"], use_cache=False)
        assert not result.passed


class TestJudgePrompt:
    def test_the_prompt_never_contains_the_expected_answer(self) -> None:
        """The judge assesses the work, not a key.

        Given the answer, a model agrees with the answer -- a subtle way to measure
        nothing while looking rigorous. `build_prompt` takes no expected value at
        all, so the mistake is unrepresentable rather than merely discouraged.
        """
        prompt = build_prompt("What is 71 times 89?", "6319", ["gives the correct product"])
        assert "6319" in prompt          # the agent's answer, which it must see
        assert "expected" not in prompt.lower()

        import inspect

        assert "expected" not in inspect.signature(build_prompt).parameters

    def test_mitigated_rubric_forbids_rewarding_style(self) -> None:
        """The three sentences that measurably changed pairwise preference."""
        for phrase in ("length", "confidence", "fluency"):
            assert phrase in JUDGE_SYSTEM_PROMPT

    def test_naive_rubric_omits_those_instructions(self) -> None:
        """The control condition. If it gained them, the bias experiment is void."""
        assert "fluency" not in NAIVE_JUDGE_SYSTEM_PROMPT
        assert "length" not in NAIVE_JUDGE_SYSTEM_PROMPT

    def test_rubric_is_part_of_the_cache_key(self) -> None:
        """Two rubrics are two instruments and must not share cached verdicts.

        Sharing them would make the bias A/B return identical results and look like
        'no bias found', which is exactly the wrong conclusion.
        """
        from judge import _cache_key

        a = _cache_key("m", "q", "ans", ["c", JUDGE_SYSTEM_PROMPT])
        b = _cache_key("m", "q", "ans", ["c", NAIVE_JUDGE_SYSTEM_PROMPT])
        assert a != b


# ---------------------------------------------------------------------------
class TestPricing:
    def test_known_model_uses_its_price(self) -> None:
        (price_in, price_out), known = price_for("openai/gpt-oss-20b")
        assert known
        assert price_in < price_out, "output tokens should cost more than input"

    def test_unknown_model_falls_back_and_says_so(self) -> None:
        """An unknown model must not silently read as free.

        Returning 0.0 would make a cost report say '$0.0000' for a model you are
        actually paying for -- a wrong number that looks like a good one.
        """
        price, known = price_for("some-model-that-does-not-exist")
        assert not known
        assert price == FALLBACK_PRICE
        assert price[0] > 0

    def test_local_models_are_free(self) -> None:
        (price_in, price_out), known = price_for("qwen2.5:7b-instruct")
        assert known
        assert price_in == 0.0 and price_out == 0.0

    def test_cost_scales_with_tokens(self) -> None:
        single = cost_usd("openai/gpt-oss-20b", 1000, 100)
        double = cost_usd("openai/gpt-oss-20b", 2000, 200)
        assert double == pytest.approx(single * 2)


class TestCostReport:
    def _report(self, runs: int, successes: int) -> CostReport:
        return CostReport(
            model="openai/gpt-oss-20b",
            price_known=True,
            runs=runs,
            successes=successes,
            prompt_tokens=30_000,
            completion_tokens=3_000,
            reasoning_tokens=1_500,
        )

    def test_cost_per_success_exceeds_cost_per_run_when_some_fail(self) -> None:
        """The whole point of the metric."""
        report = self._report(16, 8)
        assert report.cost_per_success > report.cost_per_run
        assert report.cost_per_success == pytest.approx(report.cost_per_run * 2)

    def test_cost_per_success_is_infinite_when_nothing_works(self) -> None:
        """Honest, and it makes the failure impossible to overlook in a table."""
        assert self._report(16, 0).cost_per_success == float("inf")

    def test_wasted_cost_accounts_for_failures(self) -> None:
        report = self._report(16, 12)
        assert report.wasted_cost == pytest.approx(report.cost_per_run * 4)

    def test_input_share_reflects_the_quadratic_growth(self) -> None:
        """Agents are input-heavy because every step re-sends the conversation."""
        assert self._report(16, 16).input_share > 0.85


# ---------------------------------------------------------------------------
class TestTracing:
    @pytest.fixture
    def tiny_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.add(
            ToolSpec(
                name="echo",
                description="Echo a value.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            ),
            lambda value: f"echo: {value}",
        )
        return registry

    def _trajectory(self, tiny_registry):
        from loop import run_agent

        client = ScriptedClient(
            [
                tool_response("echo", {"value": "a"}, call_id="c1"),
                tool_response("echo", {"value": "b"}, call_id="c2"),
                text_response("all done"),
            ]
        )
        return run_agent(client, "a question", tiny_registry, max_steps=5)

    def test_trace_has_one_span_per_step(self, tiny_registry) -> None:
        trace = build_trace(self._trajectory(tiny_registry), "openai/gpt-oss-20b")
        assert len(trace.spans) == 3
        assert [s.name for s in trace.spans] == ["step 1", "step 2", "step 3"]

    def test_tool_calls_nest_under_their_step(self, tiny_registry) -> None:
        """The tree structure is the point -- a flat log loses this."""
        trace = build_trace(self._trajectory(tiny_registry), "openai/gpt-oss-20b")
        first = trace.spans[0]
        kinds = [c.kind for c in first.children]
        assert kinds == ["model_call", "tool_call"]

    def test_final_step_has_no_tool_calls(self, tiny_registry) -> None:
        trace = build_trace(self._trajectory(tiny_registry), "openai/gpt-oss-20b")
        assert all(c.kind == "model_call" for c in trace.spans[-1].children)

    def test_token_rollup_does_not_double_count(self, tiny_registry) -> None:
        """The bug this test was written to catch.

        A step span mirrors its model_call child's token counts so the tree can show
        a per-step total. Summing over every span therefore counted each step twice
        and reported exactly double -- with nothing crashing and every cost figure
        silently 2x. Rollups must count model_call spans only.
        """
        trace = build_trace(self._trajectory(tiny_registry), "openai/gpt-oss-20b")
        expected = sum(s.prompt_tokens for s in trace.spans)
        assert trace.prompt_tokens == expected, "rollup is double-counting"
        assert trace.total_tokens == trace.prompt_tokens + trace.completion_tokens

    def test_tool_spans_contribute_no_tokens(self, tiny_registry) -> None:
        """Only model calls consume tokens. A tool execution costs time, not tokens."""
        trace = build_trace(self._trajectory(tiny_registry), "openai/gpt-oss-20b")
        tool_spans = [c for s in trace.spans for c in s.children if c.kind == "tool_call"]
        assert tool_spans
        assert all(s.total_tokens == 0 for s in tool_spans)

    def test_failed_tool_marks_its_span_and_step(self, tiny_registry) -> None:
        """A failure must be visible in the tree without reading every line."""
        from loop import run_agent

        client = ScriptedClient(
            [
                tool_response("does_not_exist", {}, call_id="c1"),
                text_response("recovered"),
            ]
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=4)
        trace = build_trace(trajectory, "openai/gpt-oss-20b")

        assert trace.failed_spans
        assert not trace.spans[0].ok

    def test_trace_records_the_stop_reason(self, tiny_registry) -> None:
        trace = build_trace(self._trajectory(tiny_registry), "openai/gpt-oss-20b")
        assert trace.stop_reason == "completed"

    def test_trace_round_trips_to_disk(self, tiny_registry, tmp_path, monkeypatch) -> None:
        import json

        import tracing

        monkeypatch.setattr(tracing, "TRACES_DIR", tmp_path)
        trace = build_trace(self._trajectory(tiny_registry), "openai/gpt-oss-20b")
        path = trace.save("unit_test")

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["stop_reason"] == "completed"
        assert len(raw["spans"]) == 3
        assert raw["spans"][0]["children"][0]["kind"] == "model_call"


# ---------------------------------------------------------------------------
# Commentary derived from the numbers, not asserted over them
# ---------------------------------------------------------------------------
def _cost_report(successes: int, runs: int, prompt: int, completion: int):
    from tracing import CostReport

    return CostReport(
        model="openai/gpt-oss-20b",
        price_known=True,
        runs=runs,
        successes=successes,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=0,
    )


def test_cost_verdict_quantifies_how_much_of_a_saving_a_regression_eats():
    """Lesson 7's real strict-prompt figures: cost per case -10.2%, cost per success
    only -3.8%. Note what this refutes -- the original hardcoded prose claimed cost per
    success 'went the wrong way', and it had not. It improved, just far less than the
    per-case figure suggested. The honest lesson is that the accuracy regression ate
    about two-thirds of the apparent saving, not that value got worse."""
    from tracing import cost_verdict

    before = _cost_report(15, 16, 30_313, 2_996)   # baseline: $0.000295/success
    after = _cost_report(14, 16, 27_574, 2_728)    # strict:   $0.000284/success
    verdict = cost_verdict(before, after)

    assert "cost per successful answer improved" in verdict
    assert "mostly illusory" in verdict
    assert "was, on accuracy, a regression" in verdict


def test_cost_verdict_still_names_the_outright_trap():
    """A large enough accuracy loss does flip cost per success the wrong way, and that
    case must read differently from the one above."""
    from tracing import cost_verdict

    before = _cost_report(15, 16, 30_000, 3_000)
    after = _cost_report(10, 16, 27_000, 2_700)
    verdict = cost_verdict(before, after)

    assert "cost per successful answer got worse" in verdict
    assert "outright trap" in verdict


def test_cost_verdict_does_not_claim_a_regression_when_things_improved():
    """The bug this replaced. The panel was hardcoded prose about baseline-vs-strict, so
    it announced that cost per success 'went the wrong way' while the table above it
    showed an 11% improvement. Commentary that ignores its own data is a confident
    caption on the wrong photograph."""
    from tracing import cost_verdict

    before = _cost_report(15, 16, 30_000, 3_000)
    after = _cost_report(16, 16, 27_000, 2_700)
    verdict = cost_verdict(before, after)

    assert "cost per successful answer improved" in verdict
    assert "got worse" not in verdict
    assert "Cheaper and more accurate" in verdict


def test_cost_verdict_handles_the_normal_shape_of_an_improvement():
    from tracing import cost_verdict

    before = _cost_report(14, 16, 20_000, 2_000)
    after = _cost_report(16, 16, 40_000, 4_000)
    verdict = cost_verdict(before, after)

    assert "More accurate and more expensive" in verdict


def test_cost_verdict_always_reports_the_input_share():
    from tracing import cost_verdict

    before = _cost_report(15, 16, 90_000, 10_000)
    after = _cost_report(15, 16, 90_000, 10_000)
    assert "90% of baseline tokens are input" in cost_verdict(before, after)
