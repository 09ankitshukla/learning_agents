"""Tests that call a real model. Opt-in only.

    uv run pytest lessons/06-testing -m live

Deselected by default, because they are slow, cost tokens, and fail when a provider
rate-limits you. A suite you avoid running because it is expensive gives you no
safety, so the default invocation must stay free and instant.

What belongs here is deliberately narrow. Live tests should check the things a
double *cannot*: that your assumptions about the provider still hold. They are
closer to monitoring than to unit testing, and they should be few.

They are also allowed to be tolerant. Asserting that a model produces specific
prose would make them flaky and teach you to ignore failures. Assert on shape and
on contract instead.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


class TestProviderContract:
    """Assumptions about the provider that would silently invalidate the lessons."""

    def test_configured_model_still_exists(self, live_client) -> None:
        """Hosted providers retire models. This project already lost one mid-course.

        Cheap, no generation, and it fails with an actionable message rather than
        an opaque 404 three lessons in.
        """
        available = live_client.available_models()
        if not available:
            pytest.skip("provider does not list models")
        wanted = live_client.config.model
        assert any(wanted == m or wanted.split(":")[0] in m for m in available), (
            f"{wanted} is no longer served. Available: {available}"
        )

    def test_tool_calling_still_works(self, live_client) -> None:
        """The capability every lesson from 2 onward depends on."""
        from llmkit import ToolSpec, user

        spec = ToolSpec(
            name="get_temperature",
            description="Get the current temperature for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        reply = live_client.chat(
            [user("What is the temperature in Mumbai? Use the tool.")],
            tools=[spec],
            max_tokens=512,
        )

        assert reply.wants_tools, "model returned prose instead of a tool call"
        call = reply.tool_calls[0]
        assert call.name == "get_temperature"
        assert call.is_valid, f"arguments were malformed: {call.malformed_arguments!r}"

    def test_reasoning_budget_assumption_holds(self, live_client) -> None:
        """max_tokens >= 500 should yield visible text on a reasoning model.

        Every lesson's default depends on this. If a provider changes its reasoning
        behaviour, that assumption breaks everywhere at once, and this is where it
        should surface.
        """
        from llmkit import user

        reply = live_client.chat([user("Name three primary colours.")], max_tokens=512)
        assert reply.text, (
            f"empty text at max_tokens=512 "
            f"(finish_reason={reply.finish_reason}, "
            f"reasoning_tokens={reply.usage.reasoning_tokens})"
        )

    def test_usage_accounting_is_reported(self, live_client) -> None:
        """Lesson 4's budgeting and lesson 8's costing both need real numbers."""
        from llmkit import user

        reply = live_client.chat([user("hello")], max_tokens=64)
        assert reply.usage.prompt_tokens > 0
        assert reply.usage.completion_tokens > 0


class TestTokenEstimatorAgainstReality:
    """The estimator was 91% wrong once. This is how that gets caught next time."""

    @pytest.mark.parametrize(
        "text",
        [
            "What is 2 + 2?",
            "Explain in three sentences why agents need tools. " * 6,
            '{"path": "docs/glossary.md", "start_line": 1, "max_lines": 120}' * 8,
        ],
    )
    def test_estimate_is_within_tolerance(self, live_client, text: str) -> None:
        from context import conversation_tokens
        from llmkit import user

        messages = [user(text)]
        estimated = conversation_tokens(messages)
        actual = live_client.chat(messages, max_tokens=16).usage.prompt_tokens

        error = (estimated - actual) / actual
        # Generous band on purpose. The point is to catch a *structural* break --
        # a new chat template, a different tokenizer -- not to chase a few percent.
        assert -0.35 < error < 0.55, (
            f"estimator drifted: estimated {estimated}, actual {actual} ({error:+.0%})"
        )


class TestEndToEnd:
    """One full agent run against a real model. Smoke test, not a unit test."""

    def test_agent_completes_a_multi_step_task(self, live_client, registry) -> None:
        from loop import StopReason, run_agent

        trajectory = run_agent(
            live_client,
            "What time is it in Tokyo, and what is 71 * 89?",
            registry,
            max_steps=6,
        )

        assert trajectory.stop_reason in (StopReason.COMPLETED, StopReason.MAX_STEPS)
        # Assert the agent reached for tools rather than guessing the arithmetic.
        assert trajectory.tool_sequence
        assert "calculate" in trajectory.tool_sequence

        # Check the TOOL RESULT, not the prose.
        #
        # The first version of this test asserted `"6319" in final_answer` after
        # stripping commas. It failed against a real model that wrote the answer as
        # LaTeX -- `6{,}319` -- so comma-stripping produced "6{}319". The arithmetic
        # was perfect and the assertion was wrong.
        #
        # A neat demonstration of this lesson's own rule, learned the hard way:
        # assert on the trajectory, never on prose. The tool result is deterministic;
        # how the model chooses to format it is not.
        calculations = [
            e.result
            for step in trajectory.steps
            for e in step.executions
            if e.name == "calculate" and e.ok
        ]
        assert calculations, "calculate was requested but never succeeded"
        assert any("6319" in r.replace(",", "") for r in calculations)
