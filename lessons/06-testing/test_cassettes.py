"""Tests that replay real recorded model responses. Offline, free, deterministic.

These exist to keep the hand-written doubles in test_loop.py honest. A scripted
fake encodes what you *believe* the model does; a cassette preserves what it
actually did, including the parts you would not have thought to fake.

Re-record with:

    uv run lessons/06-testing/record.py --all

If a cassette is missing these tests skip rather than fail, so a fresh clone with
no API key still gets a green suite.
"""

from __future__ import annotations

import pytest
from fakes import Cassette, CassetteClient

from loop import StopReason, run_agent


def load(name: str) -> Cassette:
    try:
        return Cassette.load(name)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


pytestmark = pytest.mark.cassette


class TestRecordedRuns:
    def test_simple_tool_use(self, registry) -> None:
        client = CassetteClient(load("simple_tool_use"))
        trajectory = run_agent(client, "What time is it in Tokyo right now?", registry, max_steps=4)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert trajectory.tool_sequence == ["get_current_time"]
        assert trajectory.final_answer

    def test_multi_step_dependent_tools(self, registry) -> None:
        """Lesson 3's motivating case, preserved exactly as the model performed it."""
        client = CassetteClient(load("multi_step"))
        trajectory = run_agent(
            client,
            "What time is it in Mumbai, and if I invoice 2450 USD how much is that "
            "in INR? Then what is 8.25% of that INR amount?",
            registry,
            max_steps=6,
        )

        assert trajectory.stop_reason is StopReason.COMPLETED
        # Asserting on the sequence, not the prose. This is the durable assertion.
        assert trajectory.tool_sequence == [
            "get_current_time",
            "convert_currency",
            "calculate",
        ]

    def test_answers_without_tools_when_none_are_needed(self, registry) -> None:
        client = CassetteClient(load("no_tool_needed"))
        trajectory = run_agent(client, "In one sentence, what is a tool call?", registry, max_steps=3)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert trajectory.tool_sequence == []

    def test_completed_does_not_mean_correct(self, registry) -> None:
        """The limitation lesson 3 documented, now pinned by a real recording.

        Asked for a live share price with no tool that can fetch one, the model
        declines and the loop reports COMPLETED. A graceful refusal and a correct
        answer are indistinguishable at this level -- which is exactly why lesson 7
        needs task-level scoring.
        """
        client = CassetteClient(load("impossible_task"))
        trajectory = run_agent(client, "What is Amazon's current share price?", registry, max_steps=4)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert trajectory.succeeded  # by the loop's definition
        assert trajectory.tool_sequence == []
        # A real evaluation would mark this a failure. stop_reason cannot.


class TestRealModelQuirks:
    """Properties of genuine responses that a hand-written fake tends to miss."""

    def test_tool_turns_carry_no_prose(self) -> None:
        """`content` is null when the model requests a tool.

        Code that assumes text is always present breaks on real traffic. Worth
        pinning from a recording rather than trusting memory.
        """
        cassette = load("simple_tool_use")
        tool_turns = [
            e for e in cassette.entries if e["response"].get("tool_calls")
        ]
        assert tool_turns
        assert all(t["response"].get("text") in (None, "") for t in tool_turns)

    def test_finish_reason_signals_tool_calls(self) -> None:
        cassette = load("simple_tool_use")
        tool_turns = [e for e in cassette.entries if e["response"].get("tool_calls")]
        assert all(t["response"]["finish_reason"] == "tool_calls" for t in tool_turns)

    def test_reasoning_tokens_are_reported_and_substantial(self) -> None:
        """Lesson 0's finding, preserved as evidence rather than an anecdote."""
        cassette = load("multi_step")
        usages = [e["response"]["usage"] for e in cassette.entries]
        assert any(u.get("reasoning_tokens", 0) > 0 for u in usages)

    def test_tool_arguments_arrive_parsed_and_sensible(self) -> None:
        """The model mapped 'Tokyo' to a valid IANA timezone unprompted."""
        cassette = load("simple_tool_use")
        calls = [c for e in cassette.entries for c in e["response"].get("tool_calls") or []]
        assert calls
        assert any("Tokyo" in str(c["arguments"].get("timezone", "")) for c in calls)


class TestCassetteMechanics:
    """Test the harness. A fixture that lies is worse than no fixture."""

    def test_replay_is_deterministic(self, registry) -> None:
        """Same cassette, two runs, identical trajectories. The whole point."""
        question = "What time is it in Tokyo right now?"
        first = run_agent(CassetteClient(load("simple_tool_use")), question, registry, max_steps=4)
        second = run_agent(CassetteClient(load("simple_tool_use")), question, registry, max_steps=4)

        assert first.tool_sequence == second.tool_sequence
        assert first.final_answer == second.final_answer
        assert first.stop_reason is second.stop_reason

    def test_exhausting_a_cassette_is_an_error(self, registry) -> None:
        """Silently repeating the last response would mask a runaway loop."""
        cassette = load("multi_step")
        client = CassetteClient(Cassette(name="trimmed", entries=cassette.entries[:1]))
        with pytest.raises(AssertionError, match="exhausted"):
            run_agent(client, "a question needing many steps", registry, max_steps=6)

    def test_strict_match_rejects_a_changed_conversation(self, registry) -> None:
        """With strict_match, altering the prompt fails loudly instead of replaying
        a response recorded for a different request."""
        client = CassetteClient(load("simple_tool_use"), strict_match=True)
        with pytest.raises(AssertionError, match="no unused entry"):
            run_agent(
                client,
                "a completely different question",
                registry,
                max_steps=3,
                system_prompt="a completely different system prompt",
            )
