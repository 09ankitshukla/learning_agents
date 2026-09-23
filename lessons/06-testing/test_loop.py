"""Testing the agent loop, using doubles instead of a real model.

The loop is where the stochastic and deterministic parts meet, and it is the part
people assume cannot be tested. It can, completely -- because `ScriptedClient`
lets you decide exactly what the model "says".

That inverts the usual difficulty. Instead of hoping a real model produces the
failure you want to check, you write the failure down:

    ScriptedClient([
        tool_response("calculate", malformed="{bad json"),
        text_response("ok, fixed"),
    ])

Every test here is deterministic, offline, free, and runs in milliseconds.

What to assert on, in order of usefulness:

  1. `trajectory.stop_reason`  -- did it finish, or give up, and why?
  2. `trajectory.tool_sequence` -- did it take the right steps?
  3. the *requests* the client received -- did we send a valid conversation?
  4. the final text -- almost never. Prose varies; it is the least stable thing
     a model produces and the least informative thing to pin.
"""

from __future__ import annotations

import pytest
from fakes import (
    ScriptedClient,
    multi_tool_response,
    starved_response,
    text_response,
    tool_response,
)
from llmkit import ToolRegistry, ToolSpec

from context import validate
from loop import StopReason, run_agent


# ---------------------------------------------------------------------------
# A tiny deterministic registry, so these tests depend on nothing external.
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_registry() -> ToolRegistry:
    calls: list[str] = []

    def echo(value: str) -> str:
        calls.append(value)
        return f"echo: {value}"

    def always_fails(reason: str = "because") -> str:
        from llmkit import ToolError

        raise ToolError(f"cannot do it: {reason}. Try value='ok' instead.")

    registry = ToolRegistry()
    registry.add(
        ToolSpec(
            name="echo",
            description="Echo a value back.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        ),
        echo,
    )
    registry.add(
        ToolSpec(
            name="always_fails",
            description="A tool that always reports a domain error.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": [],
            },
        ),
        always_fails,
    )
    return registry


# ===========================================================================
class TestStopConditions:
    """stop_reason is the loop's most important output. Test every branch."""

    def test_completes_when_the_model_returns_prose(self, tiny_registry) -> None:
        client = ScriptedClient([text_response("the answer is 42")])
        trajectory = run_agent(client, "q", tiny_registry, max_steps=5)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert trajectory.succeeded
        assert trajectory.final_answer == "the answer is 42"
        assert client.call_count == 1

    def test_runs_tools_then_completes(self, tiny_registry) -> None:
        client = ScriptedClient(
            [
                tool_response("echo", {"value": "hello"}),
                text_response("it said hello"),
            ]
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=5)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert trajectory.tool_sequence == ["echo"]
        assert client.call_count == 2

    def test_hits_max_steps_when_the_model_never_stops(self, tiny_registry) -> None:
        """The reason max_steps exists. Without it this test would not terminate."""
        client = ScriptedClient(
            [tool_response("echo", {"value": f"{i}"}, call_id=f"c{i}") for i in range(10)],
            strict=False,
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=3, warn_near_limit=False)

        assert trajectory.stop_reason is StopReason.MAX_STEPS
        assert not trajectory.succeeded
        assert len(trajectory.steps) == 3

    def test_detects_a_stall(self, tiny_registry) -> None:
        """An identical repeated call means no progress."""
        identical = tool_response("echo", {"value": "same"}, call_id="c1")
        client = ScriptedClient([identical, identical, identical], strict=False)
        trajectory = run_agent(client, "q", tiny_registry, max_steps=8, warn_near_limit=False)

        assert trajectory.stop_reason is StopReason.STALLED
        assert len(trajectory.steps) == 2  # stopped as soon as the repeat appeared

    def test_different_arguments_are_not_a_stall(self, tiny_registry) -> None:
        """Guards against over-eager stall detection killing legitimate progress."""
        client = ScriptedClient(
            [
                tool_response("echo", {"value": "a"}, call_id="c1"),
                tool_response("echo", {"value": "b"}, call_id="c2"),
                text_response("done"),
            ]
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=8)
        assert trajectory.stop_reason is StopReason.COMPLETED

    def test_no_answer_is_distinct_from_completed(self, tiny_registry) -> None:
        """Token starvation: no text, no tools, finish_reason=length (lesson 0).

        Impossible to trigger on demand from a real model; trivial with a double.
        Crucially it must NOT be reported as success -- an empty answer presented
        as complete is the worst outcome.
        """
        client = ScriptedClient([starved_response()])
        trajectory = run_agent(client, "q", tiny_registry, max_steps=5)

        assert trajectory.stop_reason is StopReason.NO_ANSWER
        assert not trajectory.succeeded
        assert "reasoning" in (trajectory.note or "").lower()


class TestErrorRecovery:
    """Lesson 2's design paying off: failures become observations, not crashes."""

    def test_a_failing_tool_does_not_end_the_run(self, tiny_registry) -> None:
        client = ScriptedClient(
            [
                tool_response("always_fails", {"reason": "nope"}, call_id="c1"),
                text_response("recovered"),
            ]
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=5)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert len(trajectory.failed_executions) == 1

    def test_the_error_text_is_fed_back_to_the_model(self, tiny_registry) -> None:
        """The mechanism behind self-correction: the model must *see* the error.

        This asserts on what the client received, not on what it returned. A test
        that only checks the final answer would pass even if the error text were
        silently dropped.
        """
        client = ScriptedClient(
            [
                tool_response("always_fails", {"reason": "nope"}, call_id="c1"),
                text_response("recovered"),
            ]
        )
        run_agent(client, "q", tiny_registry, max_steps=5)

        second_request = client.calls[1].messages
        tool_messages = [m for m in second_request if m["role"] == "tool"]
        assert tool_messages
        assert "Try value='ok' instead" in tool_messages[0]["content"]

    def test_unknown_tool_is_reported_and_the_loop_continues(self, tiny_registry) -> None:
        client = ScriptedClient(
            [
                tool_response("get_weather", {"city": "Mumbai"}, call_id="c1"),
                text_response("no such tool, sorry"),
            ]
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=5)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert trajectory.failed_executions[0].failure_kind == "unknown_tool"

    def test_malformed_arguments_are_survivable(self, tiny_registry) -> None:
        client = ScriptedClient(
            [
                tool_response("echo", malformed='{"value": "unterminated', call_id="c1"),
                text_response("fixed it"),
            ]
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=5)

        assert trajectory.stop_reason is StopReason.COMPLETED
        assert trajectory.failed_executions[0].failure_kind == "malformed_json"


class TestConversationConstruction:
    """What the loop *sends* matters as much as what it does with replies."""

    def test_every_request_is_a_valid_conversation(self, tiny_registry) -> None:
        """The property test worth having. Checks all requests, not just the last."""
        client = ScriptedClient(
            [
                tool_response("echo", {"value": "a"}, call_id="c1"),
                multi_tool_response([("echo", {"value": "b"}), ("echo", {"value": "c"})]),
                text_response("done"),
            ]
        )
        run_agent(client, "q", tiny_registry, max_steps=5)

        for index, call in enumerate(client.calls):
            assert validate(call.messages) == [], f"request {index} was invalid"

    def test_multiple_tool_calls_all_get_results(self, tiny_registry) -> None:
        """Return two results for three requests and a real provider rejects it."""
        client = ScriptedClient(
            [
                multi_tool_response(
                    [("echo", {"value": "a"}), ("echo", {"value": "b"}), ("echo", {"value": "c"})]
                ),
                text_response("done"),
            ]
        )
        trajectory = run_agent(client, "q", tiny_registry, max_steps=5)

        assert trajectory.tool_sequence == ["echo", "echo", "echo"]
        final_request = client.calls[-1].messages
        assert sum(1 for m in final_request if m["role"] == "tool") == 3

    def test_tools_are_offered_on_every_call(self, tiny_registry) -> None:
        """Forgetting tools on a later call triggers the phantom-tool bug (lesson 2)."""
        client = ScriptedClient(
            [tool_response("echo", {"value": "a"}), text_response("done")]
        )
        run_agent(client, "q", tiny_registry, max_steps=5)

        assert all("echo" in call.tools for call in client.calls)

    def test_system_prompt_is_first_and_task_second(self, tiny_registry) -> None:
        client = ScriptedClient([text_response("done")])
        run_agent(client, "the actual task", tiny_registry, max_steps=3)

        roles = client.calls[0].roles
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert client.calls[0].messages[1]["content"] == "the actual task"


class TestCompactorHook:
    """Lesson 4's integration point, tested without a model."""

    def test_compactor_sees_and_can_rewrite_the_conversation(self, tiny_registry) -> None:
        seen: list[int] = []

        def compactor(messages, step):
            seen.append(step)
            return messages

        client = ScriptedClient(
            [tool_response("echo", {"value": "a"}), text_response("done")]
        )
        run_agent(client, "q", tiny_registry, max_steps=5, compactor=compactor)

        assert seen == [1, 2]  # called before every model call

    def test_no_compactor_means_no_change(self, tiny_registry) -> None:
        """The hook must be a true no-op by default, or lesson 3 changed behaviour."""
        client = ScriptedClient([text_response("done")])
        trajectory = run_agent(client, "q", tiny_registry, max_steps=3)
        assert trajectory.stop_reason is StopReason.COMPLETED

    def test_resuming_from_existing_messages(self, tiny_registry) -> None:
        """Lesson 4's persistence relies on this."""
        from llmkit import assistant, system, user

        history = [system("sys"), user("earlier question"), assistant("earlier answer")]
        client = ScriptedClient([text_response("follow-up answer")])
        run_agent(
            client, "follow-up", tiny_registry, max_steps=3, initial_messages=history
        )

        sent = client.calls[0].messages
        assert sent[1]["content"] == "earlier question"
        assert sent[-1]["content"] == "follow-up"


class TestScriptedClientItself:
    """Test the test double. A lying fake is worse than no test."""

    def test_running_out_of_responses_is_an_error(self, tiny_registry) -> None:
        """Silently repeating the last response would let runaway loops pass."""
        client = ScriptedClient([tool_response("echo", {"value": "a"})])
        with pytest.raises(AssertionError, match="ran out of responses"):
            run_agent(client, "q", tiny_registry, max_steps=5)

    def test_recorded_messages_are_snapshots_not_references(self, tiny_registry) -> None:
        """The loop mutates its message list, so the double must copy.

        Without copying, every recorded call would point at the final conversation
        and all the assertions above would be meaningless while still passing.
        """
        client = ScriptedClient(
            [tool_response("echo", {"value": "a"}), text_response("done")]
        )
        run_agent(client, "q", tiny_registry, max_steps=5)

        assert len(client.calls[0].messages) < len(client.calls[1].messages)
