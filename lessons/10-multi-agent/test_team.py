"""Tests for delegation and handoff. Offline, free, aimed at the failure modes.

Most of these exist because the multi-agent failures are all *silent*. A sub-agent
that gives up returns prose; a sub-agent's tokens vanish from the parent's accounting;
a nested delegation loops without tripping any step cap. None of that raises, so none
of it shows up unless a test looks for it.

`ScriptedClient` from lesson 6 does the work: a sub-agent that runs out of steps is
impossible to provoke on demand from a real model and takes three lines to construct
here, which is the whole argument for hand-written doubles.
"""

from __future__ import annotations

import pytest
from fakes import ScriptedClient, multi_tool_response, text_response, tool_response
from llmkit import ToolCall, Usage
from llmkit.tools import ToolError
from toolset import build_registry

from pipeline import Stage, handoff_messages, research_write_critique, run_pipeline
from team import (
    CRITIC,
    RESEARCHER,
    TEAM,
    Delegation,
    DelegationBudget,
    DelegationLog,
    SubAgent,
    as_tool,
    build_team_registry,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _client(*responses):
    return ScriptedClient(list(responses), strict=False)


def _tiny(name="ask_helper", tools=(), steps=3):
    return SubAgent(
        name=name,
        description="A helper.",
        system_prompt="You help.",
        tools=list(tools),
        max_steps=steps,
    )


def _delegation(agent="ask_researcher", ok=True, prompt=1000, completion=200):
    return Delegation(
        agent=agent,
        task="do a thing",
        stop_reason="completed" if ok else "max_steps",
        steps=2,
        tool_sequence=["read_file"],
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion),
        duration_s=1.0,
        answer="found it" if ok else None,
        ok=ok,
    )


# ---------------------------------------------------------------------------
# A sub-agent is a tool
# ---------------------------------------------------------------------------
def test_sub_agent_becomes_an_ordinary_tool():
    """The central claim of the lesson: nothing about the parent has to change."""
    client = _client(text_response("the answer"))
    spec, fn = as_tool(_tiny(), client, build_registry(), DelegationLog(), DelegationBudget())

    assert spec.name == "ask_helper"
    assert spec.parameters["required"] == ["task"]
    # Same shape any lesson 2 tool has, so the dispatcher needs no special case.
    assert spec.to_wire()["type"] == "function"


def test_a_successful_delegation_returns_labelled_provenance():
    """A bare string carries no hint that it came from a different agent with different
    tools and no view of the conversation. The parent is a model; it needs telling."""
    client = _client(text_response("Lesson 3 covers the agent loop."))
    _, fn = as_tool(_tiny(), client, build_registry(), DelegationLog(), DelegationBudget())

    result = fn(task="which lesson covers the loop?")
    assert result.startswith("[from ask_helper: completed in 1 step(s)")
    assert "Lesson 3 covers the agent loop." in result


def test_delegation_runs_the_sub_agent_with_only_its_own_tools():
    """Least privilege, via ToolRegistry.subset -- written in lesson 2, unused until now."""
    base = build_registry()
    client = _client(tool_response("calculate", {"expression": "2+2"}), text_response("4"))
    sub = _tiny(tools=["calculate"])
    _, fn = as_tool(sub, client, base, DelegationLog(), DelegationBudget())

    fn(task="what is 2+2")
    offered = set(client.calls[0].tools or [])
    assert offered == {"calculate"}
    assert "read_file" not in offered


def test_a_sub_agent_cannot_be_given_a_tool_the_parent_lacks():
    """Asking for an unknown tool silently narrows rather than raising, because a typo
    in a team definition must not hand a specialist powers nobody granted."""
    base = build_registry()
    client = _client(text_response("done"))
    sub = _tiny(tools=["calculate", "launch_missiles"])
    _, fn = as_tool(sub, client, base, DelegationLog(), DelegationBudget())

    fn(task="anything")
    assert set(client.calls[0].tools or []) == {"calculate"}


# ---------------------------------------------------------------------------
# THE failure mode: a sub-agent's failure looks exactly like its answer
# ---------------------------------------------------------------------------
def test_a_sub_agent_that_runs_out_of_steps_raises_rather_than_returning_prose():
    """The most important test in the lesson.

    A sub-agent that hits its step cap still produces text. Returning that text as a
    successful tool result is how a parent ends up presenting a guess as a finding --
    lesson 6's 'COMPLETED does not mean correct', now one level deeper and invisible,
    because the parent sees a string either way."""
    # Three tool requests against max_steps=2: the loop gives up.
    client = _client(
        tool_response("calculate", {"expression": "1+1"}, call_id="a"),
        tool_response("calculate", {"expression": "2+2"}, call_id="b"),
        tool_response("calculate", {"expression": "3+3"}, call_id="c"),
    )
    sub = _tiny(tools=["calculate"], steps=2)
    _, fn = as_tool(sub, client, build_registry(), DelegationLog(), DelegationBudget())

    with pytest.raises(ToolError) as exc:
        fn(task="do several things")
    message = str(exc.value)
    assert "did not finish" in message
    assert "UNVERIFIED" in message or "no output" in message
    assert "Do not present it as fact" in message


def test_an_unfinished_sub_agents_partial_output_is_labelled_unverified():
    """Passing the partial text through is useful; passing it through unlabelled is not."""
    # A turn carrying BOTH a tool request and prose, at max_steps=1: the loop falls out
    # having exhausted its budget and salvages the prose so a caller has something to
    # show. That salvaged text is exactly what must not be passed off as an answer.
    client = _client(
        tool_response(
            "calculate",
            {"expression": "1+1"},
            text="I think it is probably about 40 but I ran out of time",
        ),
    )
    sub = _tiny(tools=["calculate"], steps=1)
    _, fn = as_tool(sub, client, build_registry(), DelegationLog(), DelegationBudget())

    with pytest.raises(ToolError) as exc:
        fn(task="estimate something")
    assert "UNVERIFIED" in str(exc.value)


def test_a_sub_agent_that_crashes_is_reported_as_a_failure_not_an_answer():
    class _Boom:
        config = type("C", (), {"model": "fake-model", "provider": "fake"})()

        def chat(self, *a, **k):
            raise RuntimeError("provider exploded")

    log = DelegationLog()
    _, fn = as_tool(_tiny(), _Boom(), build_registry(), log, DelegationBudget())

    with pytest.raises(ToolError) as exc:
        fn(task="anything")
    assert "Do not treat this as a finding" in str(exc.value)
    # Recorded even though it failed, so the cost and the failure are both visible.
    assert len(log.delegations) == 1
    assert not log.delegations[0].ok


def test_a_failed_delegation_is_still_logged():
    client = _client(
        tool_response("calculate", {"expression": "1+1"}, call_id="a"),
        tool_response("calculate", {"expression": "2+2"}, call_id="b"),
    )
    log = DelegationLog()
    _, fn = as_tool(_tiny(tools=["calculate"], steps=1), client, build_registry(),
                    log, DelegationBudget())
    with pytest.raises(ToolError):
        fn(task="x")

    assert log.calls == 1
    assert log.failures
    assert log.total_tokens > 0  # a failure still costs money


# ---------------------------------------------------------------------------
# Cost that vanishes
# ---------------------------------------------------------------------------
def test_sub_agent_tokens_are_invisible_to_the_parent_trajectory():
    """The fifth silent measurement bug in this project, and the first predicted.

    A sub-agent's model calls happen inside registry.dispatch(), which lesson 3's
    Trajectory does not look at -- it was designed before sub-agents existed. So a
    delegating agent's real cost is trajectory.usage PLUS the delegation log, and any
    tool reading only the first (lesson 7's CaseResult, lesson 8's CostReport)
    under-reports by the difference."""
    from loop import run_agent

    base = build_registry()
    log = DelegationLog()
    client = _client(
        tool_response("ask_helper", {"task": "find it"}),   # parent delegates
        text_response("the sub-agent said: found it"),      # sub-agent answers
        text_response("final answer"),                      # parent concludes
    )
    registry = build_team_registry(base, [_tiny()], client, log)
    trajectory = run_agent(client, "q", registry, max_steps=4)

    parent = trajectory.usage.prompt_tokens + trajectory.usage.completion_tokens
    assert log.total_tokens > 0
    # The whole point: the parent's own accounting does not include the sub-agent.
    assert parent < parent + log.total_tokens
    assert log.hidden_share(parent) > 0.0


def test_hidden_share_is_zero_when_nothing_was_delegated():
    log = DelegationLog()
    assert log.hidden_share(5_000) == 0.0
    assert log.total_tokens == 0


def test_delegation_log_sums_usage_across_calls():
    log = DelegationLog()
    log.record(_delegation(prompt=1_000, completion=100))
    log.record(_delegation(agent="ask_critic", prompt=500, completion=50))
    assert log.total_tokens == 1_650
    assert log.calls == 2
    assert log.agents_used == ["ask_researcher", "ask_critic"]


# ---------------------------------------------------------------------------
# Tool use that vanishes, which is the same blind spot as the cost
# ---------------------------------------------------------------------------
def test_effective_tool_sequence_expands_a_delegation():
    """The sharpest finding in the lesson, and it was nearly missed.

    A probe reported `arith_precision` failing under delegation while the answer was
    visibly correct ("0.285714 (to six decimal places)"). The failing check was
    `used_tools(["calculate"])`: the coordinator's sequence reads ["ask_calculator"]
    because the calculator ran one level down. Ten of sixteen eval cases assert
    `used_tools`, so a full suite run would report a regression created by the
    instrument rather than by the architecture."""
    log = DelegationLog()
    log.record(
        Delegation(
            agent="ask_calculator", task="2/7", stop_reason="completed", steps=2,
            tool_sequence=["calculate"], usage=Usage(prompt_tokens=800, completion_tokens=100),
            duration_s=1.0, answer="0.285714", ok=True,
        )
    )
    assert log.effective_tool_sequence(["ask_calculator"]) == ["calculate"]


def test_effective_tool_sequence_leaves_the_parents_own_tools_alone():
    log = DelegationLog()
    log.record(_delegation(agent="ask_researcher"))
    assert log.effective_tool_sequence(["calculate", "ask_researcher", "get_current_time"]) == [
        "calculate", "read_file", "get_current_time",
    ]


def test_repeated_delegations_to_one_agent_expand_in_order():
    """A single cursor per agent, not one shared cursor: two calls to the same
    specialist must expand to what each of them actually did."""
    log = DelegationLog()
    log.record(
        Delegation(agent="ask_researcher", task="a", stop_reason="completed", steps=1,
                   tool_sequence=["list_files"], usage=Usage(), duration_s=0.0,
                   answer="x", ok=True)
    )
    log.record(
        Delegation(agent="ask_researcher", task="b", stop_reason="completed", steps=1,
                   tool_sequence=["search_files", "read_file"], usage=Usage(), duration_s=0.0,
                   answer="y", ok=True)
    )
    assert log.effective_tool_sequence(["ask_researcher", "ask_researcher"]) == [
        "list_files", "search_files", "read_file",
    ]


def test_an_unlogged_delegation_name_survives_expansion():
    """A call that failed before anything was logged must not vanish from the record.
    Dropping it would understate what the agent attempted."""
    log = DelegationLog()
    assert log.effective_tool_sequence(["ask_researcher"]) == ["ask_researcher"]


def test_expansion_changes_a_used_tools_verdict():
    """End to end: the same answer, scored two ways, with opposite results."""
    from scorers import used_tools

    log = DelegationLog()
    log.record(
        Delegation(agent="ask_calculator", task="t", stop_reason="completed", steps=1,
                   tool_sequence=["calculate"], usage=Usage(), duration_s=0.0,
                   answer="0.285714", ok=True)
    )

    class _Naive:
        final_answer = "0.285714"
        tool_sequence = ["ask_calculator"]

    class _Effective:
        final_answer = "0.285714"
        tool_sequence = log.effective_tool_sequence(["ask_calculator"])

    scorer = used_tools(["calculate"])
    assert not scorer(_Naive).passed
    assert scorer(_Effective).passed


def test_harness_can_correct_an_execution_before_scoring():
    """Lesson 7's new hook corrects what the agent DID, not how it is judged -- which
    keeps the cache-the-execution-not-the-score rule intact. A hook that adjusted the
    verdict would be that bug wearing a different hat."""
    from dataclasses import replace as _replace

    from harness import Execution

    execution = Execution(
        final_answer="0.285714", tool_sequence=["ask_calculator"], stop_reason="completed",
        steps=2, prompt_tokens=100, completion_tokens=20, latency_s=0.1,
    )
    corrected = _replace(execution, tool_sequence=["calculate"])
    assert corrected.tool_sequence == ["calculate"]
    assert corrected.final_answer == execution.final_answer
    assert corrected.succeeded


def test_delegation_log_becomes_lesson_8_spans():
    """Nesting a sub-agent into a trace needed no new machinery, because Span already
    had children. The payoff for building tracing as a view over recorded data."""
    log = DelegationLog()
    log.record(_delegation())
    log.record(_delegation(agent="ask_critic", ok=False))

    spans = log.as_spans()
    assert [s.kind for s in spans] == ["model_call", "model_call"]
    assert spans[0].ok and not spans[1].ok
    assert spans[0].prompt_tokens == 1_000
    # kind == "model_call" matters: lesson 8's rollups filter on it, so a sub-agent
    # span tagged anything else would be silently excluded from every cost figure.
    assert sum(s.total_tokens for s in spans) == log.total_tokens


# ---------------------------------------------------------------------------
# Recursion and breadth
# ---------------------------------------------------------------------------
def test_a_sub_agent_is_offered_no_delegation_tools_by_default():
    """Depth is enforced by withholding the capability, not by refusing the call. A tool
    the model cannot see is a tool it cannot be talked into using."""
    base = build_registry()
    client = _client(text_response("x"))
    nested = build_team_registry(
        base, TEAM, client, DelegationLog(), budget=DelegationBudget(depth=1, max_depth=1)
    )
    assert nested.names == base.names
    assert not [n for n in nested.names if n.startswith("ask_")]


def test_the_coordinator_does_get_delegation_tools():
    base = build_registry()
    client = _client(text_response("x"))
    registry = build_team_registry(base, TEAM, client, DelegationLog())
    assert len(registry) == len(base) + len(TEAM)
    for sub in TEAM:
        assert sub.name in registry


def test_exceeding_the_call_budget_is_a_recoverable_tool_error():
    """Lesson 2's rule: a tool that cannot do its job returns a string for the model to
    read, it does not raise out of the loop."""
    base = build_registry()
    client = _client(text_response("x"))
    budget = DelegationBudget(max_calls=1)
    registry = build_team_registry(base, TEAM, client, DelegationLog(), budget=budget)

    first = registry.dispatch(ToolCall(id="1", name="ask_critic", arguments={"task": "a"}))
    assert first.ok
    second = registry.dispatch(ToolCall(id="2", name="ask_critic", arguments={"task": "b"}))
    assert not second.ok
    assert second.failure_kind == "tool_error"
    assert "budget exhausted" in second.result


def test_the_call_budget_counts_across_different_agents():
    """Breadth, not just repetition. Six specialists once each is the same cost as one
    specialist six times, and a per-agent counter would miss it."""
    base = build_registry()
    client = _client(text_response("x"), text_response("y"), text_response("z"))
    budget = DelegationBudget(max_calls=2)
    registry = build_team_registry(base, TEAM, client, DelegationLog(), budget=budget)

    assert registry.dispatch(ToolCall(id="1", name="ask_critic", arguments={"task": "a"})).ok
    assert registry.dispatch(ToolCall(id="2", name="ask_researcher", arguments={"task": "b"})).ok
    third = registry.dispatch(ToolCall(id="3", name="ask_calculator", arguments={"task": "c"}))
    assert not third.ok


def test_child_budget_increments_depth_and_carries_the_call_count():
    budget = DelegationBudget(max_depth=2, max_calls=5)
    budget.calls = 3
    child = budget.child()
    assert child.depth == 1
    assert child.calls == 3
    assert child.max_calls == 5


def test_router_mode_leaves_the_coordinator_with_only_its_agents():
    base = build_registry()
    client = _client(text_response("x"))
    registry = build_team_registry(base, TEAM, client, DelegationLog(), keep_own_tools=[])
    assert registry.names == sorted(s.name for s in TEAM)


def test_a_sub_agent_name_may_not_shadow_a_real_tool():
    """Registration raises on a duplicate, which is the right moment to find out --
    a shadowed tool would silently route arithmetic through an agent."""
    base = build_registry()
    client = _client(text_response("x"))
    clash = SubAgent(
        name="calculate", description="d", system_prompt="s", tools=[],
    )
    with pytest.raises(ValueError):
        build_team_registry(base, [clash], client, DelegationLog())


# ---------------------------------------------------------------------------
# The team definitions
# ---------------------------------------------------------------------------
def test_every_specialist_has_a_narrower_toolset_than_the_coordinator():
    base = build_registry()
    for sub in TEAM:
        assert set(sub.tools) < set(base.names), sub.name


def test_the_critic_has_no_tools_at_all():
    """It judges what it is sent. Giving it tools would let it go and check, which
    sounds better and makes it a second researcher."""
    assert CRITIC.tools == []


def test_specialist_descriptions_and_system_prompts_are_different_text():
    """One is read by the parent to decide whether to delegate; the other is read by
    the sub-agent to decide how to behave. Reusing one for both is the most common
    multi-agent mistake."""
    for sub in TEAM:
        assert sub.description != sub.system_prompt, sub.name
        assert "You are" not in sub.description, sub.name


def test_researcher_cannot_do_arithmetic_and_calculator_cannot_read_files():
    from team import CALCULATOR

    assert "calculate" not in RESEARCHER.tools
    assert "read_file" not in CALCULATOR.tools


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------
def test_handoff_replaces_the_system_prompt_and_keeps_everything_else():
    """The limitation worth knowing: you cannot change a system prompt mid-conversation,
    so a naive handoff leaves the previous agent's instructions in place and you get one
    agent wearing a second agent's name."""
    messages = [
        {"role": "system", "content": "you are the researcher"},
        {"role": "user", "content": "find x"},
        {"role": "assistant", "content": "found x"},
    ]
    out = handoff_messages(messages, "you are the writer")

    assert out[0] == {"role": "system", "content": "you are the writer"}
    assert out[1:] == messages[1:]
    assert len([m for m in out if m["role"] == "system"]) == 1


def test_handoff_preserves_tool_results():
    """What survives is the point of shared mode, and also its cost."""
    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "the file said X"},
    ]
    out = handoff_messages(messages, "b")
    assert any(m.get("role") == "tool" for m in out)


def test_handoff_on_an_empty_conversation_is_just_a_system_prompt():
    assert handoff_messages([], "s") == [{"role": "system", "content": "s"}]


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def test_pipeline_runs_stages_in_order_and_threads_output_forward():
    client = _client(
        text_response("the notes say the loop is in lesson 3"),
        text_response("Lesson 3 covers the agent loop."),
    )
    stages = [
        Stage(agent=_tiny("finder"), instruction="find: {topic}"),
        Stage(agent=_tiny("phraser"), instruction="write from: {previous}"),
    ]
    run = run_pipeline(client, "where is the loop", build_registry(), stages)

    assert [s.agent for s in run.stages] == ["finder", "phraser"]
    assert run.final_answer == "Lesson 3 covers the agent loop."
    # Stage 2's prompt contained stage 1's output, which is what "threading" means.
    assert "the notes say the loop is in lesson 3" in client.calls[1].messages[-1]["content"]


def test_a_required_stage_failing_stops_the_pipeline_without_an_answer():
    """Passing an unfinished stage's text on as though it were finished is how a
    pipeline launders a failure into a confident final answer three stages later."""
    client = _client(
        tool_response("calculate", {"expression": "1+1"}, call_id="a"),
        tool_response("calculate", {"expression": "2+2"}, call_id="b"),
        text_response("should never run"),
    )
    stages = [
        Stage(agent=_tiny("finder", tools=["calculate"], steps=1), instruction="{topic}"),
        Stage(agent=_tiny("phraser"), instruction="{previous}"),
    ]
    run = run_pipeline(client, "topic", build_registry(), stages)

    assert run.stopped_early
    assert run.final_answer is None
    assert len(run.stages) == 1


def test_an_optional_stage_failing_keeps_the_draft():
    """Losing a review is worse than having one, but it is not a reason to throw away a
    finished draft."""
    client = _client(
        text_response("a good draft"),
        tool_response("calculate", {"expression": "1+1"}, call_id="a"),
        tool_response("calculate", {"expression": "2+2"}, call_id="b"),
    )
    stages = [
        Stage(agent=_tiny("writer"), instruction="{topic}"),
        Stage(agent=_tiny("reviewer", tools=["calculate"], steps=1),
              instruction="{previous}", required=False),
    ]
    run = run_pipeline(client, "topic", build_registry(), stages)

    assert not run.stopped_early
    assert run.final_answer == "a good draft"
    assert run.failures


def test_shared_mode_hands_the_whole_message_list_forward():
    client = _client(text_response("findings"), text_response("draft"))
    stages = [
        Stage(agent=_tiny("finder"), instruction="{topic}"),
        Stage(agent=_tiny("phraser"), instruction="{previous}"),
    ]
    relay = run_pipeline(_client(text_response("findings"), text_response("draft")),
                         "t", build_registry(), stages, mode="relay")
    shared = run_pipeline(client, "t", build_registry(), stages, mode="shared")

    assert relay.stages[1].messages_in == 0
    assert shared.stages[1].messages_in > 0
    # And the second agent's instructions replaced the first's.
    assert client.calls[1].messages[0]["content"] == "You help."


def test_pipeline_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        run_pipeline(_client(), "t", build_registry(), [], mode="telepathy")


def test_a_later_stage_can_address_an_earlier_one_by_name():
    """Relay mode's lossiness, made concrete: {previous} gives the reviser the critique
    alone, so the draft has to be plumbed to it explicitly or it revises something it
    cannot see."""
    client = _client(text_response("the draft"), text_response("problem: too vague"),
                     text_response("the revision"))
    stages = [
        Stage(agent=_tiny("writer"), instruction="{topic}"),
        Stage(agent=_tiny("critic"), instruction="review {previous}"),
        Stage(agent=_tiny("reviser"), instruction="draft was {writer}; verdict {critic}"),
    ]
    run_pipeline(client, "t", build_registry(), stages)

    final_prompt = client.calls[2].messages[-1]["content"]
    assert "the draft" in final_prompt
    assert "too vague" in final_prompt


def test_a_template_referencing_a_stage_that_never_ran_renders_a_placeholder():
    """Rather than raising KeyError halfway through a paid pipeline."""
    client = _client(text_response("only stage"))
    # A stage referencing its own name, which cannot have produced anything yet.
    stages = [Stage(agent=_tiny("solo"), instruction="see {solo}")]
    run = run_pipeline(client, "t", build_registry(), stages)
    assert run.stages[0].ok
    assert "(not available)" in client.calls[0].messages[-1]["content"]


def test_critic_approved_distinguishes_did_not_run_from_not_approved():
    client = _client(text_response("a draft"))
    run = run_pipeline(client, "t", build_registry(),
                       [Stage(agent=_tiny("writer"), instruction="{topic}")])
    assert run.critic_approved is None  # not False


def test_the_canonical_pipeline_reuses_the_delegation_team():
    """A difference between the two architectures should be wiring, not the quality of
    the specialists -- otherwise the comparison measures prompt-writing effort."""
    stages = research_write_critique()
    assert stages[0].agent is RESEARCHER
    assert stages[2].agent is CRITIC
    assert stages[2].required is False


def test_the_revise_pipeline_adds_a_fourth_stage():
    assert len(research_write_critique()) == 3
    assert len(research_write_critique(revise=True)) == 4
