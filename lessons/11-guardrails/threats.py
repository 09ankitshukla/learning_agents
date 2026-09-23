"""The threat model, written down before any control is built.

Skipping this step is how you end up with a pile of string filters that block last
year's attack. A control that is not traceable to a threat is decoration, and a threat
with no control is at least honestly recorded.

The field that matters most here is `stops_it`. Guardrails invite security theatre, and
the difference between "this makes the attack impossible" and "this raises the effort"
is the whole difference between a control and a comfort blanket. Most of the entries
below are the second kind, and saying so is the point of the file.

Each threat names:
  - the **adversary**: who, concretely
  - what they **control**: the actual input channel
  - what a **breach** looks like: how you would know it happened
  - the **control**, and whether it stops the threat or merely narrows it
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    """How much a control actually achieves."""

    STOPS = "stops"          # the attack becomes impossible, by construction
    NARROWS = "narrows"      # the blast radius shrinks; the attack still works
    DETECTS = "detects"      # you find out afterwards, which is not prevention
    NOTHING = "nothing"      # named honestly, with no control


@dataclass(frozen=True)
class Threat:
    id: str
    adversary: str
    controls_what: str
    breach_looks_like: str
    control: str
    verdict: Verdict
    #: Where the control lives, so a reader can go and check it.
    implemented_in: str
    note: str = ""


THREATS: list[Threat] = [
    # -- the one that has no clean fix ------------------------------------------
    Threat(
        id="injection_via_tool_result",
        adversary="anyone who can write a file this agent might read",
        controls_what=(
            "the contents of any file under the sandbox, which read_file and "
            "search_files paste straight into the conversation"
        ),
        breach_looks_like=(
            "the agent follows instructions that came from a file rather than from the "
            "user: writing a note it was not asked for, changing its answer, or "
            "reporting the attacker's text as its own conclusion"
        ),
        control=(
            "wrap every tool result in an explicit untrusted-data envelope naming its "
            "source, and tell the model in the system prompt that envelope contents are "
            "data and never instructions"
        ),
        verdict=Verdict.NARROWS,
        implemented_in="guards.py: wrap_untrusted, UNTRUSTED_SYSTEM_RULE",
        note=(
            "This is the honest centre of the lesson. A model cannot reliably "
            "distinguish instructions from data, because both are just text in the same "
            "context window -- there is no privileged channel. Delimiting and labelling "
            "measurably reduces compliance and does not eliminate it, and no wording of "
            "the prompt will. Treat it as defence in depth behind controls that DO stop "
            "things: the sandbox and the approval gate."
        ),
    ),
    # -- the ones that are genuinely stopped ------------------------------------
    Threat(
        id="path_traversal",
        adversary="the model itself, confused or injected",
        controls_what="the `path` argument to read_file and list_files",
        breach_looks_like="reading /etc/passwd, ~/.ssh/id_rsa, or anything outside the project",
        control=(
            "resolve the path first, then require is_relative_to(SANDBOX). Resolving "
            "before checking is what makes it robust: '..' collapses, symlinks are "
            "followed, and there is no string to smuggle past a blocklist"
        ),
        verdict=Verdict.STOPS,
        implemented_in="lessons/03-agent-loop/toolset.py: _safe_path",
        note=(
            "An allowlist on a resolved absolute path is a real boundary. Lesson 7's "
            "files_refuse_escape case has guarded it since then, and it sits in the "
            "protected category so lesson 9's rule reverts any change that breaks it."
        ),
    ),
    Threat(
        id="secret_exfiltration_by_reading",
        adversary="an injected instruction telling the agent to read credentials",
        controls_what="which in-sandbox file the agent chooses to read",
        breach_looks_like="the contents of .env appearing in the conversation, and therefore in a provider's logs",
        control="a denylist of paths that are inside the sandbox but still refused (.env, .git, .venv)",
        verdict=Verdict.STOPS,
        implemented_in="lessons/03-agent-loop/toolset.py: DENIED_PARTS",
        note=(
            "Separate from containment and worth keeping separate: the allowlist stops "
            "traversal, the denylist expresses 'this specific thing is secret'. Note the "
            "ordering that makes this matter -- once a secret is in the message list it "
            "has already been sent to the provider, so refusing the read is the only "
            "moment the control exists."
        ),
    ),
    Threat(
        id="unauthorised_side_effect",
        adversary="an injected instruction telling the agent to act, not just answer",
        controls_what="the arguments to any tool that changes state",
        breach_looks_like="a file written, a message sent, or a record changed that the user never asked for",
        control="an approval gate in dispatch(): named tools cannot run without an explicit yes",
        verdict=Verdict.STOPS,
        implemented_in="guards.py: ApprovalGate",
        note=(
            "The only control here that stops injection outright, and it does so by not "
            "trusting the model at all. It works precisely because it does not try to "
            "detect an attack -- it requires a human for a class of action regardless of "
            "why the model wanted it. The cost is real: a gate on a frequently-used tool "
            "makes the agent useless, so the gated set has to be small and chosen by "
            "consequence rather than by suspicion."
        ),
    ),
    Threat(
        id="secret_leak_in_output",
        adversary="any path by which a credential reaches the answer",
        controls_what="nothing directly; this is the last line before a human or a log",
        breach_looks_like="an API key or token in the final answer, a trace file, or a terminal scrollback",
        control="pattern-based redaction on tool results and on the final answer",
        verdict=Verdict.NARROWS,
        implemented_in="guards.py: redact_secrets",
        note=(
            "Patterns only catch shapes you anticipated. It is worth having as the last "
            "layer and worth not believing in: a credential in an unexpected format "
            "passes straight through. Prevention is the denylist above."
        ),
    ),
    # -- named, and not solved --------------------------------------------------
    Threat(
        id="phantom_tool",
        adversary="nobody; the model invents a tool that was never offered",
        controls_what="the tool name it requests",
        breach_looks_like="the run aborts with no answer (PHANTOM_TOOL), or a provider rejects the request",
        control="the registry IS the allowlist, so an unknown name cannot reach any code",
        verdict=Verdict.STOPS,
        implemented_in="src/llmkit/tools.py: ToolRegistry.dispatch, check 1",
        note=(
            "Stopped as a security matter and unsolved as a reliability one. Lesson 2 "
            "found it, lesson 9 saw a prompt change reintroduce it three lessons later, "
            "and it still ends a run. Safe, and not fixed."
        ),
    ),
    Threat(
        id="sub_agent_failure_as_finding",
        adversary="nobody; an architecture that loses provenance",
        controls_what="what a sub-agent returns when it gives up",
        breach_looks_like="the parent reports a sub-agent's guess as an established fact",
        control="every non-completion is raised as an error with output marked UNVERIFIED",
        verdict=Verdict.NARROWS,
        implemented_in="lessons/10-multi-agent/team.py: as_tool",
        note=(
            "The label is only as good as the parent's willingness to respect it, which "
            "is the same weakness as the untrusted-data envelope. It is a prompt asking "
            "a model to be careful."
        ),
    ),
    Threat(
        id="cost_exhaustion",
        adversary="an injected instruction designed to burn tokens, or a confused model",
        controls_what="how many steps and how many tools the agent uses",
        breach_looks_like="a daily quota gone, or a bill",
        control="lesson 3's max_steps, lesson 10's DelegationBudget, output truncation in every file tool",
        verdict=Verdict.NARROWS,
        implemented_in="lessons/03-agent-loop/loop.py, lessons/10-multi-agent/team.py",
        note=(
            "Caps bound a single run. Nothing here bounds a sequence of runs, and this "
            "project has exhausted a 200,000-token daily quota twice by accident, with no "
            "adversary at all."
        ),
    ),
]


def by_id(threat_id: str) -> Threat:
    for threat in THREATS:
        if threat.id == threat_id:
            return threat
    known = ", ".join(t.id for t in THREATS)
    raise KeyError(f"No threat {threat_id!r}. Known: {known}")


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for threat in THREATS:
        counts[threat.verdict.value] = counts.get(threat.verdict.value, 0) + 1
    return counts
