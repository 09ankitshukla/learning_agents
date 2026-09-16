"""Tool registry and dispatcher, promoted from lesson 2 into the shared kit.

Lesson 2 built this by hand, deliberately, because the dispatcher is the security
boundary of an agent and you should write one yourself before you trust one. That
hand-rolled version stays in `lessons/02-tool-calling/tools.py` as the teaching
artifact.

This is the same logic generalised so later lessons stop rebuilding it: the
registry is an object you construct rather than a module-level global, so a lesson
can hold several registries at once (which lesson 10 needs, when different agents
get different tools).

The design rules carried over from lesson 2, unchanged because they are the point:

  * The registry IS the allowlist. Reaching a function requires a successful dict
    lookup against names you registered yourself. No getattr, no eval, no import.
  * `dispatch()` NEVER raises. Every failure returns a string describing what went
    wrong, which goes back to the model as an observation it can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import ToolCall, ToolSpec


class ToolError(Exception):
    """A tool could not do its job, for a reason the model should hear about.

    Distinct from a bug in your code. "That currency is not supported" is a
    ToolError and the model can recover by choosing another. A TypeError in your
    own function is not -- that one is yours to fix.
    """


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    fn: Callable[..., str]


@dataclass
class Execution:
    """What happened when we tried to run one requested tool."""

    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool
    failure_kind: str | None = None
    duration_s: float = 0.0

    @property
    def summary(self) -> str:
        """One-line form for trajectory printouts."""
        status = "ok" if self.ok else f"FAILED:{self.failure_kind}"
        return f"{self.name} -> {status}"


class ToolRegistry:
    """A named collection of tools, plus the dispatcher that runs them."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    # -- construction ---------------------------------------------------
    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"Tool {tool.spec.name!r} is already registered.")
        self._tools[tool.spec.name] = tool

    def add(self, spec: ToolSpec, fn: Callable[..., str]) -> None:
        self.register(Tool(spec, fn))

    def subset(self, names: list[str]) -> ToolRegistry:
        """A narrower registry. Useful for giving one agent fewer powers."""
        return ToolRegistry([self._tools[n] for n in names])

    # -- inspection -----------------------------------------------------
    @property
    def specs(self) -> list[ToolSpec]:
        """What gets sent to the model. Re-sent in full on every call."""
        return [tool.spec for tool in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # -- execution ------------------------------------------------------
    def dispatch(self, call: ToolCall) -> Execution:
        """Run one requested tool. Never raises.

        Check order is the security model:
          1. name registered?      blocks hallucinated tools
          2. arguments valid JSON? blocks malformed model output
          3. required args present? precise, actionable error
          4. signature accepts?    blocks wrong/extra arguments
          5. tool objected?        recoverable domain error
          6. our code broke?       a bug, reported without a traceback
        """
        import time

        started = time.perf_counter()

        def done(result: str, ok: bool, kind: str | None = None) -> Execution:
            return Execution(
                name=call.name,
                arguments=call.arguments if call.is_valid else {},
                result=result,
                ok=ok,
                failure_kind=kind,
                duration_s=time.perf_counter() - started,
            )

        # 1. Unknown name -- the registry is the allowlist.
        tool = self._tools.get(call.name)
        if tool is None:
            return done(
                f"Error: no tool named '{call.name}'. "
                f"Available tools: {', '.join(self.names)}.",
                ok=False,
                kind="unknown_tool",
            )

        # 2. The model wrote invalid JSON for the arguments.
        if not call.is_valid:
            return done(
                f"Error: your arguments for '{call.name}' were not valid JSON. "
                f"Received: {call.malformed_arguments!r}. "
                f"Send a JSON object matching the tool's schema.",
                ok=False,
                kind="malformed_json",
            )

        # 3. Required arguments, checked explicitly so the error names the field.
        missing = set(tool.spec.parameters.get("required", [])) - set(call.arguments)
        if missing:
            return done(
                f"Error: '{call.name}' is missing required argument(s): "
                f"{', '.join(sorted(missing))}.",
                ok=False,
                kind="missing_argument",
            )

        # 4-6. Run it.
        try:
            return done(tool.fn(**call.arguments), ok=True)
        except ToolError as exc:
            return done(f"Error: {exc}", ok=False, kind="tool_error")
        except TypeError as exc:
            return done(
                f"Error: '{call.name}' rejected those arguments: {exc}",
                ok=False,
                kind="bad_signature",
            )
        except Exception as exc:  # noqa: BLE001
            # A genuine bug in the tool. Report the type but never the traceback:
            # source paths are noise to the model and can leak information you
            # would rather not put in a prompt.
            return done(
                f"Error: '{call.name}' failed unexpectedly ({type(exc).__name__}).",
                ok=False,
                kind="internal_error",
            )
