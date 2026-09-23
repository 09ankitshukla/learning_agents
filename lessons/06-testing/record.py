"""Record real model responses into a cassette, so tests can replay them offline.

    uv run lessons/06-testing/record.py --list
    uv run lessons/06-testing/record.py --scenario tool_recovery
    uv run lessons/06-testing/record.py --all

This is the only script in lesson 6 that costs tokens, and it is run rarely -- once
per scenario, then committed. The tests themselves never touch the network.

Why bother, when `ScriptedClient` already exists? Because hand-written doubles
drift into fiction. You write what you *think* the model returns, and quietly
encode your misconceptions into the test suite. A cassette preserves what the model
actually did: `content: null` on tool turns, the exact argument JSON, reasoning
token counts, the `finish_reason` values you did not expect.

Use scripted doubles to test specific behaviour, and cassettes to keep the doubles
honest. When a cassette and a hand-written fake disagree, the cassette is right.

Cassettes are committed deliberately. They are test fixtures, they are small, and a
clone should be able to run the whole suite offline with no key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from llmkit import ConfigError, console, get_client

_LESSONS = Path(__file__).resolve().parents[1]
for _folder in ("02-tool-calling", "03-agent-loop"):
    _path = str(_LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from loop import run_agent  # noqa: E402
from toolset import build_registry  # noqa: E402

from fakes import RecordingClient  # noqa: E402


# ---------------------------------------------------------------------------
# Scenarios. Each is a real agent run worth preserving.
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict] = {
    "simple_tool_use": {
        "question": "What time is it in Tokyo right now?",
        "max_steps": 4,
        "why": "The happy path: one tool, one answer. The baseline shape.",
    },
    "multi_step": {
        "question": (
            "What time is it in Mumbai, and if I invoice 2450 USD how much is "
            "that in INR? Then what is 8.25% of that INR amount?"
        ),
        "max_steps": 6,
        "why": "Three sequentially dependent tools. Lesson 3's motivating case.",
    },
    "no_tool_needed": {
        "question": "In one sentence, what is a tool call?",
        "max_steps": 3,
        "why": "The model answers without touching a tool. Tests the early exit.",
    },
    "impossible_task": {
        "question": "What is Amazon's current share price?",
        "max_steps": 4,
        "why": (
            "No tool can do this. Preserves how a real model declines, which is "
            "where COMPLETED-but-not-correct comes from."
        ),
    },
}


def record_scenario(name: str) -> Path | None:
    spec = SCENARIOS[name]
    console.print(Panel.fit(f"recording: {name}", style="bold cyan"))
    console.print(f"[dim]{spec['why']}[/dim]")
    console.print(f"[dim]question: {spec['question']}[/dim]\n")

    try:
        inner = get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return None

    client = RecordingClient(inner, cassette_name=name)
    registry = build_registry()

    try:
        trajectory = run_agent(client, spec["question"], registry, max_steps=spec["max_steps"])
    except Exception as exc:  # noqa: BLE001
        # A failed run is still worth saving if anything was captured -- a recorded
        # rate-limit or provider error is a legitimate fixture.
        console.print(f"[yellow]run ended with {type(exc).__name__}: {str(exc)[:160]}[/yellow]")
        if not client.cassette.entries:
            return None
        trajectory = None

    path = client.save()
    console.print(f"[green]saved[/green] {len(client.cassette.entries)} exchange(s) -> {path}")
    if trajectory is not None:
        console.print(
            f"[dim]stop_reason={trajectory.stop_reason.value}, "
            f"tools={' -> '.join(trajectory.tool_sequence) or '(none)'}, "
            f"{trajectory.usage.total_tokens} tokens[/dim]"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Record model responses for tests.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), help="Record one scenario.")
    parser.add_argument("--all", action="store_true", help="Record every scenario.")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit.")
    args = parser.parse_args()

    if args.list or not (args.scenario or args.all):
        table = Table(title="cassette scenarios")
        table.add_column("name", style="cyan")
        table.add_column("recorded?", width=10)
        table.add_column("why it exists", overflow="fold")
        from fakes import CASSETTE_DIR

        for name, spec in sorted(SCENARIOS.items()):
            exists = (CASSETTE_DIR / f"{name}.json").exists()
            table.add_row(name, "[green]yes[/green]" if exists else "[dim]no[/dim]", spec["why"])
        console.print(table)
        console.print(
            "\n[dim]Recording costs tokens and is done rarely. The test suite "
            "replays these offline and for free.[/dim]"
        )
        return 0

    names = sorted(SCENARIOS) if args.all else [args.scenario]
    for name in names:
        record_scenario(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
