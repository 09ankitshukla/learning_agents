"""Verify the environment and measure what this machine can actually do.

Run this first, and re-run it any time a lesson misbehaves in a way that smells
like infrastructure rather than logic.

    uv run lessons/00-setup/check_env.py

Six checks:
  1. Python version
  2. dependencies importable
  3. configuration resolves
  4. model server reachable
  5. the configured model exists
  6. a real generation, timed -- so you know your tokens/sec and your
     tool-calling reliability before you start building on top of them
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

# This script's whole job is diagnosing a broken environment, so it must survive
# its own dependencies being absent. `llmkit` is imported defensively here and
# checked properly below; everything else uses the shared UTF-8-safe console.
# See src/llmkit/terminal.py for why that matters on Windows.
try:
    from llmkit import console
except ImportError:
    from rich.console import Console

    console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]
PASS, FAIL, WARN = "[green]PASS[/green]", "[red]FAIL[/red]", "[yellow]WARN[/yellow]"


def main() -> int:
    console.print(Panel.fit("Environment check", style="bold cyan"))
    results: list[tuple[str, str, str]] = []

    # -- 1. Python ---------------------------------------------------------
    v = sys.version_info
    py_ok = (v.major, v.minor) >= (3, 11)
    results.append(
        (
            "Python >= 3.11",
            PASS if py_ok else FAIL,
            f"{platform.python_version()} on {platform.system()}",
        )
    )
    if not py_ok:
        _render(results)
        console.print("\n[red]Python 3.11+ required.[/red] Try:  uv python install 3.12")
        return 1

    # -- 2. Dependencies ---------------------------------------------------
    try:
        import llmkit  # noqa: F401

        results.append(("llmkit importable", PASS, "shared model layer found"))
    except ImportError as exc:
        results.append(("llmkit importable", FAIL, str(exc)))
        _render(results)
        console.print(
            "\n[red]Dependencies are not installed.[/red]\n"
            "Run from the repo root:\n    uv sync"
        )
        return 1

    from llmkit import ConfigError, ToolSpec, get_client, load_config, system, user

    # -- 3. Config ---------------------------------------------------------
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        results.append((".env present", WARN, "missing -- using defaults (ollama)"))
    else:
        results.append((".env present", PASS, str(env_file.name)))

    try:
        config = load_config()
        results.append(("Config resolves", PASS, config.describe()))
    except ConfigError as exc:
        results.append(("Config resolves", FAIL, str(exc).splitlines()[0]))
        _render(results)
        console.print(f"\n[red]{exc}[/red]")
        return 1

    # -- 4. Server reachable ----------------------------------------------
    client = get_client()
    healthy, detail = client.health()
    results.append(("Server reachable", PASS if healthy else FAIL, detail))
    if not healthy:
        _render(results)
        console.print(Panel(_fix_hint(config, detail), title="How to fix", style="yellow"))
        return 1

    # -- 5. Model present -------------------------------------------------
    installed = client.available_models()
    if installed:
        # Ollama reports "qwen2.5:7b-instruct"; some servers drop the tag.
        found = any(config.model == m or config.model.split(":")[0] in m for m in installed)
        results.append(
            (
                "Model available",
                PASS if found else FAIL,
                config.model if found else f"{config.model} not in {installed}",
            )
        )
        if not found:
            _render(results)
            if config.is_local:
                console.print(f"\n[yellow]Pull it:[/yellow]  ollama pull {config.model}")
            else:
                # Hosted providers retire models regularly. This is a normal,
                # recurring maintenance event, not an exotic failure.
                console.print(
                    f"\n[yellow]{config.provider} does not serve "
                    f"{config.model!r}.[/yellow] Models get retired; pick a current one "
                    "from the list above and set LLM_MODEL in .env."
                )
            return 1
    else:
        results.append(("Model available", WARN, "server did not list models"))

    # -- 6. Real generation, timed ----------------------------------------
    console.print("\n[dim]Running a live generation (this is the slow part)...[/dim]")
    try:
        # max_tokens is generous on purpose. Reasoning models spend most of their
        # output budget on hidden deliberation, so a tight cap here produces an
        # empty answer and a misleading FAIL.
        reply = client.chat(
            [
                system("You are concise."),
                user("Name three primary colours. Answer in one short sentence."),
            ],
            max_tokens=512,
        )
    except Exception as exc:  # noqa: BLE001
        results.append(("Generation works", FAIL, f"{type(exc).__name__}"))
        _render(results)
        console.print(f"\n[red]{exc}[/red]")
        return 1

    tps = reply.usage.tokens_per_second
    detail = (
        f"{reply.usage.completion_tokens} tokens in {reply.usage.latency_s:.1f}s "
        f"({tps:.1f} tok/s)"
    )
    status = PASS
    if reply.starved:
        status = WARN
        detail += " -- all budget spent reasoning, no answer"
    elif reply.truncated:
        status = WARN
        detail += " -- truncated at max_tokens"
    results.append(("Generation works", status, detail))

    # Reasoning models are now common enough that not knowing you have one is a
    # recurring source of confusion. Surface it explicitly.
    if reply.usage.reasoning_tokens:
        share = reply.usage.reasoning_tokens / max(1, reply.usage.completion_tokens)
        results.append(
            (
                "Reasoning model",
                PASS,
                f"yes - {reply.usage.reasoning_tokens}/{reply.usage.completion_tokens} "
                f"output tokens ({share:.0%}) were hidden reasoning",
            )
        )

    # -- 6b. Tool-calling probe -------------------------------------------
    # This matters more than raw speed. An agent is useless if the model cannot
    # reliably emit a tool call, and small quantised models are hit-and-miss.
    console.print("[dim]Probing tool-calling support...[/dim]")
    probe = ToolSpec(
        name="get_temperature",
        description="Get the current temperature for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    )
    try:
        tool_reply = client.chat(
            [user("What is the temperature in Mumbai right now? Use the tool.")],
            tools=[probe],
            max_tokens=512,
        )
        # "requested", not "called" -- the wording matters. No function is
        # executed here, and none exists: `probe` is a description only. The
        # model emits a structured request; running it is your code's job, which
        # is what lesson 2 builds.
        if tool_reply.wants_tools:
            call = tool_reply.tool_calls[0]
            if call.is_valid:
                results.append(
                    ("Tool calling", PASS, f"requested {call.name}({call.arguments})")
                )
            else:
                results.append(
                    ("Tool calling", WARN, f"requested {call.name} with malformed JSON")
                )
        else:
            results.append(
                ("Tool calling", WARN, "model replied with text instead of a tool call")
            )
    except Exception as exc:  # noqa: BLE001
        results.append(("Tool calling", WARN, f"{type(exc).__name__}: unsupported?"))

    _render(results)
    _advise(config, tps, reply.text or "")
    return 0


def _fix_hint(config, detail: str) -> str:
    """Advice for the provider actually configured, not a generic wall of text."""
    if "key rejected" in detail:
        return (
            f"{config.provider} rejected your API key.\n\n"
            "  1. Check LLM_API_KEY in .env for typos or stray whitespace.\n"
            f"  2. Confirm the key is still active in the {config.provider} console.\n"
            "  3. Keys are secrets: keep them in .env, which is gitignored.\n\n"
            "No working key? Switch to a local model instead:\n"
            "  LLM_PROVIDER=ollama\n"
            "  LLM_MODEL=qwen2.5:7b-instruct"
        )

    if config.is_local:
        return (
            f"Nothing is listening at {config.base_url}\n\n"
            "  1. Install:    winget install Ollama.Ollama\n"
            "  2. Start it:   Windows runs it as a service; elsewhere `ollama serve`\n"
            f"  3. Pull model: ollama pull {config.model}\n\n"
            "Prefer a hosted model? Get a free key at console.groq.com, then in .env:\n"
            "  LLM_PROVIDER=groq\n"
            "  LLM_MODEL=openai/gpt-oss-120b\n"
            "  LLM_API_KEY=gsk_..."
        )

    return (
        f"Could not reach {config.provider} ({detail}).\n\n"
        "  1. Check your network connection.\n"
        "  2. Check LLM_BASE_URL in .env if you overrode it.\n"
        f"  3. Check the {config.provider} status page for an outage."
    )


def _render(results: list[tuple[str, str, str]]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", width=22)
    table.add_column("Result", width=8)
    table.add_column("Detail", overflow="fold")
    for name, status, detail in results:
        table.add_row(name, status, detail)
    console.print()
    console.print(table)


def _advise(config, tps: float, sample: str) -> None:
    console.print(f"\n[dim]Model said:[/dim] {sample.strip()[:200]}")

    if not config.is_local:
        console.print(
            f"\n[green]Ready.[/green] Using hosted provider "
            f"[bold]{config.provider}[/bold] -- fast enough for every lesson."
        )
    elif tps >= 25:
        speed = "[green]comfortable[/green] for all lessons"
    elif tps >= 8:
        speed = (
            "[yellow]workable[/yellow] for lessons 1-6. For the evaluation lessons "
            "(7-9), which make hundreds of calls, add a free Groq key or expect long runs"
        )
    else:
        speed = (
            "[red]slow[/red]. Switch to qwen2.5:3b-instruct for iteration, and get a "
            "free Groq key before lesson 7"
        )

    if config.is_local:
        console.print(f"\n[green]Ready.[/green] At {tps:.1f} tok/s this machine is {speed}.")

    console.print("\nNext:  uv run lessons/00-setup/hello_model.py")


if __name__ == "__main__":
    sys.exit(main())
