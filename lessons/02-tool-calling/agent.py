"""Lesson 2 deliverable: a tool-using assistant, wired by hand.

    uv run lessons/02-tool-calling/agent.py
    uv run lessons/02-tool-calling/agent.py --question "what time is it in Tokyo?"
    uv run lessons/02-tool-calling/agent.py --no-tools        # the same question, crippled
    uv run lessons/02-tool-calling/agent.py --show-wire       # the literal JSON sent
    uv run lessons/02-tool-calling/agent.py --arithmetic      # model vs. calculator

There is no loop in this file. One user question produces at most one round of
tool calls, then a final answer. That is a deliberate limitation: it keeps the
five steps visible. Lesson 3 turns it into a loop, at which point it becomes an
agent proper.

The five steps, which are the entire content of this lesson:

    1. SEND      the question plus descriptions of available tools
    2. RECEIVE   a structured request for a tool, instead of prose
    3. EXECUTE   your code runs the function -- the model never does
    4. RETURN    the result goes back as a `role: "tool"` message
    5. ANSWER    the model writes prose using the result
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from llmkit import ConfigError, LLMResponse, Usage, console, get_client, system, tool_result, user
from llmkit.openai_compat import PhantomToolCall
from tools import ALL_SPECS, REGISTRY, dispatch

# Reasoning models spend most of their output budget thinking (lesson 0), so a
# tight cap here yields empty answers rather than short ones.
MAX_TOKENS = 800

SYSTEM_PROMPT = (
    "You are a precise assistant with access to tools. "
    "Use a tool whenever it can give you a fact you cannot know for certain, "
    "including the current time and any arithmetic. "
    "Never guess a value a tool could give you exactly. "
    "After receiving tool results, answer the user's question directly and concisely."
)

# Used by --no-tools. The difference from SYSTEM_PROMPT is the whole point of that
# experiment: a prompt that insists on tools while no tools exist is a
# contradiction, and the model resolves it by inventing a tool. Tell it instead
# that admitting ignorance is acceptable and it does so correctly.
NO_TOOLS_SYSTEM_PROMPT = (
    "You are a precise assistant. You have no tools. "
    "If you cannot know something for certain, say so plainly instead of guessing."
)

# One tool answers this, so the full five steps complete and you see a real
# prose answer built from a real tool result.
DEFAULT_QUESTION = "What time is it in Mumbai right now, and what day of the week is it?"

# This one needs three tools in sequence: the time, then a conversion, then a
# percentage of the conversion's result. It cannot be satisfied in one round,
# because the model cannot ask for the percentage until it has seen the converted
# amount. Run it with --multi-step to watch this script hit its own ceiling.
MULTI_STEP_QUESTION = (
    "What time is it in Mumbai right now, and if I invoice 2,450 USD today, "
    "how much is that in INR? Also what is 8.25% of that INR amount?"
)


def answer(
    client,
    question: str,
    use_tools: bool = True,
    verbose: bool = True,
    system_prompt: str | None = None,
) -> tuple[str | None, Usage, list[dict]]:
    """Run the five steps. Returns the final text, total usage, and the transcript."""
    messages: list[dict] = [system(system_prompt or SYSTEM_PROMPT), user(question)]
    total = Usage()
    tools = ALL_SPECS if use_tools else None

    # ---- STEP 1 + 2: send, and see what comes back --------------------
    if verbose:
        console.print(Rule("Step 1-2: send the question, receive a tool request"))
        console.print(
            f"[dim]offering {len(tools) if tools else 0} tool(s): "
            f"{', '.join(t.name for t in tools) if tools else 'none'}[/dim]"
        )

    try:
        first: LLMResponse = client.chat(messages, tools=tools, max_tokens=MAX_TOKENS)
    except PhantomToolCall as exc:
        # Reached most often via --no-tools. Not a bug in this script: the model
        # went looking for a capability we did not give it. See the class
        # docstring in llmkit/openai_compat.py.
        console.print(
            Panel(
                str(exc),
                title="the model reached for a tool that does not exist",
                style="red",
            )
        )
        console.print(
            "[bold]This is the lesson, not an error.[/bold] Withholding tools does not reliably "
            "make a model admit ignorance -- it may invent a capability instead.\n"
            "It is also live proof that scenario 1 in failures.py is not hypothetical: "
            "models really do request tools you never published, which is why the "
            "dispatcher rejects unknown names by default."
        )
        return None, total, messages

    total = total + first.usage
    messages.append(first.as_message())

    if verbose:
        _describe_response(first)

    # No tool request means the model either answered from its own knowledge or
    # admitted it could not. Both are worth seeing -- try --no-tools.
    if not first.wants_tools:
        if verbose:
            console.print(
                Panel(
                    first.text or "[red](empty)[/red]",
                    title="answered without tools",
                    style="yellow",
                )
            )
        return first.text, total, messages

    # ---- STEP 3: execute. This is your code, not the model's. ----------
    if verbose:
        console.print(Rule("Step 3: your code executes the requested tools"))

    executions = []
    for call in first.tool_calls:
        execution = dispatch(call)
        executions.append(execution)

        # ---- STEP 4: the result becomes a message ----------------------
        # `call.id` is what pairs this result with the request. Get it wrong and
        # the provider rejects the whole conversation.
        messages.append(tool_result(call.id, execution.result))

        if verbose:
            style = "green" if execution.ok else "red"
            marker = "ok" if execution.ok else f"FAILED ({execution.failure_kind})"
            console.print(
                f"  [{style}]{marker}[/{style}] "
                f"{execution.name}({json.dumps(execution.arguments)})"
            )
            console.print(f"       -> [dim]{execution.result}[/dim]")

    # ---- STEP 5: send everything back for a final answer ---------------
    if verbose:
        console.print(Rule("Step 5: send the results back for a prose answer"))
        console.print(
            f"[dim]the conversation is now {len(messages)} messages: "
            f"{' -> '.join(m['role'] for m in messages)}[/dim]"
        )

    # Tools are offered again on purpose. A real agent must be allowed to ask for
    # more -- and if it does here, we stop and say so, because handling that
    # properly *is* the loop, which is lesson 3.
    second: LLMResponse = client.chat(messages, tools=tools, max_tokens=MAX_TOKENS)
    total = total + second.usage
    messages.append(second.as_message())

    if verbose:
        _describe_response(second)

    if second.wants_tools:
        wanted = ", ".join(c.name for c in second.tool_calls)
        console.print(
            Panel(
                f"The model asked for more tools ({wanted}) instead of answering.\n\n"
                "This script stops here by design: it handles exactly one round. "
                "Handling 'keep going until done' is the agent loop, and it is the "
                "whole of lesson 3. You have just found the reason that lesson exists.",
                title="hit the one-round limit",
                style="yellow",
            )
        )
        return second.text, total, messages

    if verbose:
        console.print(
            Panel(second.text or "[red](empty)[/red]", title="final answer", style="green")
        )
    return second.text, total, messages


def _describe_response(reply: LLMResponse) -> None:
    """Show what the model actually returned, in wire terms."""
    bits = [f"finish_reason={reply.finish_reason}"]
    bits.append(f"content={'null' if reply.text is None else repr(reply.text[:60])}")
    if reply.tool_calls:
        bits.append(f"tool_calls={len(reply.tool_calls)}")
    console.print(f"[dim]{' | '.join(bits)}[/dim]")

    for call in reply.tool_calls:
        if call.is_valid:
            console.print(
                f"  [cyan]requested[/cyan] {call.name}"
                f"({json.dumps(call.arguments)})  [dim]id={call.id}[/dim]"
            )
        else:
            console.print(
                f"  [red]requested[/red] {call.name} with malformed JSON: "
                f"{call.malformed_arguments!r}"
            )

    if reply.usage.reasoning_tokens:
        console.print(
            f"[dim]  ({reply.usage.reasoning_tokens} of "
            f"{reply.usage.completion_tokens} output tokens were hidden reasoning)[/dim]"
        )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def show_wire(question: str, model: str) -> None:
    """Print the literal JSON body sent to the provider. No magic anywhere."""
    body = {
        "model": model,
        "messages": [system(SYSTEM_PROMPT), user(question)],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "tools": [spec.to_wire() for spec in ALL_SPECS],
        "tool_choice": "auto",
    }
    console.print(
        Panel(
            Syntax(json.dumps(body, indent=2), "json", theme="ansi_dark", word_wrap=True),
            title="the entire HTTP request body",
            style="cyan",
        )
    )
    console.print(
        "Every tool you own is re-sent, in full, on every single call. Tool schemas "
        "are part of your prompt and they consume context.\n"
        "Ten verbose tools can cost more input tokens than the conversation itself, "
        "which is why 'just add another tool' is not free."
    )


def arithmetic_comparison(client) -> None:
    """The same sum, with and without a calculator."""
    console.print(Rule("Arithmetic: model alone vs. model with a tool"))

    problem = "What is 48239 * 7841 + 15% of 92300?"
    truth = 48239 * 7841 + 0.15 * 92300

    console.print(f"[bold]question:[/bold] {problem}")
    console.print(f"[bold]true answer:[/bold] {truth:,.2f}\n")

    unaided = client.chat(
        [system("Answer with just the number."), user(problem)], max_tokens=MAX_TOKENS
    )
    console.print(f"[yellow]without tools[/yellow] {(unaided.text or '').strip()[:120]}")

    aided, _, _ = answer(client, problem, use_tools=True, verbose=False)
    console.print(f"[green]with tools   [/green] {(aided or '').strip()[:200]}")

    console.print(
        "\nA capable model often gets this right unaided, especially a reasoning "
        "model that works digit by digit. That is not the point.\n"
        "The point is that unaided it is [bold]sometimes[/bold] right, and with the tool it is "
        "[bold]always[/bold] right, for a fraction of the reasoning tokens. Correctness you can "
        "depend on beats correctness you have to spot-check."
    )


def no_tools_experiment(client, question: str) -> None:
    """No tools, two different system prompts. The contrast is the lesson.

    A capability gap cannot be prompted away -- but *how* you frame the gap
    decides whether the model fails honestly or invents a way around it.
    """
    console.print(Rule("No tools, prompt A: 'use tools for facts you cannot know'"))
    console.print(
        "[dim]The same system prompt the tool-enabled version uses. It insists on tool "
        "use, but the tool list is now empty -- a contradiction.[/dim]\n"
    )
    answer(client, question, use_tools=False, system_prompt=SYSTEM_PROMPT)

    console.print(Rule("No tools, prompt B: 'you have no tools; say so if unsure'"))
    console.print("[dim]Same model, same question, honest framing.[/dim]\n")
    answer(client, question, use_tools=False, system_prompt=NO_TOOLS_SYSTEM_PROMPT)

    console.print(
        Panel(
            "Prompt A usually fails, and interestingly: told to use a tool with no tools "
            "available, the model invents one. Observed attempts include "
            "[cyan]container.exec[/cyan] (run a Python script) and [cyan]browser.search[/cyan] -- both "
            "tools from its training environment, neither offered by us. The provider "
            "rejects the generation with HTTP 400.\n\n"
            "Prompt B succeeds: it states plainly that it cannot access real-time data, "
            "and usually offers what it does know (that Mumbai is UTC+05:30).\n\n"
            "[bold]Two lessons.[/bold]\n"
            "1. A capability gap cannot be prompted away. Neither prompt produces the "
            "actual time, because there is no clock to read.\n"
            "2. Your system prompt and your tool list must agree. A prompt that insists "
            "on tools you did not provide is a contradiction, and the model resolves it "
            "by hallucinating. If you disable tools at runtime, update the prompt too.\n\n"
            "It also makes failures.py scenario 1 concrete: models genuinely do request "
            "tools that were never published, which is exactly why the dispatcher rejects "
            "unknown names by default.",
            title="what just happened",
            style="cyan",
        )
    )


def print_transcript(messages: list[dict]) -> None:
    """The message list, which is the only state an agent has."""
    console.print(Rule("The full conversation"))
    table = Table(show_lines=True)
    table.add_column("#", width=3)
    table.add_column("role", width=10)
    table.add_column("content / tool_calls", overflow="fold")

    for i, msg in enumerate(messages):
        content = msg.get("content")
        rendered = "" if content is None else str(content)
        if msg.get("tool_calls"):
            calls = "\n".join(
                f"-> {c['function']['name']}({c['function']['arguments']})"
                for c in msg["tool_calls"]
            )
            rendered = (rendered + "\n" + calls).strip() or calls
        if msg.get("tool_call_id"):
            rendered = f"[dim](for {msg['tool_call_id']})[/dim] {rendered}"
        table.add_row(str(i), msg["role"], rendered or "[dim]null[/dim]")

    console.print(table)
    console.print(
        "That list is the entire memory of this interaction. There is no state on "
        "the server -- lesson 0, experiment 2. Note that it grew from 2 messages to "
        f"{len(messages)}, and every one of them is re-sent on each call. Lesson 4 is "
        "about keeping this from growing without bound."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="A hand-wired tool-using assistant.")
    parser.add_argument("--question", help="Ask something else.")
    parser.add_argument(
        "--no-tools", action="store_true", help="Withhold the tools, to see the difference."
    )
    parser.add_argument(
        "--show-wire", action="store_true", help="Print the raw JSON request and exit."
    )
    parser.add_argument(
        "--arithmetic", action="store_true", help="Compare model arithmetic vs the calculator."
    )
    parser.add_argument(
        "--multi-step",
        action="store_true",
        help="Ask a question needing several sequential tools, to hit the one-round limit.",
    )
    parser.add_argument(
        "--transcript", action="store_true", help="Print the full message list at the end."
    )
    parser.add_argument("--list-tools", action="store_true", help="Show the registry and exit.")
    args = parser.parse_args()

    question = args.question or (MULTI_STEP_QUESTION if args.multi_step else DEFAULT_QUESTION)

    if args.list_tools:
        table = Table(title="tool registry")
        table.add_column("name", style="cyan")
        table.add_column("required args")
        table.add_column("description", overflow="fold")
        for name, tool in sorted(REGISTRY.items()):
            table.add_row(
                name,
                ", ".join(tool.spec.parameters.get("required", [])) or "[dim]none[/dim]",
                tool.spec.description,
            )
        console.print(table)
        console.print(
            "\nThis dict IS the allowlist. A tool name the model invents cannot be "
            "reached, because reaching a function requires a lookup here."
        )
        return 0

    try:
        client = get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if args.show_wire:
        show_wire(question, client.config.model)
        return 0

    console.print(Panel.fit(client.config.describe(), style="bold cyan"))

    if args.arithmetic:
        arithmetic_comparison(client)
        return 0

    console.print(Panel(question, title="question", style="blue"))

    if args.no_tools:
        no_tools_experiment(client, question)
        return 0

    _, usage, messages = answer(client, question, use_tools=True)

    console.print(
        f"\n[dim]{usage.total_tokens} tokens "
        f"({usage.prompt_tokens} in, {usage.completion_tokens} out) "
        f"in {usage.latency_s:.1f}s across {'2' if not args.no_tools else '1'} model call(s)"
        + (f", {usage.reasoning_tokens} reasoning" if usage.reasoning_tokens else "")
        + "[/dim]"
    )
    console.print(
        "[dim]Note the token cost: answering one question with tools took two round "
        "trips, and the second one re-sent everything from the first.[/dim]"
    )

    if args.transcript:
        print_transcript(messages)

    return 0


if __name__ == "__main__":
    sys.exit(main())
