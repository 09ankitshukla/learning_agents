"""Six ways tool calling breaks, demonstrated rather than described.

    uv run lessons/02-tool-calling/failures.py

Most tutorials show the happy path and leave you to discover the rest in
production. The gap between a demo agent and a deployable one is almost entirely
this file.

Scenarios 1-4 are deterministic: we hand-build the exact `ToolCall` a
misbehaving model would emit and push it through `dispatch()`. That is a
legitimate and underused technique -- to test how your agent handles bad model
output, you do not need a bad model, you need a fake response. It also means
these demos never flake, which matters for teaching. Lesson 6 builds this idea
into a proper test suite.

Scenarios 5-6 use the real model, because they are about its judgement.
"""

from __future__ import annotations

import sys

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llmkit import ConfigError, ToolCall, ToolSpec, console, get_client, system, user
from tools import ALL_SPECS, ToolError, calculate, dispatch

MAX_TOKENS = 800


def show(label: str, call: ToolCall) -> None:
    """Push a hand-made tool call through the dispatcher and report."""
    execution = dispatch(call)
    style = "green" if execution.ok else "yellow"
    console.print(f"[bold]{label}[/bold]")
    console.print(f"  model emitted: [cyan]{call.name}[/cyan] "
                  f"args={call.arguments if call.is_valid else call.malformed_arguments!r}")
    console.print(f"  dispatcher:    [{style}]{execution.failure_kind or 'ok'}[/{style}]")
    console.print(f"  fed back:      [dim]{execution.result}[/dim]\n")


def scenario_1_hallucinated_tool() -> None:
    console.print(Rule("1. The model invents a tool that does not exist"))
    console.print(
        "Models sometimes request a tool that sounds plausible but was never offered --\n"
        "often one they saw in training, or a name they merged from two of yours.\n"
    )
    show(
        "a tool we never published",
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Mumbai"}),
    )
    show(
        "a plausible near-miss of a real tool",
        ToolCall(id="call_2", name="calculate_expression", arguments={"expression": "2+2"}),
    )
    console.print(
        Panel(
            "The dispatcher refuses by [bold]construction[/bold], not by cleverness: reaching a "
            "function requires a key in REGISTRY, so an invented name has nowhere to go.\n\n"
            "Note what the error message does -- it lists the real tool names. That turns "
            "a dead end into a correction the model can act on, and it usually retries "
            "with the right name.",
            style="cyan",
        )
    )


def scenario_2_malformed_arguments() -> None:
    console.print(Rule("2. The arguments are not valid JSON"))
    console.print(
        "`arguments` arrives as a *string* the model generated, so it can be malformed.\n"
        "Smaller models do this often; truncation at max_tokens causes it in any model.\n"
    )
    show(
        "truncated mid-object (classic max_tokens cut-off)",
        ToolCall(id="call_3", name="calculate", malformed_arguments='{"expression": "48239 *'),
    )
    show(
        "prose where JSON belongs",
        ToolCall(id="call_4", name="calculate", malformed_arguments="expression = 2 + 2"),
    )
    console.print(
        Panel(
            "This is why `ToolCall` carries a `malformed_arguments` field instead of raising "
            "at parse time. A model writing bad JSON is a normal runtime event, so it has to "
            "be representable in your data model -- if the only way to express it is an "
            "exception, your agent cannot recover from it.\n\n"
            "Check `finish_reason` when you see this in the wild: truncation is a budget "
            "problem, not a model problem, and no retry will fix it.",
            style="cyan",
        )
    )


def scenario_3_wrong_arguments() -> None:
    console.print(Rule("3. Valid JSON, wrong contents"))
    console.print(
        "json_mode and schemas constrain shape, never meaning. All of these parse fine.\n"
    )
    show(
        "required argument missing",
        ToolCall(id="call_5", name="convert_currency",
                 arguments={"amount": 100, "from_currency": "USD"}),
    )
    show(
        "argument the tool does not accept",
        ToolCall(id="call_6", name="get_current_time",
                 arguments={"timezone": "UTC", "format": "12-hour"}),
    )
    show(
        "plausible value the tool rejects",
        ToolCall(id="call_7", name="convert_currency",
                 arguments={"amount": 100, "from_currency": "USD", "to_currency": "BTC"}),
    )
    show(
        "a city name where an IANA timezone belongs",
        ToolCall(id="call_8", name="get_current_time", arguments={"timezone": "Mumbai"}),
    )
    console.print(
        Panel(
            "The last two are the interesting ones. 'BTC' and 'Mumbai' are both entirely "
            "reasonable guesses -- the model is not being stupid, your schema simply did not "
            "say what it wanted.\n\n"
            "Two fixes, and prefer the first: [bold]make the schema more specific[/bold] (an enum of "
            "supported currencies; an example IANA name in the description), and make the "
            "error message name the valid options so the model can self-correct on the next "
            "turn. Both are cheaper than a smarter model.",
            style="cyan",
        )
    )


def scenario_4_hostile_input() -> None:
    console.print(Rule("4. The tool input is actively dangerous"))
    console.print(
        "The natural way to write a calculator tool is `eval(expression)`. Here is why\n"
        "that is a remote code execution vulnerability, not a shortcut.\n"
    )

    attacks = [
        ("import and shell out", "__import__('os').system('echo pwned')"),
        ("read a file", "open('.env').read()"),
        ("reach into builtins", "().__class__.__bases__[0].__subclasses__()"),
        ("resource exhaustion", "9**9**9"),
    ]

    table = Table(show_lines=True)
    table.add_column("attack", width=22)
    table.add_column("expression", overflow="fold")
    table.add_column("our calculator says", overflow="fold")
    for label, expression in attacks:
        try:
            outcome = calculate(expression)
        except ToolError as exc:
            outcome = f"[green]refused: {exc}[/green]"
        table.add_row(label, expression, outcome)
    console.print(table)

    console.print(
        Panel(
            "With `eval()`, rows 1-3 execute and row 4 hangs the process. The AST walker "
            "refuses all four.\n\n"
            "The reason it holds up is that it is an [bold]allowlist[/bold]: it enumerates the node "
            "types it permits and rejects everything else, so it blocks attribute access and "
            "imports without needing to know those specific attacks exist. A blocklist of "
            "dangerous strings is a game you lose to the next encoding trick.\n\n"
            "Also note rows 2 and 4 are different problems. Row 2 is code execution; row 4 is "
            "resource exhaustion, which needs a separate limit. Fixing one does not fix the "
            "other.\n\n"
            "Worth sitting with: nothing here required a malicious *user*. A model that reads "
            "a poisoned web page or document can be talked into emitting exactly these "
            "strings. That is prompt injection, and it is lesson 11.",
            title="why the allowlist",
            style="red",
        )
    )


def scenario_5_no_tool_when_needed(client) -> None:
    console.print(Rule("5. The model decides whether to use your tool -- and may decline"))
    console.print(
        "A tool the model does not choose is worth nothing. With tool_choice='auto' the\n"
        "decision is the model's, and it is based on its own confidence.\n"
    )

    vague = ToolSpec(
        name="calculate",
        description="Does math.",  # deliberately terrible
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )

    hard = "What is 91273 times 4482?"
    trivial = "What is 2 + 2?"

    rows: list[tuple[str, str, bool]] = []
    for label, question, specs, prompt in (
        ("hard sum, vague description", hard, [vague], "You are helpful."),
        ("hard sum, good description", hard, ALL_SPECS, "You are helpful."),
        ("hard sum, good + explicit instruction", hard, ALL_SPECS,
         "You are helpful. Always use the calculate tool for arithmetic."),
        ("TRIVIAL sum, good description", trivial, ALL_SPECS, "You are helpful."),
    ):
        reply = client.chat([system(prompt), user(question)], tools=specs, max_tokens=MAX_TOKENS)
        used = reply.wants_tools
        detail = (
            f"[green]used {reply.tool_calls[0].name}[/green]"
            if used
            else f"[yellow]answered directly: {(reply.text or '').strip()[:40]}[/yellow]"
        )
        rows.append((label, detail, used))
        console.print(f"  {label:<38} {detail}")

    # Now remove the choice entirely.
    forced = client.chat(
        [system("You are helpful."), user(trivial)],
        tools=ALL_SPECS,
        max_tokens=MAX_TOKENS,
        tool_choice="required",
    )
    forced_detail = (
        f"[green]used {forced.tool_calls[0].name}"
        f"({forced.tool_calls[0].arguments})[/green]"
        if forced.wants_tools
        else "[red]still answered directly[/red]"
    )
    console.print(f"  {'TRIVIAL sum, tool_choice=required':<38} {forced_detail}")

    hard_used = sum(1 for _, _, used in rows[:3] if used)
    trivial_used = rows[3][2]

    if hard_used == 3 and not trivial_used:
        verdict = (
            "This model used the tool for the hard sum in all three variants -- even with "
            "the description 'Does math.' -- and skipped it for 2 + 2.\n\n"
            "That is not the tool description winning or losing. It is the model judging "
            "[bold]necessity[/bold]: it is confident about 2 + 2 and unsure about a five-digit "
            "multiplication, so it delegates only the second. Sensible, and completely "
            "outside your control while tool_choice is 'auto'."
        )
    elif hard_used < 3:
        verdict = (
            f"Only {hard_used} of 3 variants used the tool for the hard sum, and the vague "
            "description is the likely culprit. Tool descriptions are prompt engineering, "
            "and they are the highest-leverage text in an agent."
        )
    else:
        verdict = (
            "This model reached for the tool every time, including for 2 + 2 -- an extra "
            "round trip to compute something it certainly knows.\n\n"
            "Before calling that overcaution, reread our own description in tools.py: "
            "[italic]'Use this for any calculation rather than working it out yourself.'[/italic] The "
            "model is following it exactly. The behaviour you are seeing is the "
            "description working as written, not the model misjudging.\n\n"
            "That is the lesson in miniature: when an agent behaves oddly, suspect the text "
            "you wrote before you suspect the model. If you wanted trivial arithmetic "
            "handled inline, the fix is a description that says so -- 'use for "
            "multi-digit arithmetic where exactness matters' -- not a different model."
        )

    console.print(
        Panel(
            verdict + "\n\n"
            "[bold]tool_choice is the override.[/bold] 'auto' lets the model decide; 'required' removes "
            "the option and guarantees a tool call. Use 'required' when a tool call is the "
            "only acceptable outcome -- structured extraction, or a lookup you must not let "
            "the model fake from memory.\n\n"
            "And note the honest caveat: a capable model masks a bad tool description. Run "
            "this against openai/gpt-oss-20b or a local 3B and the vague variant starts "
            "failing. Which is the real lesson -- tool selection is probabilistic, so it "
            "belongs in a measured dataset (lesson 7), not in a single run you got lucky on.",
            style="cyan",
        )
    )


def scenario_6_recovery(client) -> None:
    console.print(Rule("6. Recovery: a failed tool call the model fixes itself"))
    console.print(
        "The payoff for returning errors as observations instead of raising.\n"
    )

    from llmkit import tool_result

    messages = [
        system("You are precise. Use tools for facts you cannot know."),
        user("What time is it in Mumbai?"),
    ]

    first = client.chat(messages, tools=ALL_SPECS, max_tokens=MAX_TOKENS)
    if not first.wants_tools:
        console.print("[yellow]Model did not request a tool; skipping this scenario.[/yellow]")
        return

    call = first.tool_calls[0]
    messages.append(first.as_message())
    console.print(f"  attempt 1: [cyan]{call.name}({call.arguments})[/cyan]")

    # Force the failure so the demo is reproducible regardless of what the model
    # guessed: pretend it passed a city name rather than an IANA zone.
    forced = ToolCall(id=call.id, name="get_current_time", arguments={"timezone": "Mumbai"})
    execution = dispatch(forced)
    console.print(f"  dispatcher: [yellow]{execution.failure_kind}[/yellow]")
    console.print(f"  fed back:   [dim]{execution.result}[/dim]")

    messages.append(tool_result(call.id, execution.result))

    second = client.chat(messages, tools=ALL_SPECS, max_tokens=MAX_TOKENS)
    if second.wants_tools:
        retry = second.tool_calls[0]
        console.print(f"  attempt 2: [green]{retry.name}({retry.arguments})[/green]")
        retried = dispatch(retry)
        console.print(f"  result:    [green]{retried.result}[/green]")
        console.print(
            Panel(
                "The model read the error, learned that IANA names are required, and "
                "corrected itself without any code from you telling it how.\n\n"
                "That is the entire argument for treating tool failures as data. Had "
                "`dispatch()` raised, the process would have died on a mistake the model "
                "could fix in one turn.\n\n"
                "It also shows why error text is prompt text: 'Error: invalid timezone' "
                "would not have been enough. 'Use an IANA name such as Asia/Kolkata' was.",
                title="self-correction",
                style="green",
            )
        )
    else:
        console.print(f"  model replied with text: [dim]{(second.text or '')[:150]}[/dim]")
        console.print(
            "[yellow]No retry this time. Non-deterministic -- run it again. The pattern "
            "holds on average, which is exactly why lesson 7 measures instead of "
            "eyeballing.[/yellow]"
        )


def main() -> int:
    console.print(
        Panel.fit(
            "Six failure modes of tool calling.\n"
            "Scenarios 1-4 are deterministic (hand-built tool calls).\n"
            "Scenarios 5-6 use the live model and may vary between runs.",
            style="bold cyan",
        )
    )

    scenario_1_hallucinated_tool()
    scenario_2_malformed_arguments()
    scenario_3_wrong_arguments()
    scenario_4_hostile_input()

    try:
        client = get_client()
    except ConfigError as exc:
        console.print(f"\n[yellow]Skipping live scenarios: {exc}[/yellow]")
        return 0

    scenario_5_no_tool_when_needed(client)
    scenario_6_recovery(client)

    console.print(Rule("Summary"))
    table = Table()
    table.add_column("failure", style="cyan")
    table.add_column("defence")
    for failure, defence in [
        ("invented tool name", "registry lookup; error names the real tools"),
        ("malformed JSON arguments", "represent it in the data model, feed the error back"),
        ("missing / extra arguments", "check required up front; catch TypeError"),
        ("plausible but invalid value", "tighter schema + error listing valid options"),
        ("dangerous tool input", "allowlist parsing, never eval; separate resource limits"),
        ("tool not used at all", "better description and explicit system-prompt instruction"),
        ("tool raised", "catch, convert to an observation, let the model retry"),
    ]:
        table.add_row(failure, defence)
    console.print(table)
    console.print(
        "\nEvery defence returns a [bold]string the model can read[/bold]. That is the "
        "shape to remember: a tool failure is a turn in a conversation, not an exception."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
