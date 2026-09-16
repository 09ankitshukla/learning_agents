"""Your first model calls -- four experiments in the mechanics.

    uv run lessons/00-setup/hello_model.py

Nothing here is an agent yet. The goal is to make four things concrete, because
every later lesson assumes you have felt them:

  1. A model call is stateless. You resend the entire conversation every time.
  2. Temperature controls randomness, and it is the reason agents need tests.
  3. Tokens are the unit of cost, latency, and memory limits.
  4. The system prompt steers behaviour more cheaply than anything else.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.rule import Rule

from llmkit import Usage, assistant, console, get_client, system, user


def experiment_1_single_call(client) -> Usage:
    """The smallest possible interaction."""
    console.print(Rule("1. One call, one reply"))

    messages = [
        system("You are a concise technical writer. Answer in one sentence."),
        user("What is a language model?"),
    ]
    reply = client.chat(messages)

    console.print(Panel(reply.text or "(empty)", title="response", style="green"))
    console.print(
        f"[dim]prompt {reply.usage.prompt_tokens} tok | "
        f"output {reply.usage.completion_tokens} tok | "
        f"{reply.usage.latency_s:.1f}s | "
        f"finish_reason={reply.finish_reason}[/dim]"
    )
    console.print(
        "\n[bold]Note[/bold] finish_reason. 'stop' means the model chose to end. "
        "'length' means it was cut off by max_tokens -- a truncated tool call is a "
        "common and confusing agent bug.\n"
    )
    return reply.usage


def experiment_2_statelessness(client) -> Usage:
    """Show that memory is something *you* maintain, not the server."""
    console.print(Rule("2. The model has no memory"))

    # First turn.
    first = client.chat([user("My name is Ankit. Remember it.")])
    console.print(f"[cyan]turn 1[/cyan] {(first.text or '').strip()[:150]}")

    # Ask again with NO history. The model cannot know.
    amnesia = client.chat([user("What is my name?")])
    console.print(f"[red]without history[/red] {(amnesia.text or '').strip()[:150]}")

    # Ask again WITH history. Now it can.
    with_history = client.chat(
        [
            user("My name is Ankit. Remember it."),
            assistant(first.text),
            user("What is my name?"),
        ]
    )
    console.print(f"[green]with history[/green] {(with_history.text or '').strip()[:150]}")

    console.print(
        "\n[bold]This is the single most important mechanic in the course.[/bold]\n"
        "There is no session on the server. 'Memory' is you appending to a list and\n"
        "resending it. That is why long conversations get slower and more expensive,\n"
        "and why lesson 4 is entirely about managing that list.\n"
    )
    return first.usage + amnesia.usage + with_history.usage


def experiment_3_temperature(client) -> Usage:
    """Same prompt, different randomness."""
    console.print(Rule("3. Temperature and reproducibility"))

    prompt = [user("Invent a name for a coffee shop. Reply with the name only.")]
    total = Usage()

    # max_tokens=300 for a one-word answer looks absurd until you remember
    # experiment 5: on a reasoning model most of this budget is spent thinking,
    # and a tight cap returns an empty string instead of a short name.
    for temp in (0.0, 0.0, 1.2, 1.2):
        reply = client.chat(prompt, temperature=temp, max_tokens=300)
        total = total + reply.usage
        colour = "green" if temp == 0.0 else "magenta"
        console.print(f"[{colour}]temp={temp:<4}[/{colour}] {(reply.text or '').strip()[:80]}")

    console.print(
        "\nLow temperature repeats itself; high temperature explores. For agents that\n"
        "must call tools correctly you want low. For drafting copy you want higher.\n"
        "[bold]Caveat:[/bold] temperature 0 is not a guarantee of identical output --\n"
        "batching changes floating-point order on the server. Never write a test that\n"
        "asserts exact model text. Lesson 6 shows what to assert instead.\n"
    )
    return total


def experiment_4_system_prompt(client) -> Usage:
    """The cheapest steering mechanism you have."""
    console.print(Rule("4. The system prompt is the control surface"))

    question = user("Should I use a database index here?")
    personas = {
        "terse expert": "Answer in under 15 words. No preamble.",
        "socratic": "Never answer directly. Reply with one clarifying question.",
        "strict JSON": 'Reply only with JSON: {"answer": "...", "confidence": 0.0}',
    }

    total = Usage()
    for label, instruction in personas.items():
        reply = client.chat([system(instruction), question], max_tokens=500)
        total = total + reply.usage
        body = (reply.text or "").strip() or "[red](empty -- see experiment 5)[/red]"
        console.print(Panel(body, title=label, style="blue"))

    console.print(
        "Same model, same question, three behaviours. Before reaching for a bigger\n"
        "model or a framework, rewrite the system prompt -- it is free and immediate.\n"
        "The third persona is a preview of lesson 1: asking for JSON is easy, and\n"
        "*reliably getting valid JSON* is the actual problem.\n"
    )
    return total


def experiment_5_reasoning(client) -> Usage:
    """Reasoning models, and the token budget trap they create.

    Modern open models (GPT-OSS, DeepSeek-R1, Qwen3 in thinking mode) generate
    hidden deliberation before their visible answer. You pay for those tokens at
    the output rate, they consume your max_tokens budget, and they are invisible
    in `text`. Skip this and you will eventually spend an hour debugging an
    "empty response" that is really a budget problem.
    """
    console.print(Rule("5. Reasoning tokens and the budget trap"))

    total = Usage()
    probe = [user("A ticket says: 'charged twice, and CSV export is broken'. "
                  "Best single category: billing, bug, or other? One word.")]

    first = client.chat(probe, max_tokens=800)
    total = total + first.usage

    if not first.usage.reasoning_tokens:
        console.print(
            "This model does not report reasoning tokens, so it is probably not a\n"
            "reasoning model. Read on anyway -- you will meet one soon, and the\n"
            "failure mode below is worth recognising.\n"
        )
        console.print(f"answer: {(first.text or '').strip()[:120]}")
        return total

    share = first.usage.reasoning_tokens / max(1, first.usage.completion_tokens)
    console.print(
        f"answer: [green]{(first.text or '').strip()[:60]}[/green]\n"
        f"output tokens: {first.usage.completion_tokens} total, "
        f"[yellow]{first.usage.reasoning_tokens} hidden reasoning ({share:.0%})[/yellow], "
        f"{first.usage.answer_tokens} visible"
    )
    if first.reasoning:
        console.print(
            Panel(first.reasoning.strip()[:400], title="the hidden reasoning", style="dim")
        )

    # The trap: a budget that looks generous for a one-word answer, but is not.
    console.print("\n[bold]Same question, max_tokens=40:[/bold]")
    starved = client.chat(probe, max_tokens=40)
    total = total + starved.usage
    console.print(
        f"  text: {starved.text!r}  finish_reason={starved.finish_reason}  "
        f"reasoning_tokens={starved.usage.reasoning_tokens}"
    )
    console.print(
        "  [red]Empty. Not a refusal, not a failure -- the model spent the whole "
        "budget thinking.[/red]\n"
        "  You were still billed. `response.starved` exists to detect exactly this."
    )

    # The dial: effort changes the answer, not just the verbosity.
    console.print("\n[bold]Same question at three reasoning efforts:[/bold]")
    for effort in ("low", "medium", "high"):
        try:
            r = client.chat(probe, max_tokens=800, reasoning_effort=effort)
            total = total + r.usage
            console.print(
                f"  effort={effort:<7} answer=[cyan]{(r.text or '(empty)').strip()[:30]:<30}[/cyan] "
                f"reasoning_tokens={r.usage.reasoning_tokens:<4} {r.usage.latency_s:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"  effort={effort}: not supported ({type(exc).__name__})")

    console.print(
        "\nIf the answers differ across efforts, sit with that for a second: the\n"
        "quality dial [bold]changes the answer[/bold], it does not just pad it. That makes\n"
        "reasoning effort a variable you must hold fixed when comparing anything else,\n"
        "and a thing worth A/B testing deliberately in lesson 9.\n"
    )
    return total


def main() -> None:
    client = get_client()
    console.print(Panel.fit(f"Talking to {client.config.describe()}", style="bold cyan"))
    if client.config.is_local:
        console.print("[dim]Local CPU inference -- expect pauses between sections.[/dim]\n")

    total = Usage()
    for experiment in (
        experiment_1_single_call,
        experiment_2_statelessness,
        experiment_3_temperature,
        experiment_4_system_prompt,
        experiment_5_reasoning,
    ):
        total = total + experiment(client)

    console.print(Rule("Totals"))
    line = (
        f"{total.total_tokens} tokens across all calls "
        f"({total.prompt_tokens} in, {total.completion_tokens} out"
    )
    if total.reasoning_tokens:
        line += f", of which {total.reasoning_tokens} hidden reasoning"
    console.print(line + f") in {total.latency_s:.1f}s of model time.")

    console.print(
        "\n[dim]That was one script. An agent loops, and every iteration resends the "
        "whole history, so token use grows quadratically with conversation length. On a "
        "free tier that shows up as rate limits; on a paid one, as a bill. Lesson 8 "
        "measures it properly.[/dim]"
    )
    console.print("\n[bold green]Lesson 0 complete.[/bold green] Read NOTES.md, then go to lesson 01.")


if __name__ == "__main__":
    main()
