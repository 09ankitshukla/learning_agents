"""Lesson 1 deliverable: a support-ticket triage extractor.

Takes messy human text and returns validated, typed data your program can act on.

    uv run lessons/01-structured-output/extract.py
    uv run lessons/01-structured-output/extract.py --file lessons/01-structured-output/sample_ticket.txt
    uv run lessons/01-structured-output/extract.py --text "printer on fire, invoice wrong too"
    uv run lessons/01-structured-output/extract.py --compare
    uv run lessons/01-structured-output/extract.py --reliability 5

Not an agent yet -- there is no loop and no tool. But this is the component an
agent is made of, and the retry-on-invalid-output pattern here is the one you
will reuse in every later lesson.
"""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llmkit import ConfigError, console, get_client
from structured import ExtractionResult, extract_structured

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# The schema. This IS the prompt -- field names, descriptions, enum values AND
# docstrings are all serialised into the JSON Schema and sent to the model.
#
# Which means: keep explanatory notes for *humans* in comments like this one,
# not in docstrings. A docstring on a model or an enum becomes prompt text, so
# commentary there wastes tokens and can actively confuse the model. Run
# `--show-schema` to see exactly what crosses the wire; it is a useful habit.
#
# Enums are the cheapest way to stop a model inventing labels. Without one, this
# ticket gets "Billing", the next gets "billing issue", and a third gets
# "Payment/Invoice Problem" -- and nothing downstream can group them. Constrain
# the output space wherever the domain allows it.
# ---------------------------------------------------------------------------
class Category(str, Enum):
    BILLING = "billing"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    ACCOUNT_ACCESS = "account_access"
    HOW_TO = "how_to"
    OTHER = "other"


class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class Sentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class Ticket(BaseModel):
    # This docstring IS sent to the model as the schema's top-level description,
    # so it is written for the model, not for you. Keep it short and factual.
    """A support ticket reduced to structured fields."""

    summary: str = Field(description="One sentence, under 20 words, describing the problem.")
    category: Category = Field(description="The single best-fitting category.")
    priority: Priority = Field(
        description=(
            "urgent = service is down or money is actively being lost; "
            "high = blocked with no workaround; "
            "normal = inconvenient but has a workaround; "
            "low = cosmetic or a nice-to-have."
        )
    )
    sentiment: Sentiment = Field(description="The customer's emotional tone.")
    customer_name: str | None = Field(
        default=None,
        description="The customer's name if stated in the text, otherwise null. Do not guess.",
    )
    affected_product: str | None = Field(
        default=None, description="Product or component named in the text, otherwise null."
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Concrete next steps for the support agent. Each an imperative phrase.",
        max_length=5,
    )
    needs_human: bool = Field(
        description="True if this requires human judgement rather than an automated reply."
    )


DEFAULT_TICKET = """
Subject: STILL waiting - third time contacting you

This is Priya Raghavan from Northwind Logistics. I have now emailed three times
about being double-charged on invoice INV-88213 for our Fleet Tracker Pro
subscription. 4,800 rupees taken twice on the 2nd. Your billing portal shows one
charge, my bank statement shows two.

Meanwhile the export-to-CSV button in the reporting dashboard has been throwing
a 500 error since your update last week, so I cannot even reconcile this myself.
I have 40 drivers depending on those reports.

If I do not hear back today I am escalating to our account manager and we will
be reviewing the contract at renewal.
""".strip()


# ---------------------------------------------------------------------------
def render(result: ExtractionResult) -> None:
    if not result.ok:
        console.print(
            Panel(
                "Extraction failed after "
                f"{result.attempts} attempt(s).\n\n" + "\n".join(result.errors),
                title="failed",
                style="red",
            )
        )
        if result.raw_replies:
            console.print(
                Panel(result.raw_replies[-1][:800], title="last raw reply", style="dim")
            )
        console.print(
            "\n[yellow]Not a dead end -- this is the lesson.[/yellow] Try --attempts 5, "
            "or a larger model. A small model failing a strict schema is exactly why "
            "the repair loop exists."
        )
        return

    ticket: Ticket = result.value
    table = Table(show_header=False, box=None)
    table.add_column("field", style="cyan", width=17)
    table.add_column("value", overflow="fold")
    table.add_row("summary", ticket.summary)
    table.add_row("category", ticket.category.value)
    table.add_row("priority", ticket.priority.value)
    table.add_row("sentiment", ticket.sentiment.value)
    table.add_row("customer", ticket.customer_name or "[dim]not stated[/dim]")
    table.add_row("product", ticket.affected_product or "[dim]not stated[/dim]")
    table.add_row("needs human", "yes" if ticket.needs_human else "no")
    table.add_row(
        "action items",
        "\n".join(f"{i}. {a}" for i, a in enumerate(ticket.action_items, 1)) or "[dim]none[/dim]",
    )
    console.print(Panel(table, title="extracted ticket", style="green"))

    note = " [yellow](required repair)[/yellow]" if result.repaired else ""
    console.print(
        f"[dim]{result.attempts} attempt(s){note} | "
        f"{result.usage.total_tokens} tokens | {result.usage.latency_s:.1f}s[/dim]"
    )
    for err in result.errors:
        console.print(f"[dim yellow]  recovered from: {err}[/dim yellow]")

    # The payoff: it is now ordinary typed Python, not text.
    if ticket.priority in (Priority.URGENT, Priority.HIGH) and ticket.needs_human:
        console.print("\n[bold red]>> would page the on-call support lead[/bold red]")
    console.print(
        "[dim]That branch is the whole point: you can write normal control flow "
        "against a model's output once it is validated.[/dim]"
    )


def compare_modes(client, text: str, attempts: int) -> None:
    """Prompt-only vs. server-enforced JSON mode."""
    console.print(Rule("Prompt-only JSON"))
    a = extract_structured(client, Ticket, text, max_attempts=attempts, verbose=True)
    console.print(
        f"  -> {'ok' if a.ok else 'FAILED'} in {a.attempts} attempt(s), "
        f"{a.usage.total_tokens} tokens"
    )

    console.print(Rule("Server-enforced json_mode"))
    try:
        b = extract_structured(
            client, Ticket, text, max_attempts=attempts, use_json_mode=True, verbose=True
        )
        console.print(
            f"  -> {'ok' if b.ok else 'FAILED'} in {b.attempts} attempt(s), "
            f"{b.usage.total_tokens} tokens"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  -> [yellow]not supported by this server: {exc}[/yellow]")
        return

    console.print(
        Panel(
            "json_mode guarantees [bold]syntactically valid[/bold] JSON. It does not "
            "guarantee [bold]correct[/bold] JSON -- wrong enum values, missing fields "
            "and invented names all pass it. So validation and repair stay mandatory "
            "either way. Use json_mode when available to remove one class of failure, "
            "not to remove the loop.",
            title="takeaway",
            style="cyan",
        )
    )


def reliability_run(client, text: str, n: int, attempts: int) -> None:
    """Run the same input N times. Consistency is a property you must measure."""
    console.print(Rule(f"Reliability: {n} identical runs at temperature 0"))
    rows: list[tuple[str, str, str, int]] = []
    first_pass = 0

    for i in range(1, n + 1):
        r = extract_structured(client, Ticket, text, max_attempts=attempts)
        if r.ok:
            t: Ticket = r.value
            rows.append((str(i), t.category.value, t.priority.value, r.attempts))
            if r.attempts == 1:
                first_pass += 1
        else:
            rows.append((str(i), "FAILED", "-", r.attempts))
        console.print(f"[dim]  run {i}/{n} done[/dim]")

    table = Table()
    table.add_column("run")
    table.add_column("category")
    table.add_column("priority")
    table.add_column("attempts")
    for row in rows:
        table.add_row(row[0], row[1], row[2], str(row[3]))
    console.print(table)

    categories = {r[1] for r in rows}
    priorities = {r[2] for r in rows}
    console.print(
        f"\nfirst-attempt success: {first_pass}/{n}\n"
        f"distinct categories: {sorted(categories)}\n"
        f"distinct priorities: {sorted(priorities)}"
    )
    console.print(
        Panel(
            "If those sets have more than one entry, your extractor is "
            "non-deterministic at temperature 0. That is normal and it is why "
            "lesson 6 tests behaviour rather than exact strings, and why lesson 7 "
            "scores across a dataset instead of eyeballing one run.",
            style="cyan",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured data from a support ticket.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", help="Ticket text to analyse.")
    src.add_argument("--file", type=Path, help="Read ticket text from a file.")
    parser.add_argument("--attempts", type=int, default=3, help="Max repair attempts (default 3).")
    parser.add_argument("--json-mode", action="store_true", help="Use server-enforced JSON mode.")
    parser.add_argument("--compare", action="store_true", help="Compare prompt-only vs json_mode.")
    parser.add_argument("--reliability", type=int, metavar="N", help="Run the same input N times.")
    parser.add_argument("--show-schema", action="store_true", help="Print the prompt sent to the model.")
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            console.print(f"[red]No such file: {args.file}[/red]")
            return 1
        text = args.file.read_text(encoding="utf-8")
    else:
        text = args.text or DEFAULT_TICKET

    if args.show_schema:
        from structured import schema_prompt

        console.print(Panel(schema_prompt(Ticket), title="system prompt", style="dim"))
        return 0

    try:
        client = get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(Panel.fit(f"{client.config.describe()}", style="bold cyan"))
    console.print(Panel(text[:600], title="input text", style="dim"))

    if args.compare:
        compare_modes(client, text, args.attempts)
    elif args.reliability:
        reliability_run(client, text, args.reliability, args.attempts)
    else:
        result = extract_structured(
            client, Ticket, text, max_attempts=args.attempts,
            use_json_mode=args.json_mode, verbose=True,
        )
        render(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
