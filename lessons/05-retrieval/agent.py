"""Lesson 5 deliverable: an agent that searches your notes by meaning.

    uv run lessons/05-retrieval/agent.py --build          # index the notes
    uv run lessons/05-retrieval/agent.py --search "why is my answer empty"
    uv run lessons/05-retrieval/agent.py --measure        # 3 methods, 10 labelled queries
    uv run lessons/05-retrieval/agent.py --chunking       # does the heading matter?
    uv run lessons/05-retrieval/agent.py --ask "..."      # full agent, both search tools
    uv run lessons/05-retrieval/agent.py --stuffing       # retrieval vs context stuffing

Lesson 3 gave the agent `search_files`, which matches literal words. This lesson
adds `search_notes`, which matches meaning, and then measures whether that
actually helps rather than assuming it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llmkit import ConfigError, ToolSpec, console, get_client, system, user

_LESSON_03 = Path(__file__).resolve().parents[1] / "03-agent-loop"
_LESSON_04 = Path(__file__).resolve().parents[1] / "04-memory-context"
for _path in (_LESSON_03, _LESSON_04):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from loop import run_agent  # noqa: E402
from toolset import build_registry  # noqa: E402

# Lesson 4's context manager, used unchanged. Not decoration: without it the
# agent below exceeds Groq's 8,000 tokens-per-minute ceiling and the provider
# returns HTTP 413. Retrieval results are large, seven tool schemas are re-sent
# every call, and it accumulates exactly as lesson 4 measured.
from context import ContextManager  # noqa: E402

from chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS, build_corpus  # noqa: E402
from queries import QUERY_SET  # noqa: E402
from store import DEFAULT_MODEL, NoteStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "lessons" / "05-retrieval" / ".cache" / "notes_index"


# ---------------------------------------------------------------------------
def load_store(verbose: bool = True, model: str = DEFAULT_MODEL) -> NoteStore:
    corpus = build_corpus(REPO_ROOT)
    store = NoteStore(corpus, model_name=model)
    if verbose:
        stats = corpus.stats()
        console.print(
            f"[dim]corpus: {stats['chunks']} chunks from {stats['files']} files, "
            f"mean {stats['mean_chars']:.0f} chars "
            f"(min {stats['min_chars']:.0f}, max {stats['max_chars']:.0f})[/dim]"
        )
    store.build(cache_path=CACHE, verbose=verbose)
    if verbose:
        console.print(f"[dim]vectors: {len(corpus)} x {store.dimensions}d[/dim]")
    return store


def show_hits(hits, method: str) -> None:
    if not hits:
        console.print(f"  [yellow]{method}: no results[/yellow]")
        return
    for rank, hit in enumerate(hits, start=1):
        extra = ""
        if hit.method == "hybrid":
            extra = f" [dim](sem {hit.semantic_score:.3f} / kw {hit.keyword_score:.1f})[/dim]"
        console.print(f"  {rank}. [cyan]{hit.cite()}[/cyan]  {hit.score:.3f}{extra}")
        console.print(f"     [dim]{hit.chunk.text[:120].replace(chr(10), ' ')}[/dim]")


# ---------------------------------------------------------------------------
def experiment_search(store: NoteStore, query: str, top_k: int) -> None:
    console.print(Rule(f"Query: {query}"))
    for method in ("keyword", "semantic", "hybrid"):
        console.print(f"\n[bold]{method}[/bold]")
        show_hits(store.search(query, top_k=top_k, method=method), method)

    console.print(
        Panel(
            "Note how compressed the semantic scores are -- typically 0.6 to 0.8 even "
            "for a bad match. [bold]The absolute number is close to meaningless[/bold]; only the "
            "ranking and the gap between first and second place carry information.\n\n"
            "That matters practically: a similarity threshold like 'only use hits above "
            "0.75' looks principled and is almost impossible to tune, because the band "
            "shifts with the model and the corpus. Prefer top-k plus a check on the gap.",
            style="cyan",
        )
    )


def experiment_measure(store: NoteStore, top_k: int) -> None:
    """The heart of the lesson: is semantic search actually better?"""
    console.print(Rule("Measured: three methods against 10 labelled queries"))
    console.print(
        "[dim]Every query is phrased to avoid the target note's own vocabulary. "
        "If a question reuses the document's words, keyword search already works "
        "and embeddings add nothing.[/dim]\n"
    )

    table = Table(show_lines=False)
    table.add_column("query", width=42, overflow="fold")
    table.add_column("keyword", width=8)
    table.add_column("semantic", width=9)
    table.add_column("hybrid", width=7)

    METHODS = ("keyword", "semantic", "hybrid")
    top1 = {m: 0 for m in METHODS}
    # Recall@k: was the right material anywhere in the results? Reported because
    # top-1 alone cannot distinguish "never found it" from "found it, ranked 2nd",
    # and for an agent those are very different -- it reads all k results.
    recall = {m: 0 for m in METHODS}
    failures: list[tuple[str, str, str]] = []

    for labelled in QUERY_SET:
        cells = {}
        for method in METHODS:
            hits = store.search(labelled.question, top_k=top_k, method=method)
            # Match on the full label (path + headings), not the path alone.
            hit_at_1 = bool(hits) and labelled.is_correct(hits[0].chunk.label)
            hit_at_k = any(labelled.is_correct(h.chunk.label) for h in hits)
            top1[method] += hit_at_1
            recall[method] += hit_at_k

            if hit_at_1:
                cells[method] = "[green]1st[/green]"
            elif hit_at_k:
                position = next(
                    i for i, h in enumerate(hits, 1) if labelled.is_correct(h.chunk.label)
                )
                cells[method] = f"[yellow]{position}th[/yellow]"
            else:
                cells[method] = "[red]miss[/red]"

            if method == "semantic" and not hit_at_k:
                got = hits[0].chunk.label if hits else "(nothing)"
                failures.append((labelled.question, labelled.expected_source, got))

        table.add_row(labelled.question, cells["keyword"], cells["semantic"], cells["hybrid"])

    console.print(table)

    n = len(QUERY_SET)
    scores = Table()
    scores.add_column("metric", width=22)
    for method in METHODS:
        scores.add_column(method, width=10, justify="right")
    scores.add_row("top-1 accuracy", *[f"{top1[m]}/{n}" for m in METHODS])
    scores.add_row(f"recall@{top_k}", *[f"{recall[m]}/{n}" for m in METHODS])
    console.print(scores)

    console.print(
        f"\n[dim]recall@{top_k} is the metric that matters for an agent: it reads all "
        f"{top_k} results, so a correct chunk ranked 3rd is still useful. top-1 matters "
        f"when you show a single answer to a user.[/dim]"
    )

    if failures:
        console.print("\n[yellow]semantic misses (not in top-k at all):[/yellow]")
        for question, expected, got in failures:
            console.print(f"  [dim]{question}[/dim]")
            console.print(f"    wanted {expected}, top hit was [red]{got}[/red]")

    console.print(
        Panel(
            "Two things worth taking seriously here.\n\n"
            "**Semantic search wins, but it is not magic.** It answers questions that "
            "share no words with their answer, which keyword search structurally "
            "cannot. It also misses. A retrieval system that is right most of the time "
            "still needs an agent that can recover when it is wrong -- which is why "
            "`search_notes` sits alongside `read_file` rather than replacing it.\n\n"
            "**Hybrid is not automatically better.** If it scored below semantic here, "
            "that is the naive min-max blend being unstable on a small corpus: one "
            "strong keyword match stretches the scale and drags an unrelated chunk "
            "above the correct semantic hit. Rank-based fusion fixes it, and it is an "
            "exercise in the README. The transferable lesson is that 'combine both "
            "signals' is not free -- combining badly is worse than not combining.",
            style="cyan",
        )
    )


def experiment_chunking(top_k: int) -> None:
    """Does embedding the heading with the text actually matter? Measure it."""
    console.print(Rule("Chunking: does the heading earn its tokens?"))
    console.print(
        "[dim]Same corpus, same model, same queries. The only difference is whether "
        "the heading path is embedded alongside the chunk body.[/dim]\n"
    )

    corpus = build_corpus(REPO_ROOT)

    results = []
    for label, include_heading in (("body only", False), ("heading + body", True)):
        store = NoteStore(corpus)
        texts = [
            (c.embedding_text if include_heading else c.text) for c in corpus.chunks
        ]
        console.print(f"  embedding {len(texts)} chunks ({label})...")
        store._vectors = store.embed_texts(texts)  # noqa: SLF001 - deliberate for the experiment

        # Match on .label (source + heading path), NOT .source.
        #
        # This line had the path-only bug that experiment_measure already had and
        # that was already fixed there -- the same mistake, made twice, producing
        # a result that contradicted the other experiment. That contradiction is
        # the only reason it was caught. Two measurements of the same thing
        # disagreeing is a gift; one measurement quietly wrong is not.
        top1 = 0
        recall_k = 0
        for q in QUERY_SET:
            results_for_q = store.search_semantic(q.question, top_k)
            if results_for_q and q.is_correct(results_for_q[0].chunk.label):
                top1 += 1
            if any(q.is_correct(h.chunk.label) for h in results_for_q):
                recall_k += 1
        results.append((label, top1, recall_k))

    n = len(QUERY_SET)
    table = Table()
    table.add_column("embedded text", width=18)
    table.add_column("top-1", width=10, justify="right")
    table.add_column(f"recall@{top_k}", width=12, justify="right")
    for label, top1, recall_k in results:
        table.add_row(label, f"{top1}/{n}", f"{recall_k}/{n}")
    console.print(table)

    delta = results[1][1] - results[0][1]
    console.print(
        Panel(
            f"Including the heading changed top-1 by {delta:+d} queries, and recall@"
            f"{top_k} by {results[1][2] - results[0][2]:+d}.\n\n"
            "[bold]This experiment exists because it refuted its own hypothesis.[/bold]\n\n"
            "The theory is sound: a chunk lifted from mid-document has lost its context, "
            "so prefixing the heading should supply the vocabulary a human question uses. "
            "Tested on a hand-written sample of 8 chunks it worked well -- top-1 went "
            "from 4/7 to 6/7, beating what a model with twice the dimensions achieved.\n\n"
            "On the real 193-chunk corpus it did not replicate. Two reasons worth "
            "knowing: the small sample had no competing documents and uniformly "
            "informative headings, whereas at scale many headings are generic ('The one "
            "idea', 'Carry forward') and add noise. And a 1-query difference on a "
            "10-query set is not a difference at all.\n\n"
            "So the transferable lessons are that [bold]a result from a tiny hand-picked "
            "sample is a hypothesis, not a finding[/bold], and that this eval set is too small "
            "to resolve small effects. Both point at lesson 7.\n\n"
            "The heading is still embedded, but for citation quality rather than "
            "ranking -- and saying so plainly is the point.",
            style="cyan",
        )
    )


def experiment_stuffing(client, store: NoteStore) -> None:
    """Retrieval versus putting everything in the prompt. The lesson-4 connection."""
    console.print(Rule("Retrieval vs context stuffing"))

    corpus = store.corpus
    all_chars = sum(len(c.text) for c in corpus.chunks)
    stuffed_tokens = all_chars // 4

    question = "Why might my agent return an empty answer, and what should I change?"
    hits = store.search_semantic(question, top_k=3)
    retrieved_chars = sum(len(h.chunk.text) for h in hits)

    table = Table()
    table.add_column("approach", width=26)
    table.add_column("chars sent", justify="right", width=12)
    table.add_column("est. tokens", justify="right", width=12)
    table.add_column("", overflow="fold")
    table.add_row(
        "stuff the whole corpus",
        f"{all_chars:,}",
        f"~{stuffed_tokens:,}",
        "every note, every call",
    )
    table.add_row(
        "retrieve top 3 chunks",
        f"{retrieved_chars:,}",
        f"~{retrieved_chars // 4:,}",
        "only what the question needs",
    )
    console.print(table)
    console.print(
        f"\nratio: retrieval sends [bold]{retrieved_chars / max(1, all_chars):.1%}[/bold] "
        f"of the corpus"
    )

    console.print("\n[bold]Answering from the retrieved chunks only:[/bold]")
    context = "\n\n---\n\n".join(f"[{h.cite()}]\n{h.chunk.text}" for h in hits)
    reply = client.chat(
        [
            system(
                "Answer using only the provided notes. Cite the source label for each "
                "claim. If the notes do not contain the answer, say so."
            ),
            user(f"Notes:\n\n{context}\n\nQuestion: {question}"),
        ],
        max_tokens=800,
    )
    console.print(Panel(reply.text or "(empty)", style="green"))
    console.print(
        f"[dim]{reply.usage.prompt_tokens} input tokens for that call[/dim]"
    )

    console.print(
        Panel(
            "This is the direct answer to lesson 4's problem. Lesson 4 was about "
            "salvaging a context window that had already been filled; retrieval is "
            "about not filling it in the first place.\n\n"
            "It is also a real trade, not a pure win. Stuffing cannot miss -- the "
            "answer is definitely in there somewhere. Retrieval can miss, and then the "
            "model answers confidently from the wrong three chunks. You are buying a "
            "large token saving with a small chance of retrieving nothing useful, which "
            "is why the measured accuracy above matters and why the system prompt "
            "explicitly permits 'the notes do not contain the answer'.",
            style="cyan",
        )
    )


# ---------------------------------------------------------------------------
def make_search_notes_tool(store: NoteStore, top_k: int = 4):
    """Wrap retrieval as a tool the agent can choose to call."""

    # Each hit is truncated to an excerpt rather than returned whole. This is
    # standard RAG practice and it was also forced by a real failure: returning
    # four full 1,200-character chunks produced a ~6,200-token tool result, which
    # lesson 4's compactor then crushed to 765 tokens -- destroying the content the
    # search had just found. The agent could not answer and re-searched five times
    # until it hit the step cap.
    #
    # Bounding the result at source is better than compressing it after the fact.
    # Retrieval's job is to point at the right place; `read_file` gets the detail.
    EXCERPT_CHARS = 450

    def search_notes(query: str, top_k_override: int | None = None) -> str:
        limit = max(1, min(int(top_k_override or top_k), 8))
        hits = store.search_semantic(query, top_k=limit)
        if not hits:
            return f"No notes matched {query!r}."

        parts = [f"{len(hits)} relevant note section(s) for {query!r}:"]
        for rank, hit in enumerate(hits, start=1):
            text = hit.chunk.text
            if len(text) > EXCERPT_CHARS:
                text = text[:EXCERPT_CHARS] + " ...[excerpt truncated]"
            parts.append(
                f"\n[{rank}] {hit.cite()}  (similarity {hit.score:.3f})\n{text}"
            )
        parts.append(
            "\n\nThese are excerpts ranked by meaning, not exact matches. If one looks "
            "right but is cut short, call read_file on its source path for the full "
            "text. If none answers the question, say so rather than searching again "
            "with a similar query."
        )
        return "\n".join(parts)

    spec = ToolSpec(
        name="search_notes",
        description=(
            "Search this project's lesson notes and documentation by MEANING rather "
            "than exact words, and return the most relevant sections with their source "
            "file and heading. Use this when you want to find where a topic is "
            "explained but do not know the exact wording used. Prefer this over "
            "search_files for conceptual questions; prefer search_files when you need "
            "a literal string such as a function name."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What you are looking for, phrased as a question or topic. "
                        "Natural language works better here than keywords."
                    ),
                },
                "top_k_override": {
                    "type": "integer",
                    "description": "How many sections to return (default 4, max 8).",
                },
            },
            "required": ["query"],
        },
    )
    return spec, search_notes


def experiment_ask(client, store: NoteStore, question: str, max_steps: int) -> None:
    """The deliverable: an agent with both search tools, choosing between them."""
    console.print(Rule("Agent with semantic + keyword search"))

    registry = build_registry()  # lesson 3's six tools, including search_files
    spec, fn = make_search_notes_tool(store)
    registry.add(spec, fn)

    console.print(
        f"[dim]{len(registry)} tools available: {', '.join(registry.names)}[/dim]"
    )
    console.print(Panel(question, title="question", style="blue"))

    # Lesson 4's compactor, and it is load-bearing here rather than a nicety.
    # Retrieval returns several 1,200-character chunks per call, seven tool schemas
    # are re-sent every step, and history accumulates. Without this the request
    # crossed Groq's 8,000 tokens-per-minute ceiling and came back as HTTP 413.
    #
    # Budget 4,000 leaves room for the schemas and the reply inside that ceiling.
    #
    # tool_result_max_chars is raised well above lesson 4's 600-character default
    # on purpose. Retrieval results are information-dense: compressing them to 600
    # characters deleted the excerpts the search had just found, and the agent
    # looped re-searching for what it had already retrieved. Compression thresholds
    # are not universal -- they depend on how much of a tool's output is signal.
    manager = ContextManager(
        budget=4000,
        strategy="auto",
        client=client,
        tool_result_max_chars=1800,
        verbose=True,
    )

    trajectory = run_agent(
        client,
        question,
        registry,
        max_steps=max_steps,
        compactor=manager,
        # The "at most two searches" instruction is not padding. Without it, when
        # the corpus genuinely lacks an answer the agent rephrases the query and
        # searches again, indefinitely, until it hits the step cap. Stall detection
        # does not catch it because each query differs slightly.
        #
        # This is the retrieval-specific failure mode: an agent with a search tool
        # will keep searching rather than concluding the information is not there.
        # Telling it when to stop is your job, not the model's.
        system_prompt=(
            "You answer questions about this project using its notes.\n"
            "Search for relevant notes, then answer from what you find and cite the "
            "source file and heading for each claim.\n"
            "Use at most two searches. After that, answer with what you have.\n"
            "If the notes genuinely do not cover the question, say so plainly and "
            "stop -- do not keep rephrasing the same search, and do not answer from "
            "your own knowledge instead."
        ),
    )

    console.print(
        Panel(
            trajectory.final_answer or "(no answer)",
            title=f"answer ({trajectory.stop_reason.value})",
            style="green" if trajectory.succeeded else "yellow",
        )
    )
    console.print(f"[dim]tools: {' -> '.join(trajectory.tool_sequence) or '(none)'}[/dim]")
    console.print(
        f"[dim]{trajectory.usage.prompt_tokens} input tokens, "
        f"{len(trajectory.steps)} step(s), "
        f"{len(manager.history)} compaction(s) saving {manager.total_saved} tokens[/dim]"
    )
    console.print(
        "[dim]Which search tool did it pick? The choice is driven entirely by the two "
        "tool descriptions -- lesson 2's point, now with two genuinely similar "
        "options to disambiguate.[/dim]"
    )
    if manager.history:
        console.print(
            "[dim]Note the compactions: lessons compose. Retrieval makes tool results "
            "large, so lesson 4's context management stops this agent exceeding the "
            "provider's per-minute token ceiling.[/dim]"
        )


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval over this project's notes.")
    parser.add_argument("--build", action="store_true", help="Index the notes and exit.")
    parser.add_argument("--search", metavar="QUERY", help="Compare all 3 methods on a query.")
    parser.add_argument("--measure", action="store_true", help="Score methods on labelled queries.")
    parser.add_argument("--chunking", action="store_true", help="Does the heading matter?")
    parser.add_argument("--stuffing", action="store_true", help="Retrieval vs stuffing.")
    parser.add_argument("--ask", metavar="QUESTION", help="Run the full agent.")
    parser.add_argument("--top-k", type=int, default=4, help="Results per search (default 4).")
    parser.add_argument("--max-steps", type=int, default=6, help="Agent step cap.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore the embedding cache.")
    args = parser.parse_args()

    if args.rebuild:
        for suffix in (".npy", ".json"):
            target = CACHE.with_suffix(suffix)
            if target.exists():
                target.unlink()
        console.print("[dim]cache cleared[/dim]")

    # Chunking experiment builds its own stores, so it needs no shared one.
    if args.chunking:
        experiment_chunking(args.top_k)
        return 0

    console.print(
        Panel.fit(
            f"embedding model: {args.model}  |  chunks from {REPO_ROOT.name}",
            style="bold cyan",
        )
    )
    store = load_store(verbose=True, model=args.model)

    if args.build:
        console.print("\n[green]Index ready.[/green] Try:")
        console.print('  uv run lessons/05-retrieval/agent.py --search "why is my answer empty"')
        console.print("  uv run lessons/05-retrieval/agent.py --measure")
        return 0

    if args.search:
        experiment_search(store, args.search, args.top_k)
        return 0
    if args.measure:
        experiment_measure(store, args.top_k)
        return 0

    # Remaining modes need a chat model.
    try:
        client = get_client()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if args.stuffing:
        experiment_stuffing(client, store)
    elif args.ask:
        experiment_ask(client, store, args.ask, args.max_steps)
    else:
        experiment_ask(
            client,
            store,
            "Why might my agent return an empty answer, and what should I change?",
            args.max_steps,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
