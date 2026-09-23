"""Splitting documents into retrievable pieces.

Chunking is where most retrieval quality is won or lost, so it gets its own file.

**A cautionary tale about how this file was written**, because it is more useful
than the code.

The theory says a chunk pulled from the middle of a document has lost its context,
so you should embed its heading alongside its body. This text

    "Resolve the path first, then check containment."

ought to be harder to match against "how do I stop a tool reading files outside
the project?" than this text

    "03-agent-loop/NOTES.md > Sandboxing filesystem tools
     Resolve the path first, then check containment."

Tested on a hand-written sample of 8 chunks, that held up beautifully: top-1
accuracy went from 4/7 to 6/7, a bigger gain than doubling the embedding model's
size produced. So the technique went in as the headline finding.

Then it was measured on the real corpus -- 193 chunks from 14 files -- and the
result **did not replicate**: 8/10 top-1 for body only against 7/10 with the
heading, and recall@4 identical at 8/10. Slightly *worse*, or noise.

Two things to take from that:

  * **A result from a tiny hand-picked sample is a hypothesis, not a finding.** The
    8-chunk probe had no competing documents and headings that were all
    informative. At 193 chunks many headings are generic ("The one idea", "Carry
    forward") and add noise rather than signal.
  * **A 1-query difference on a 10-query set is not a difference.** Resolving that
    honestly needs a bigger eval set, which is lesson 7.

The heading is still included below, for a reason that is not retrieval accuracy:
it gives every hit a citation the reader can check. Being explicit about *why* a
design choice survives measurement matters more than the choice.

The trade-off that does hold: chunks too small lose the context needed to
understand them, and chunks too large dilute their own embedding -- one vector has
to represent several ideas, so it sits near none of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Target size in characters. ~1200 chars is roughly 300 tokens: big enough to
#: hold a complete idea, small enough that its embedding stays focused.
DEFAULT_MAX_CHARS = 1200

#: Overlap between consecutive chunks, so a fact sitting on a boundary appears in
#: both. Without it, a sentence split down the middle is retrievable by neither
#: half -- the chunk-boundary problem, which overlap mitigates but never fully
#: solves.
DEFAULT_OVERLAP_CHARS = 150

#: Chunks shorter than this are usually headings with no body, or stray fragments.
#: They embed badly (too little signal) and crowd out real results.
MIN_CHARS = 80


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    text: str
    source: str            # e.g. "lessons/03-agent-loop/NOTES.md"
    heading: str           # e.g. "The one idea"
    heading_path: str      # e.g. "Lesson 3 Notes > The one idea"
    start_line: int
    index: int = 0

    @property
    def label(self) -> str:
        """Human-readable citation, shown to the model and the reader."""
        return f"{self.source} > {self.heading_path}" if self.heading_path else self.source

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded -- heading path first, then body.

        Measured on the real 193-chunk corpus this makes no significant
        difference (7/10 vs 8/10 top-1, identical recall@4). It is kept because
        the heading gives every hit a checkable citation, not because it improves
        ranking. Run `agent.py --chunking` to re-measure on your own notes.
        """
        return f"{self.label}\n{self.text}"

    @property
    def tokens_estimate(self) -> int:
        return max(1, len(self.text) // 4)


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def stats(self) -> dict[str, float]:
        if not self.chunks:
            return {}
        sizes = [len(c.text) for c in self.chunks]
        return {
            "chunks": len(self.chunks),
            "files": len(self.sources),
            "mean_chars": sum(sizes) / len(sizes),
            "min_chars": min(sizes),
            "max_chars": max(sizes),
        }


def _split_oversized(body: str, max_chars: int, overlap: int) -> list[str]:
    """Break a too-long section into overlapping windows, preferring paragraphs.

    Splitting on paragraph boundaries where possible keeps ideas intact. Falling
    back to a hard character window is ugly but necessary -- some sections are one
    enormous paragraph, and a chunk that exceeds the embedding model's input
    limit gets silently truncated by the model, which is worse than splitting it
    yourself.
    """
    if len(body) <= max_chars:
        return [body]

    paragraphs = re.split(r"\n\s*\n", body)
    windows: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            windows.append(current)
        # A single paragraph larger than the budget: hard-split with overlap.
        while len(paragraph) > max_chars:
            windows.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars - overlap :]
        current = paragraph

    if current:
        windows.append(current)
    return [w for w in windows if w.strip()]


def chunk_markdown(
    path: Path,
    root: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split a markdown file on its headings.

    Heading-aware rather than fixed-size, because markdown already tells you
    where the semantic boundaries are. A blind 1000-character window would cut
    mid-sentence and mid-table; splitting on `##` respects the author's own
    structure and gives every chunk a title for free.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    source = path.relative_to(root).as_posix()
    lines = raw.splitlines()

    # Walk the file, tracking the current heading stack so each chunk knows where
    # it sits in the document's hierarchy.
    sections: list[tuple[list[str], int, list[str]]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    section_start = 1
    current_path: list[str] = []
    in_code_fence = False

    def flush(end_line: int) -> None:
        if buffer:
            sections.append((list(current_path), section_start, list(buffer)))
        buffer.clear()

    for number, line in enumerate(lines, start=1):
        # Headings inside fenced code blocks are not headings.
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence

        heading = re.match(r"^(#{1,6})\s+(.*)$", line) if not in_code_fence else None
        if heading:
            flush(number - 1)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = [t for _, t in stack]
            section_start = number
        else:
            buffer.append(line)

    flush(len(lines))

    chunks: list[Chunk] = []
    for heading_path, start_line, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if len(body) < MIN_CHARS:
            continue
        path_text = " > ".join(heading_path)
        for piece in _split_oversized(body, max_chars, overlap):
            if len(piece.strip()) < MIN_CHARS:
                continue
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    source=source,
                    heading=heading_path[-1] if heading_path else "",
                    heading_path=path_text,
                    start_line=start_line,
                )
            )

    return chunks


def build_corpus(
    root: Path,
    patterns: tuple[str, ...] = ("lessons/**/*.md", "docs/*.md", "README.md"),
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> Corpus:
    """Chunk this project's own markdown into a searchable corpus.

    Using the repo's own notes is deliberate. It keeps the lesson honest (you can
    verify every retrieved answer by opening the file), it needs no downloaded
    dataset, and it makes the agent genuinely useful to you -- by lesson 5 there
    are enough notes that finding the right one is a real problem.
    """
    corpus = Corpus()
    seen: set[Path] = set()

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            corpus.chunks.extend(chunk_markdown(path, root, max_chars, overlap))

    for position, chunk in enumerate(corpus.chunks):
        chunk.index = position

    return corpus
