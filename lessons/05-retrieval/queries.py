"""A labelled query set, so retrieval quality is measured rather than asserted.

Every query here is phrased the way a person would actually ask it, deliberately
*avoiding* the vocabulary used in the target note. That is the whole point: if the
question reuses the document's words, keyword search already works and embeddings
add nothing.

This file is also the seed of lesson 7. Thirty lines of (input, expected output)
pairs is an eval dataset, and once you have one you can compare two retrieval
methods, two chunk sizes or two models honestly instead of trying three queries by
hand and trusting your impression.

Note the honesty requirement: these labels were written by reading the notes and
deciding where each answer lives. Where an answer legitimately appears in two
places, `also_acceptable` records that, because marking a correct retrieval as
wrong is just as damaging to a measurement as the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LabelledQuery:
    question: str
    #: Which lesson's material answers this. Matched against the chunk's full
    #: label (source path AND heading path), not just the path -- see below.
    expected_source: str
    #: Other lessons whose material would also be a correct top hit.
    also_acceptable: list[str] = field(default_factory=list)
    #: Why this query is in the set -- what it is testing.
    tests: str = ""

    def _patterns(self) -> list[str]:
        """Every way the expected material can legitimately be labelled.

        This method exists because of a real measurement bug found while building
        the lesson. The first version matched only the source *path*, so a query
        expecting "02-tool-calling" was scored as a MISS when retrieval returned:

            docs/00-index.md > Key learnings index > Lesson 02 - Tool calling

        That is the condensed summary of exactly the right learning. Semantic
        search had found the right content and the label called it wrong.

        The lesson generalises well beyond this repo: **your ground truth encodes
        assumptions, and an apparent miss is sometimes the system finding a better
        answer than the one you wrote down.** Always read your failures before
        trusting a score. An eval that is subtly wrong is more dangerous than no
        eval, because it looks authoritative.
        """
        out: list[str] = []
        for source in [self.expected_source, *self.also_acceptable]:
            out.append(source)
            # "02-tool-calling" -> also accept "Lesson 02", which is how
            # docs/00-index.md labels the same material.
            number = source.split("-")[0]
            if number.isdigit():
                out.append(f"Lesson {number}")
        return out

    def is_correct(self, chunk_label: str) -> bool:
        """`chunk_label` should be Chunk.label: source path + heading path."""
        return any(p in chunk_label for p in self._patterns())


QUERY_SET: list[LabelledQuery] = [
    LabelledQuery(
        question="How do we stop the agent running dangerous code?",
        expected_source="02-tool-calling",
        tests="pure vocabulary mismatch: the note says eval/AST/allowlist, never "
              "'dangerous'. The hardest query in the set.",
    ),
    LabelledQuery(
        question="What makes something an agent rather than just a chatbot?",
        expected_source="03-agent-loop",
        tests="conceptual question against a definitional note.",
    ),
    LabelledQuery(
        question="Why did the provider reject my shortened conversation?",
        expected_source="04-memory-context",
        tests="paraphrase: 'shortened' for 'trimmed', 'provider' for 'API'.",
    ),
    LabelledQuery(
        question="Why is my model returning an empty answer?",
        expected_source="00-setup",
        also_acceptable=["04-memory-context"],
        tests="a symptom rather than a term. Documented in two places, so both count.",
    ),
    LabelledQuery(
        question="How do I stop a tool reading files outside the project folder?",
        expected_source="03-agent-loop",
        tests="describes the goal, not the mechanism (resolve + containment).",
    ),
    LabelledQuery(
        question="What should I do when the model gives me invalid JSON?",
        expected_source="01-structured-output",
        tests="close lexical overlap -- keyword search should manage this one too.",
    ),
    LabelledQuery(
        question="cheapest way to make the conversation smaller",
        expected_source="04-memory-context",
        tests="terse, keyword-free phrasing of 'compress tool results'.",
    ),
    LabelledQuery(
        question="Does the model actually execute my functions itself?",
        expected_source="02-tool-calling",
        tests="the misconception lesson 2 exists to correct.",
    ),
    LabelledQuery(
        question="how much do I pay for thinking I never see",
        expected_source="00-setup",
        tests="colloquial phrasing of 'reasoning tokens are billed as output'.",
    ),
    LabelledQuery(
        question="what stops an agent from looping forever",
        expected_source="03-agent-loop",
        tests="goal-phrased question about max_steps and stall detection.",
    ),
]


def summary() -> str:
    return (
        f"{len(QUERY_SET)} labelled queries across "
        f"{len({q.expected_source for q in QUERY_SET})} source areas"
    )
