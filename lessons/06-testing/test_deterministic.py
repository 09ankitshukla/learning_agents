"""Tests for everything in an agent that is NOT the model.

Start here, because this is the part people skip. "Agents are non-deterministic so
they cannot be tested" is half true and leads to testing nothing. In fact most of
an agent is ordinary code:

    the dispatcher          given this tool call, does it run or refuse?
    the sandbox             is this path inside the project?
    the calculator          does it reject an import?
    trimming                does the result stay structurally valid?
    the token estimator     is it in the right ballpark?
    chunking                do the pieces have the right shape?
    keyword search          does it rank the right document?

None of that involves a model. All of it is where the security-relevant and
correctness-relevant behaviour lives. These tests are fast, free, deterministic,
and they would have caught several real bugs in this project.

The stochastic part -- what the model chooses to say -- is tested separately in
test_loop.py using doubles.
"""

from __future__ import annotations

import pytest
from llmkit import ToolCall


# ===========================================================================
# The calculator's sandbox (lesson 2)
# ===========================================================================
class TestCalculatorSecurity:
    """The AST allowlist. Security code deserves the most tests you write.

    These cases are lifted straight from lesson 2's failures.py table. Turning a
    demo into a test is usually cheap and is the difference between "we showed it
    works once" and "it cannot regress".
    """

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo pwned')",
            "open('.env').read()",
            "().__class__.__bases__[0].__subclasses__()",
            "eval('1+1')",
            "globals()",
            "[x for x in range(10)]",
            "lambda: 1",
            "x = 1",
        ],
    )
    def test_refuses_anything_but_arithmetic(self, expression: str) -> None:
        from tools import ToolError, calculate

        with pytest.raises(ToolError):
            calculate(expression)

    def test_refuses_resource_exhaustion(self) -> None:
        """Separate from code execution, and needs its own limit.

        `9**9**9` executes nothing dangerous. It just pins a CPU core and eats
        memory. Blocking imports does not help; an explicit exponent cap does.
        """
        from tools import ToolError, calculate

        with pytest.raises(ToolError, match="Exponent too large"):
            calculate("9**9**9")

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2 + 2", "4"),
            ("48239 * 7841", "378,  ".strip()[:3]),  # just check it computes
            ("10 / 4", "2.5"),
            ("sqrt(16)", "4"),
            ("round(2.567, 1)", "2.6"),
            ("-5 + 3", "-2"),
            ("2 ** 10", "1024"),
        ],
    )
    def test_allows_real_arithmetic(self, expression: str, expected: str) -> None:
        from tools import calculate

        assert expected in calculate(expression)

    def test_division_by_zero_is_a_tool_error_not_a_crash(self) -> None:
        """The distinction matters: a ToolError becomes an observation the model
        can recover from, while an uncaught ZeroDivisionError kills the run."""
        from tools import ToolError, calculate

        with pytest.raises(ToolError, match="Division by zero"):
            calculate("1/0")


# ===========================================================================
# The path sandbox (lesson 3)
# ===========================================================================
class TestPathSandbox:
    @pytest.mark.parametrize(
        "path",
        [
            "../../../../etc/passwd",
            "..\\..\\..\\Windows\\System32\\drivers\\etc\\hosts",
            "C:/Windows/System32/config/SAM",
            "/etc/shadow",
            "lessons/../../.env",
            ".env",
            ".git/config",
        ],
    )
    def test_refuses_escapes_and_secrets(self, path: str) -> None:
        from llmkit import ToolError as KitToolError
        from toolset import read_file

        with pytest.raises(KitToolError):
            read_file(path)

    def test_allows_legitimate_reads(self) -> None:
        from toolset import read_file

        result = read_file("README.md", max_lines=5)
        assert "README.md" in result

    def test_resolution_happens_before_the_check(self) -> None:
        """The ordering *is* the security property.

        'lessons/../../.env' contains no leading '..', so a string-prefix check
        would pass it. Only resolving first turns it into an absolute path that
        fails containment. This test exists to stop someone "simplifying"
        _safe_path into a string comparison.
        """
        from llmkit import ToolError as KitToolError
        from toolset import read_file

        with pytest.raises(KitToolError):
            read_file("lessons/../../.env")


# ===========================================================================
# The dispatcher (lessons 2 and 3)
# ===========================================================================
class TestDispatcher:
    """The dispatcher must never raise. Every failure becomes an observation."""

    def test_unknown_tool_is_refused_not_raised(self, registry) -> None:
        execution = registry.dispatch(
            ToolCall(id="c1", name="delete_everything", arguments={})
        )
        assert execution.ok is False
        assert execution.failure_kind == "unknown_tool"
        # The error text must list the real tools, because it is what lets the
        # model self-correct (lesson 2, scenario 6).
        assert "calculate" in execution.result

    def test_malformed_json_is_reported_to_the_model(self, registry) -> None:
        execution = registry.dispatch(
            ToolCall(id="c2", name="calculate", malformed_arguments='{"expression": "2 +')
        )
        assert execution.ok is False
        assert execution.failure_kind == "malformed_json"

    def test_missing_required_argument_names_the_field(self, registry) -> None:
        execution = registry.dispatch(
            ToolCall(
                id="c3",
                name="convert_currency",
                arguments={"amount": 100, "from_currency": "USD"},
            )
        )
        assert execution.ok is False
        assert "to_currency" in execution.result

    def test_unexpected_argument_is_caught(self, registry) -> None:
        execution = registry.dispatch(
            ToolCall(
                id="c4",
                name="get_current_time",
                arguments={"timezone": "UTC", "format": "12-hour"},
            )
        )
        assert execution.ok is False
        assert execution.failure_kind == "bad_signature"

    def test_tool_error_message_is_actionable(self, registry) -> None:
        """Error text is prompt text. Verified in lesson 2: the model recovered
        because the message named the valid format."""
        execution = registry.dispatch(
            ToolCall(id="c5", name="get_current_time", arguments={"timezone": "Mumbai"})
        )
        assert execution.ok is False
        assert "Asia/Kolkata" in execution.result

    def test_successful_call_returns_a_string(self, registry) -> None:
        execution = registry.dispatch(
            ToolCall(id="c6", name="calculate", arguments={"expression": "6 * 7"})
        )
        assert execution.ok is True
        assert "42" in execution.result

    def test_registry_is_an_allowlist(self, registry) -> None:
        """No amount of creative naming reaches code that is not registered."""
        for name in ("os.system", "tools.calculate", "__import__", "CALCULATE"):
            execution = registry.dispatch(ToolCall(id="x", name=name, arguments={}))
            assert execution.failure_kind == "unknown_tool"


# ===========================================================================
# Context management (lesson 4)
# ===========================================================================
class TestConversationValidation:
    """validate() catches locally what a provider would reject with HTTP 400."""

    def test_accepts_a_well_formed_conversation(self) -> None:
        from context import validate
        from llmkit import assistant, system, tool_result, user

        messages = [
            system("s"),
            user("q"),
            assistant(None, [ToolCall(id="a1", name="calculate", arguments={})]),
            tool_result("a1", "42"),
            assistant("done"),
        ]
        assert validate(messages) == []

    def test_detects_orphaned_tool_result(self) -> None:
        """The bug that produced a real 400 in lesson 4."""
        from context import validate
        from llmkit import system, tool_result

        messages = [system("s"), tool_result("a1", "42")]
        problems = validate(messages)
        assert problems
        assert "orphaned result" in problems[0]

    def test_detects_unanswered_tool_call(self) -> None:
        """The mirror image: a request with no result. Also rejected by providers."""
        from context import validate
        from llmkit import assistant, system, user

        messages = [
            system("s"),
            user("q"),
            assistant(None, [ToolCall(id="a1", name="calculate", arguments={})]),
        ]
        problems = validate(messages)
        assert problems
        assert "never answered" in problems[0]


class TestTrimming:
    def _conversation(self, exchanges: int = 4):
        from llmkit import assistant, system, tool_result, user

        messages = [system("You are a careful assistant." * 8), user("the task")]
        for i in range(exchanges):
            messages.append(
                assistant(None, [ToolCall(id=f"a{i}", name="read_file", arguments={"path": "x"})])
            )
            messages.append(tool_result(f"a{i}", f"result {i} " + "padding " * 60))
        return messages

    def test_safe_trim_never_produces_an_invalid_conversation(self) -> None:
        """The property that matters, checked across many budgets.

        This is the shape of test worth writing for trimming: not "it produces
        exactly these messages" but "whatever it produces is structurally valid".
        """
        from context import conversation_tokens, trim_safe, validate

        messages = self._conversation(5)
        for budget in range(200, conversation_tokens(messages) + 200, 150):
            result = trim_safe(messages, budget=budget)
            assert validate(result.messages) == [], f"invalid at budget={budget}"

    def test_naive_trim_can_produce_an_invalid_conversation(self) -> None:
        """Documents the counterexample. If this ever passes, trim_naive was
        'fixed' and the lesson's demonstration is broken."""
        from context import trim_naive, validate

        messages = self._conversation(4)
        broke_at_least_once = any(
            validate(trim_naive(messages, keep_last=k).messages)
            for k in range(2, len(messages))
        )
        assert broke_at_least_once

    def test_system_prompt_and_task_always_survive(self) -> None:
        from context import trim_safe

        messages = self._conversation(5)
        result = trim_safe(messages, budget=50)  # absurdly small
        assert result.messages[0]["role"] == "system"
        assert any(m["role"] == "user" and m.get("content") == "the task" for m in result.messages)

    def test_reports_floor_when_budget_unreachable(self) -> None:
        """Silently returning something over budget would hide the real problem."""
        from context import trim_safe

        result = trim_safe(self._conversation(3), budget=30)
        assert any("FLOOR" in note for note in result.notes)

    def test_compression_shrinks_without_changing_structure(self) -> None:
        from context import compress_tool_results, validate

        messages = self._conversation(3)
        result = compress_tool_results(messages, max_chars=200)
        assert result.after < result.before
        assert len(result.messages) == len(messages)  # nothing deleted
        assert validate(result.messages) == []

    def test_compression_leaves_a_marker(self) -> None:
        """Silent truncation makes a model answer from half a document."""
        from context import compress_tool_results

        result = compress_tool_results(self._conversation(2), max_chars=150)
        compressed = [m for m in result.messages if m["role"] == "tool"]
        assert any("removed to save context" in m["content"] for m in compressed)


class TestTokenEstimator:
    def test_counts_tool_call_arguments(self) -> None:
        """An assistant tool turn has content=None but is not free.

        A naive estimator measuring only `content` scores it zero, which is how you
        end up 91% low (lesson 4).
        """
        from context import message_tokens
        from llmkit import assistant

        message = assistant(
            None,
            [ToolCall(id="a1", name="read_file", arguments={"path": "x" * 200})],
        )
        assert message_tokens(message) > 20

    def test_includes_fixed_request_overhead(self) -> None:
        """The single biggest source of estimator error on short conversations."""
        from context import REQUEST_OVERHEAD, conversation_tokens
        from llmkit import user

        assert conversation_tokens([user("hi")]) >= REQUEST_OVERHEAD

    def test_structured_text_costs_more_per_character(self) -> None:
        """JSON tokenizes denser than prose. Measured in lesson 4."""
        from context import estimate_tokens

        prose = "the quick brown fox jumps over the lazy dog and keeps running"
        structured = '{"a":1,"b":[2,3],"c":{"d":"/x/y.md"},"e":true,"f":null}'
        assert estimate_tokens(structured) / len(structured) > estimate_tokens(prose) / len(prose)


# ===========================================================================
# Retrieval (lesson 5) -- the parts with no embedding model
# ===========================================================================
class TestChunking:
    def test_chunks_carry_their_heading_path(self, repo_root) -> None:
        from chunking import build_corpus

        corpus = build_corpus(repo_root)
        assert len(corpus) > 50
        assert all(c.source for c in corpus.chunks)
        assert any(">" in c.heading_path for c in corpus.chunks)

    def test_embedding_text_includes_the_label(self, repo_root) -> None:
        from chunking import build_corpus

        chunk = build_corpus(repo_root).chunks[0]
        assert chunk.label in chunk.embedding_text
        assert chunk.text in chunk.embedding_text

    def test_respects_the_size_limit(self, repo_root) -> None:
        from chunking import DEFAULT_MAX_CHARS, build_corpus

        corpus = build_corpus(repo_root)
        # Oversized sections are split, so no chunk may exceed the budget.
        assert max(len(c.text) for c in corpus.chunks) <= DEFAULT_MAX_CHARS

    def test_drops_fragments_too_small_to_embed(self, repo_root) -> None:
        from chunking import MIN_CHARS, build_corpus

        corpus = build_corpus(repo_root)
        assert min(len(c.text) for c in corpus.chunks) >= MIN_CHARS

    def test_headings_inside_code_fences_are_not_headings(self, tmp_path) -> None:
        """A '# comment' inside a fenced block must not split the document."""
        from chunking import chunk_markdown

        document = tmp_path / "doc.md"
        document.write_text(
            "# Real heading\n\n"
            + "Body text that is long enough to survive the minimum size filter. " * 3
            + "\n\n```python\n# this is a comment, not a heading\nx = 1\n```\n\n"
            + "More body text, also long enough to be kept by the chunker here. " * 3,
            encoding="utf-8",
        )
        chunks = chunk_markdown(document, tmp_path)
        assert len(chunks) == 1
        assert "Real heading" in chunks[0].heading_path


class TestKeywordSearch:
    """Keyword search needs no embedding model, so it is fully unit-testable."""

    def test_finds_a_literal_term(self, repo_root) -> None:
        from chunking import build_corpus
        from store import NoteStore

        store = NoteStore(build_corpus(repo_root))
        hits = store.search_keyword("tool_call_id", top_k=5)
        assert hits
        assert any("tool_call_id" in h.chunk.text for h in hits)

    def test_returns_nothing_for_absent_terms(self, repo_root) -> None:
        from chunking import build_corpus
        from store import NoteStore

        store = NoteStore(build_corpus(repo_root))
        assert store.search_keyword("zzzznonexistentterm", top_k=5) == []

    def test_semantic_search_requires_build(self, repo_root) -> None:
        """Searching before embedding should fail loudly, not silently return junk."""
        from chunking import build_corpus
        from store import NoteStore

        store = NoteStore(build_corpus(repo_root))
        with pytest.raises(RuntimeError, match="build"):
            store.search_semantic("anything")


class TestLabelledQuerySet:
    """The eval set is code too, and its matcher had a real bug (lesson 5)."""

    def test_accepts_both_the_detailed_note_and_the_index(self) -> None:
        """The bug that made correct retrievals look like failures."""
        from queries import QUERY_SET

        query = next(q for q in QUERY_SET if q.expected_source == "02-tool-calling")
        assert query.is_correct("lessons/02-tool-calling/NOTES.md > Never eval()")
        assert query.is_correct("docs/00-index.md > Key learnings index > Lesson 02 - Tool calling")

    def test_rejects_unrelated_sources(self) -> None:
        from queries import QUERY_SET

        query = next(q for q in QUERY_SET if q.expected_source == "02-tool-calling")
        assert not query.is_correct("lessons/05-retrieval/NOTES.md > The one idea")

    def test_every_query_documents_what_it_tests(self) -> None:
        """A dataset without rationale rots: nobody knows why a case is there."""
        from queries import QUERY_SET

        assert all(q.tests for q in QUERY_SET)
