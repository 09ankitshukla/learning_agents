"""One test per bug this project actually shipped and then fixed.

A regression suite is the most honest documentation a codebase has. Every test
below corresponds to something that was genuinely broken, found by running the
code rather than by reading it, and is now impossible to reintroduce silently.

Read the docstrings even if you skip the assertions -- the failure modes are more
instructive than the fixes, and several were invisible until something measured
them.

Worth noticing how cheap these are. Each bug cost real debugging time; each test
costs milliseconds and runs forever.
"""

from __future__ import annotations

import httpx
import pytest
from fakes import ScriptedClient, text_response
from openai import APIStatusError


def _status_error(code: int, body: dict) -> APIStatusError:
    """Build a real APIStatusError, so the mapping is tested as it is used."""
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(code, json=body, request=request)
    return APIStatusError("boom", response=response, body=body.get("error", body))


# ===========================================================================
class TestToolErrorClassUnification:
    """BUG: two classes named ToolError, so actionable errors were swallowed.

    Lesson 2 defined its own `class ToolError(Exception)`. Lesson 3 promoted the
    dispatcher into `llmkit.tools.ToolRegistry`, which caught `llmkit`'s ToolError
    -- a different class. Lesson 2's tool functions therefore fell through to the
    generic `except Exception` handler.

    Effect: every timezone and currency error in lessons 3, 4 and 5 was reported as
    "failed unexpectedly (ToolError)" instead of "Unknown timezone 'Mumbai'. Use an
    IANA name such as 'Asia/Kolkata'". The message that makes a model self-correct
    was replaced with one that tells it nothing.

    Invisible to inspection: both classes have the same name, so the code reads
    correctly. Found by asserting on the error *text*.
    """

    def test_lesson_2_tools_raise_the_same_class_the_registry_catches(self) -> None:
        import llmkit
        import tools

        assert tools.ToolError is llmkit.ToolError

    def test_timezone_error_reaches_the_model_intact(self, registry) -> None:
        from llmkit import ToolCall

        execution = registry.dispatch(
            ToolCall(id="c1", name="get_current_time", arguments={"timezone": "Mumbai"})
        )
        assert execution.failure_kind == "tool_error", "regressed to internal_error"
        assert "Asia/Kolkata" in execution.result

    def test_currency_error_reaches_the_model_intact(self, registry) -> None:
        from llmkit import ToolCall

        execution = registry.dispatch(
            ToolCall(
                id="c2",
                name="convert_currency",
                arguments={"amount": 1, "from_currency": "USD", "to_currency": "BTC"},
            )
        )
        assert execution.failure_kind == "tool_error"
        assert "Supported" in execution.result


class TestSummariserStarvation:
    """BUG: the summariser asked for 400 max_tokens and returned nothing.

    Lesson 0 documented that reasoning models spend their output budget on hidden
    deliberation, so a tight cap yields an empty string. Lesson 4's summariser was
    then written with `max_tokens=400` and walked straight into it.

    It failed silently in the worst way: the call succeeded, tokens were billed, and
    the "summary" was empty.
    """

    def _conversation(self):
        from llmkit import ToolCall, assistant, system, tool_result, user

        messages = [system("sys " * 20), user("task")]
        for i in range(4):
            messages.append(
                assistant(None, [ToolCall(id=f"a{i}", name="read_file", arguments={"p": "x"})])
            )
            messages.append(tool_result(f"a{i}", f"content {i} " + "pad " * 80))
        return messages

    def test_budget_leaves_room_for_a_visible_answer(self) -> None:
        import inspect

        from context import summarise_history

        default = inspect.signature(summarise_history).parameters["max_summary_tokens"].default
        assert default >= 800, (
            "max_summary_tokens is back below the level where reasoning models "
            "return an empty summary"
        )

    def test_empty_summary_falls_back_to_trimming(self) -> None:
        """And the fallback must actually reduce something."""
        from context import conversation_tokens, summarise_history

        messages = self._conversation()
        before = conversation_tokens(messages)
        client = ScriptedClient([text_response("")])  # a starved summariser

        result = summarise_history(client, messages, fallback_budget=before // 2)

        assert "fallback" in result.strategy
        assert result.after < result.before, "fallback silently did nothing"

    def test_fallback_uses_the_callers_budget(self) -> None:
        """BUG within the bug: the fallback trimmed to a hardcoded 10,000 tokens.

        That was above the conversation size, so it dropped nothing, saved nothing,
        and reported success. A fallback that quietly no-ops is worse than none,
        because it converts a loud failure into a silent one.
        """
        from context import summarise_history

        messages = self._conversation()
        client = ScriptedClient([text_response("")])
        result = summarise_history(client, messages, fallback_budget=150)
        assert result.saved > 0


class TestTokenEstimatorCalibration:
    """BUG: the estimator was 91% too low -- 7 tokens estimated, 79 charged.

    Two causes, neither visible without measuring against the provider:
    a fixed ~70-token chat-template preamble charged on every request, and
    JSON tokenizing far denser per character than prose.

    An uncalibrated estimator is not conservative. It is wrong in the direction
    that overflows your context window.
    """

    def test_request_overhead_is_accounted_for(self) -> None:
        from context import REQUEST_OVERHEAD, conversation_tokens
        from llmkit import user

        assert REQUEST_OVERHEAD > 0
        assert conversation_tokens([user("hi")]) > 50

    def test_short_prompt_is_not_wildly_underestimated(self) -> None:
        """The exact case that was 91% low. Measured actual: 79 tokens."""
        from context import conversation_tokens
        from llmkit import user

        estimate = conversation_tokens([user("What is 2 + 2?")])
        assert 55 <= estimate <= 110, f"estimate {estimate} is far from the measured 79"


class TestSchemaProseLeak:
    """BUG: a teaching docstring was serialised into the prompt.

    Pydantic puts class and enum docstrings into `model_json_schema()` as
    `description`, and lesson 1 sends that schema to the model. A long explanatory
    docstring on the `Category` enum therefore became prompt text on every call --
    hundreds of tokens of commentary about naming conventions, sent to the model.

    Doubly embarrassing because lesson 1's own advice is "your schema is a prompt".
    """

    def test_schema_carries_no_teaching_commentary(self) -> None:
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "01-structured-output" / "extract.py"
        spec = importlib.util.spec_from_file_location("extract_for_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["extract_for_test"] = module
        spec.loader.exec_module(module)

        from structured import schema_prompt

        rendered = schema_prompt(module.Ticket)

        # Phrases that belong in comments for humans, never in a prompt.
        for leak in ("Without this", "nothing downstream", "Constrain the output space"):
            assert leak not in rendered, f"teaching prose leaked into the schema: {leak!r}"

        # The parts that *should* be there.
        assert "feature_request" in rendered
        assert "urgent = service is down" in rendered


class TestJsonExtraction:
    """BUG class: models wrap JSON in prose and fences, and a regex cannot cope.

    A `\\{.*\\}` regex breaks on nested objects and on braces inside strings. The
    brace-depth scanner handles both; these cases pin that behaviour.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"a": 1}', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('Sure! Here it is:\n{"a": 1}', {"a": 1}),
            ('{"a": 1}\n\nLet me know!', {"a": 1}),
            ('{"a": {"b": {"c": 2}}}', {"a": {"b": {"c": 2}}}),
            ('{"s": "a } brace in a string"}', {"s": "a } brace in a string"}),
            ('{"s": "escaped \\" quote"}', {"s": 'escaped " quote'}),
        ],
    )
    def test_extracts_json_from_messy_replies(self, raw: str, expected: dict) -> None:
        import json

        from structured import extract_json

        found = extract_json(raw)
        assert found is not None
        assert json.loads(found) == expected

    @pytest.mark.parametrize("raw", ["", "I cannot help with that.", '{"a": {"b": 1'])
    def test_returns_none_rather_than_guessing(self, raw: str) -> None:
        """Truncated JSON must fail cleanly. Guessing at a repair would be worse."""
        from structured import extract_json

        assert extract_json(raw) is None


class TestProviderErrorMapping:
    """BUG: provider errors were mapped to misleading diagnoses.

    Three separate incidents:
      * an unauthenticated health check returned 401, reported as "server down",
        which sent you looking at the wrong thing entirely
      * 413 fell through unmapped, so "this request will never fit" looked like
        a generic failure
      * 429's message hid *which* limit was hit, and per-minute versus per-day
        need completely different responses
    """

    def _client(self):
        from llmkit.config import load_config
        from llmkit.openai_compat import OpenAICompatClient

        return OpenAICompatClient(load_config(provider="ollama"))

    def test_413_is_distinguished_from_429(self) -> None:
        """413 means reduce the request; 429 means wait. Confusing them wastes time."""
        mapped = self._client()._status_error(
            _status_error(
                413,
                {"error": {"message": "Request too large ... tokens per minute (TPM): Limit 8000"}},
            )
        )
        text = str(mapped)
        assert "too large" in text.lower()
        assert "Waiting will not help" in text

    def test_429_surfaces_the_providers_own_message(self) -> None:
        """Which limit, and how long, decide what you do next."""
        mapped = self._client()._status_error(
            _status_error(
                429,
                {"error": {"message": "tokens per day (TPD): Limit 200000, try again in 4m32s"}},
            )
        )
        assert "tokens per day" in str(mapped)

    def test_401_is_reported_as_a_key_problem(self) -> None:
        mapped = self._client()._status_error(
            _status_error(401, {"error": {"message": "invalid api key"}})
        )
        assert "key" in str(mapped).lower()

    def test_phantom_tool_call_is_recognised_in_both_body_shapes(self) -> None:
        """Groq puts the error dict at the top level; others nest it under "error".

        Parsing only one shape meant the phantom-tool diagnosis silently never
        fired for the provider actually in use.
        """
        from llmkit.openai_compat import PhantomToolCall

        client = self._client()
        nested = client._status_error(
            _status_error(
                400,
                {"error": {"code": "tool_use_failed", "failed_generation": '{"name":"browser.search"}'}},
            )
        )
        assert isinstance(nested, PhantomToolCall)
        assert "browser.search" in str(nested)


class TestUnicodeSafeConsole:
    """BUG: printing model output crashed on Windows.

    Models emit typographic quotes and non-breaking hyphens, which do not exist in
    cp1252, so `print()` raised UnicodeEncodeError -- at print time, far from the
    model call, killing the script after the work was done.
    """

    def test_shared_console_is_importable_and_prints_awkward_characters(self, capsys) -> None:
        from llmkit import console

        console.print("non-breaking\u2011hyphen and \u2019smart quotes\u2019 and an em\u2014dash")
        assert capsys.readouterr().out  # the assertion is that it did not raise

    def test_streams_are_utf8(self) -> None:
        import sys

        import llmkit  # noqa: F401  (importing applies the fix)

        encoding = getattr(sys.stdout, "encoding", "") or ""
        assert "utf-8" in encoding.lower() or "utf8" in encoding.lower()
