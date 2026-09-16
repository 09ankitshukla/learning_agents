"""Turning a language model into a function that returns typed data.

This is the foundational pattern of the whole course. Read it slowly.

A model returns text. Your program needs data. Bridging that gap reliably takes
four moves, and every agent framework you will ever use is doing some version of
them underneath:

    1. DESCRIBE  - tell the model the exact shape you want (a JSON Schema)
    2. EXTRACT   - pull JSON out of a reply that may contain prose or fences
    3. VALIDATE  - check it against the schema (Pydantic)
    4. REPAIR    - on failure, show the model its error and ask again

Step 4 is the one beginners skip and the one that makes the difference. A model
is not a parser and not an API; it is a component with a failure rate. The
correct response to a failure rate is a feedback loop, not hope.

The same describe/validate/repair cycle reappears in lesson 2 (malformed tool
arguments) and lesson 3 (a tool raising at runtime). Learn it once here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from llmkit import LLMClient, Usage, assistant, system, user

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """What happened, not just what came out.

    `attempts` and `errors` are as important as `value`. An extractor that
    succeeds on attempt 3 every single time is telling you your prompt or your
    schema is wrong, and you cannot see that if you only return the value.
    Measurement lives in the return type from lesson 1 onward, deliberately.
    """

    value: Any | None
    attempts: int = 0
    usage: Usage = field(default_factory=Usage)
    errors: list[str] = field(default_factory=list)
    raw_replies: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def repaired(self) -> bool:
        """True when it only worked because we retried. Worth logging."""
        return self.ok and self.attempts > 1


# ---------------------------------------------------------------------------
# 1. DESCRIBE
# ---------------------------------------------------------------------------
def schema_prompt(model_cls: type[BaseModel]) -> str:
    """Build the instruction that tells the model what shape to produce.

    We derive the schema from the Pydantic model rather than writing it in prose,
    so the description the model sees and the validation it must pass can never
    drift apart. Duplicating a schema in a prompt string is a guaranteed
    source of confusing bugs later.

    Field `description=` values in your model become documentation for the model.
    They are the highest-leverage text in the whole file -- treat them as prompt
    engineering, not code comments.
    """
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return (
        "You extract structured data from text.\n\n"
        "Return a single JSON object that conforms to this JSON Schema:\n\n"
        f"{schema}\n\n"
        "Rules:\n"
        "- Output JSON only. No prose, no explanation, no markdown code fences.\n"
        "- Include every required field.\n"
        "- Use only the allowed values for enum fields.\n"
        "- If a value is genuinely not present in the text, use null for optional "
        "fields rather than inventing one."
    )


# ---------------------------------------------------------------------------
# 2. EXTRACT
# ---------------------------------------------------------------------------
def extract_json(raw: str) -> str | None:
    """Find a JSON object inside whatever the model actually said.

    You asked for JSON only. You will still receive, in rough order of
    frequency:

        ```json\n{...}\n```
        Sure! Here is the JSON you requested:\n{...}
        {...}\n\nLet me know if you need anything else!
        {...} with a trailing comma

    A regex like r'\\{.*\\}' breaks on nested objects. So we scan for the first
    '{' and walk forward tracking brace depth, ignoring braces inside strings,
    which handles nesting correctly.
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip a markdown fence if the whole reply is wrapped in one.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop ```json
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i, ch in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None  # unbalanced -- usually means max_tokens truncated the reply


# ---------------------------------------------------------------------------
# 3 + 4. VALIDATE and REPAIR
# ---------------------------------------------------------------------------
def extract_structured(
    client: LLMClient,
    model_cls: type[T],
    text: str,
    max_attempts: int = 3,
    use_json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 800,
    verbose: bool = False,
) -> ExtractionResult:
    """Ask the model for `model_cls`, validate, and repair on failure.

    The repair loop is the point. On a failed attempt we append two things to the
    conversation: what the model said, and what was wrong with it. That turns a
    one-shot gamble into a conversation where the model can see and correct its
    own mistake. Success rates on small local models improve dramatically.

    `use_json_mode` asks the server to constrain output to syntactically valid
    JSON. When available it removes step 2's problems entirely -- but note what
    it does *not* do: it guarantees valid JSON, never *correct* JSON. Enum
    violations, missing fields and hallucinated values all survive json_mode. So
    validation and repair stay necessary either way. Run the CLI with --compare
    to see this yourself.
    """
    messages = [system(schema_prompt(model_cls)), user(f"Extract from this text:\n\n{text}")]

    result = ExtractionResult(value=None)

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt

        reply = client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=use_json_mode,
        )
        result.usage = result.usage + reply.usage
        raw = reply.text or ""
        result.raw_replies.append(raw)

        if verbose:
            print(f"  attempt {attempt}: {len(raw)} chars, {reply.usage.latency_s:.1f}s")

        # Truncation is a common, silently confusing failure. Name it explicitly.
        if reply.finish_reason == "length":
            problem = (
                "Your reply was cut off before the JSON was complete. "
                "Be more concise and return only the JSON object."
            )
            result.errors.append(f"attempt {attempt}: truncated (finish_reason=length)")
            messages += [assistant(raw), user(problem)]
            continue

        candidate = extract_json(raw)
        if candidate is None:
            problem = (
                "I could not find a JSON object in your reply. "
                "Respond with only a JSON object, starting with { and ending with }."
            )
            result.errors.append(f"attempt {attempt}: no JSON object found")
            messages += [assistant(raw), user(problem)]
            continue

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            problem = (
                f"Your JSON is syntactically invalid: {exc.msg} at line {exc.lineno}, "
                f"column {exc.colno}. Return corrected JSON only."
            )
            result.errors.append(f"attempt {attempt}: invalid JSON - {exc.msg}")
            messages += [assistant(raw), user(problem)]
            continue

        try:
            result.value = model_cls.model_validate(parsed)
            return result  # success
        except ValidationError as exc:
            # Pydantic's error text is already precise and actionable, which is
            # exactly what a model needs to fix itself. Pass it through.
            problem = (
                "Your JSON does not match the required schema:\n\n"
                f"{_format_validation_errors(exc)}\n\n"
                "Return the corrected JSON object only."
            )
            result.errors.append(f"attempt {attempt}: schema violation - {exc.error_count()} error(s)")
            messages += [assistant(raw), user(problem)]

    return result  # exhausted attempts; value stays None


def _format_validation_errors(exc: ValidationError) -> str:
    """Render Pydantic errors as a short, model-readable list."""
    lines = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"- field '{location}': {err['msg']}")
    return "\n".join(lines)
