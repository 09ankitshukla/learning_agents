# Lesson 1 — Notes

## The one idea

**A model is a component with a failure rate, not a parser.** Engineer around it the way you'd engineer around any flaky dependency: validate the output, and retry with feedback about what was wrong.

## The pattern

```
DESCRIBE -> EXTRACT -> VALIDATE -> REPAIR (loop back)
```

1. **Describe** — generate the JSON Schema from your Pydantic model so the prompt and the validation can't drift apart. Never hand-write a schema in a prompt string.
2. **Extract** — models wrap JSON in fences and prose. Scan for the first `{` and walk forward tracking brace depth, ignoring braces inside strings. A `\{.*\}` regex breaks on nesting.
3. **Validate** — Pydantic. Its error messages are precise and already model-readable.
4. **Repair** — append the model's bad reply *and* the validation error to the conversation, then ask again. The model can now see its own mistake.

Why repair works: the model isn't retrying blind, it's responding to a specific correction. Same reason a compiler error helps you more than "build failed."

## Rules of thumb

**Constrain the output space.** Enums over free strings. Five allowed values are far easier to hit than open text, and they make downstream grouping possible. Free-text categories give you `Billing`, `billing issue`, and `Payment/Invoice Problem` for the same thing.

**Field descriptions are prompt engineering.** They're serialised into the schema and sent to the model. Weakening one measurably reduces consistency. Write them for a reader with no other context.

**Optional beats required when data may be absent.** A required field is a demand, and a model would rather invent a plausible value than fail. `customer_name: str | None` plus "use null, do not guess" is much safer than `customer_name: str`.

**Return metadata, not just the value.** `ExtractionResult` carries `attempts`, `errors` and `usage`. An extractor that always succeeds on attempt 3 is telling you something is wrong, and you can't see that if you only return the parsed object. Retrofitting measurement later is painful — build it in from lesson 1.

## json_mode: what it does and doesn't do

| Failure | Fixed by json_mode? |
|---|---|
| markdown fences, prose preamble | yes |
| trailing commas, unquoted keys | yes |
| wrong enum value | **no** |
| missing required field | **no** |
| hallucinated value | **no** |

**Valid is not correct.** Use json_mode when the server supports it, to remove one class of failure. Keep the validation and repair loop regardless. Support is also uneven across local models, which is a second reason not to depend on it.

Constrained decoding via a full JSON Schema (rather than just `json_object`) is stronger, and some servers support it. It still can't stop a hallucinated-but-well-typed value.

## Gotchas worth remembering

**`finish_reason == "length"` is a truncation bug wearing a costume.** The reply got cut off mid-JSON by `max_tokens`, so parsing fails and it looks like the model can't produce JSON. Check this field first whenever output looks mangled. Handle it separately from a parse error, because the fix is different (raise the budget or shorten the schema).

**Temperature 0 is not determinism.** Identical input can produce different categories across runs, because server-side batching changes floating-point reduction order. Consequence: never assert exact model output in a test, and never judge a change by one run.

**Model size hits tool/schema reliability harder than it hits chat quality.** A 3B model chats fine and then produces invalid structured output far more often than a 7B. Below ~7B, expect the repair loop to earn its keep.

## The payoff

Once validated, it's ordinary typed Python:

```python
if ticket.priority in (Priority.URGENT, Priority.HIGH) and ticket.needs_human:
    page_oncall()
```

That boundary — text on one side, typed data and normal control flow on the other — is where a language model becomes usable software.

## Carry forward

The same describe/validate/repair cycle handles malformed tool arguments in lesson 2, runtime tool exceptions in lesson 3, and guardrail violations in lesson 11. It's the same shape every time.

## Measured results

On `openai/gpt-oss-120b` via Groq, against the built-in double-charge ticket:

**Reliability (5 identical runs, temperature 0):**

- first-attempt success: **3/5**
- distinct categories across runs: **1** (`billing`)
- distinct priorities across runs: **1** (`urgent`)

Read that carefully, because it's the lesson in one line: a strong model still failed schema validation on 40% of first attempts, and the repair loop absorbed every one of those invisibly. Final output was perfectly consistent. Without the loop, this extractor would look like it fails two runs in five.

**json_mode vs. prompt-only (same input):**

| Approach | Attempts | Tokens |
|---|---|---|
| Prompt-only | 2 | 3,044 |
| `json_mode` | 1 | 1,716 |

json_mode nearly halved token use by removing the retry. Worth using where supported — but it still guarantees only valid JSON, not correct JSON, so the loop stays.

**Your turn:** rerun these on a smaller model (`openai/gpt-oss-20b`, or `qwen2.5:3b-instruct` locally) and record the difference. That's your own evidence for the size/reliability trade-off:

- first-attempt success on a smaller model: `___`
- distinct categories across 5 runs: `___`

## Reasoning models and structured output

If your model is a reasoning model (lesson 0), two things follow here:

**Keep `max_tokens` generous.** Reasoning consumes the same budget as the answer. A tight cap yields a truncated or empty reply, which the loop then dutifully "repairs" — burning attempts on a problem no prompt can fix. That's why `extract_structured` treats `finish_reason == "length"` as its own case with its own message rather than lumping it in with a parse error.

**Expect better schema compliance for more tokens.** Reasoning models are generally stronger at hitting a schema, and they cost several times more output tokens to do it. That's a real trade, not a free win.
