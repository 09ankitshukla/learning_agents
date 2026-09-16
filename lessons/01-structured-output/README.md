# Lesson 1 — Structured output: making a model behave like a function

**Time:** 1–1.5 hours
**You will end with:** a CLI that turns messy support tickets into validated, typed Python objects, and recovers on its own when the model gets it wrong
**Depends on:** lesson 0

---

## Learn first

### The problem

You want this:

```python
ticket = triage(raw_text)
if ticket.priority == "urgent":
    page_oncall()
```

What you get from a model is text. Sometimes that text is the JSON you asked for. Sometimes it's:

````text
Sure! Here's the extracted information:

```json
{
  "summary": "Customer double-charged",
  "priority": "Very High",
}
```

Let me know if you'd like me to adjust anything!
````

Four problems in one reply: a prose preamble, a markdown fence, a trailing comma, and `"Very High"` when your enum only allows `urgent | high | normal | low`. Every one of these breaks `json.loads` or your type system.

The instinct is to write a better prompt until this stops happening. That instinct is wrong, and understanding why is the point of this lesson.

### The reframe

**A model is not a parser or an API. It is a component with a failure rate.**

You already know how to engineer around components with failure rates: you validate their output and you retry with feedback. You don't ask a flaky network to try harder; you add a retry policy. Same here.

So the shape is a loop, not a prompt:

```
DESCRIBE  ->  send an exact JSON Schema, generated from your type
EXTRACT   ->  find the JSON inside whatever text came back
VALIDATE  ->  check it against the schema
REPAIR    ->  on failure, show the model its own error and ask again
```

The repair step is what beginners skip. It is also what turns a 60%-reliable extractor into a 95%-reliable one on a small local model, because a validation error is precise, actionable feedback and models are good at acting on precise feedback.

### Why this is the foundation of the course

The same cycle reappears everywhere:

| Lesson | The unreliable output | The validation | The repair |
|---|---|---|---|
| 1 | JSON that must match a schema | Pydantic | error text fed back |
| 2 | tool arguments | schema check | ask for the call again |
| 3 | a tool that raises at runtime | the exception | error becomes an observation |
| 11 | output that must pass a safety rule | guardrail | block or regenerate |

Learn it properly once, here, on the simplest case.

### Two supporting ideas

**Your schema is a prompt.** Field names, `description=` text and enum values are all serialised and sent to the model. `priority: str` gets you nothing; `Priority` as an enum with a description explaining what "urgent" means gets you consistency. Constrain the output space wherever the domain allows — a closed set of five values is far easier for a model to hit than free text.

**json_mode is not a solution, it's one layer of it.** Many servers support `response_format={"type": "json_object"}`, which constrains generation to syntactically valid JSON. Useful — it eliminates fences, preambles and trailing commas. It does *nothing* about wrong enum values, missing required fields, or invented customer names. Valid is not correct. You still need validation and repair. `--compare` demonstrates this directly.

---

## Then apply

### Run it

```powershell
uv run lessons/01-structured-output/extract.py
```

The built-in ticket is deliberately hard: it contains two distinct problems (a billing error and a bug), an explicit threat to escalate, a named customer, and an implied priority never stated outright. Watch the `attempt N` lines — on a 7B local model you will often see a repair happen.

### See what the model actually receives

```powershell
uv run lessons/01-structured-output/extract.py --show-schema
```

Read this output properly. It's your Pydantic model converted to JSON Schema and wrapped in instructions. Nothing magic is happening — it's a string.

### The three experiments

**1. Where does it break?**

```powershell
uv run lessons/01-structured-output/extract.py --file lessons/01-structured-output/sample_ticket.txt
```

A rambling, low-urgency ticket where the customer explicitly says nobody is blocked but a deadline exists. Does your model choose `low` or `normal`? Is `how_to` or `bug` the right category when the docs link is genuinely broken? There's no single correct answer, which is exactly the ambiguity that makes agent evaluation hard — note your own judgement now, because lesson 7 turns these into labelled test cases.

**2. Prompt-only vs json_mode**

```powershell
uv run lessons/01-structured-output/extract.py --compare
```

Compare attempts and token counts. If Ollama reports json_mode as unsupported for your model, that's a real finding: features you depend on aren't uniformly available, which is an argument for keeping the repair loop regardless.

**3. Consistency**

```powershell
uv run lessons/01-structured-output/extract.py --reliability 5
```

Five identical runs, temperature 0. Watch two separate things: whether the *final answers* agree, and how many runs needed a repair.

When this was run on `gpt-oss-120b`, all five produced identical output — but only **3 of 5 succeeded on the first attempt**. A capable model failed schema validation 40% of the time, and the loop hid it completely. That single result is the best argument for the whole pattern.

### Make changes

Small edits with visible consequences:

1. **Weaken a description.** Change `priority`'s description to just `"How urgent it is."` and rerun `--reliability 5`. Consistency usually drops. This is the cheapest demonstration that prompt text in your schema is load-bearing.
2. **Break the repair loop.** Run with `--attempts 1`. On a small model you'll see failures that three attempts would have absorbed.
3. **Force truncation.** In `structured.py`, set `max_tokens=60`. You'll hit `finish_reason == "length"` and see the truncation branch fire. This is a bug you *will* hit in the wild, and it looks nothing like its cause.
4. **Add a field.** Something genuinely hard, like `estimated_refund_amount: float | None`. Watch whether the model invents a number when none is stated. Hallucination under schema pressure is real: a required field is a demand, and models would rather satisfy it than admit ignorance. Prefer optional fields with explicit "use null, do not guess" instructions.
5. **Swap models.** Set `LLM_MODEL=openai/gpt-oss-20b` in `.env` (or `qwen2.5:3b-instruct` locally) and rerun `--reliability 5`. This is your first hard data on the size/reliability trade-off — record it in `NOTES.md`.

---

## Checkpoint

You're done when you can answer these without looking:

- Why isn't a better prompt the right fix for malformed JSON?
- What does `json_mode` guarantee, and what does it not?
- Why does the repair loop feed the *validation error* back rather than just retrying the same request?
- Why is a required field riskier than an optional one?

Then read [NOTES.md](./NOTES.md).

## Files

| File | What it is |
|---|---|
| `structured.py` | The reusable pattern: describe, extract, validate, repair. The real lesson. |
| `extract.py` | The deliverable CLI, the `Ticket` schema, and the three experiments. |
| `sample_ticket.txt` | A deliberately ambiguous ticket. |
| `NOTES.md` | Revision notes. |

**Next:** lesson 02, where the model stops just returning data and starts asking your code to *do* things.
