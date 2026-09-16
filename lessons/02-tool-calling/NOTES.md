# Lesson 2 — Notes

## The one idea

**The model never executes anything.** It emits a structured *request* and stops. Your code decides whether to run it, runs it, and feeds the result back. "Tool calling" is a misleading name for "the model can ask."

Consequence: the model has no power. Your dispatcher has all of it. An agent's blast radius is a property of your code, not the model's intentions.

## The five steps

```
1. SEND      question + tool descriptions
2. RECEIVE   finish_reason="tool_calls", content=null, a tool_calls array
3. EXECUTE   your code calls the function
4. RETURN    result appended as {"role": "tool", "tool_call_id": ..., "content": ...}
5. ANSWER    model writes prose from the result
```

`tool_call_id` is what pairs a result with its request. Get it wrong and the provider rejects the whole conversation.

## Wire-format details that matter

**`arguments` is a JSON *string*, not an object.** It's model-generated text, so it can be malformed. That's why `ToolCall` carries a `malformed_arguments` field rather than raising at parse time — a model writing bad JSON is a normal runtime event, and if the only way to express it is an exception, your agent can't recover from it.

**`content` is `null` on a tool-calling turn.** The model wrote no prose at all.

**`finish_reason` becomes `tool_calls`** rather than `stop`.

**Tool schemas are re-sent in full on every call.** They are prompt text and they consume context. Ten verbose tools can cost more input tokens than the conversation. "Just add another tool" isn't free.

## Two reasons tools exist

| Reason | Example | Why a prompt can't fix it |
|---|---|---|
| Model **cannot** know | current time, your database, live prices | no clock, no network, weights frozen at training |
| Model is **unreliable** | long multiplication, precise decimals | predicts tokens, doesn't run an ALU |

The second is the sneakier one. "Usually correct" requires spot-checking; "always correct" doesn't.

## The dispatcher is the security boundary

Check order, which is the whole model:

1. Is the name in the registry? → blocks hallucinated tools
2. Did arguments parse as JSON? → blocks malformed output
3. Are required arguments present? → precise error naming the field
4. Does the signature accept them? → catch `TypeError`
5. Did the tool object? → `ToolError`, recoverable
6. Did our code break? → report the type, never the traceback

**`dispatch()` never raises.** Every path returns a string that goes back as an observation.

What's deliberately absent: no `getattr`, no `eval`, no `import`, no string interpolation into a shell. The only route to a function is a dict lookup against names you defined. **The registry *is* the allowlist, by construction** — an invented tool name has nowhere to go.

## Error messages are prompt engineering

The single highest-value line in this lesson:

- `Error: invalid timezone` → dead end
- `Error: Unknown timezone 'Mumbai'. Use an IANA name such as 'Asia/Kolkata'` → the model fixes itself next turn

Verified in scenario 6: given the second message, the model self-corrects with no code from you telling it how. Always name the valid options in the error.

## Never `eval()` model output

```python
def calculate(expression):
    return str(eval(expression))   # remote code execution
```

`expression` is model-generated, and models can be steered by text they read (a document, a web page — lesson 11). So `eval()` is arbitrary code execution in your process.

The fix is an **allowlist**: parse to an AST, permit an explicit set of node types, refuse everything else. It blocks attacks it was never written to anticipate, because attribute access and imports simply aren't in the permitted set. A blocklist of dangerous strings loses to the next encoding trick.

Verified refusals: `__import__('os').system(...)`, `open('.env').read()`, `().__class__.__bases__[0].__subclasses__()`, `9**9**9`.

**Code execution and resource exhaustion are separate problems.** `9**9**9` executes nothing but pins a core and exhausts memory. Fixing one doesn't fix the other; you need an explicit exponent cap too.

## `tool_choice`: who decides

| Value | Meaning | Use when |
|---|---|---|
| `auto` | model decides (default) | almost always |
| `required` | must call some tool | a tool call is the only acceptable outcome |
| `none` | tools visible, must not call | you want the model to know what exists |

Anthropic spells these `auto` / `any` / `none`; the adapter translates.

## Measured on gpt-oss-120b

**The model judges necessity, not just relevance.** With `tool_choice="auto"` it used `calculate` for `91273 * 4482` in all three prompt variants — *including* one whose description was just `"Does math."` It also used it for `2 + 2`.

That last one looks like overcaution until you reread our own description: *"Use this for any calculation rather than working it out yourself."* The model is following it exactly. **When an agent behaves oddly, suspect the text you wrote before you suspect the model.**

**A capable model masks a bad tool description.** The vague variant still worked here. On `gpt-oss-20b` or a local 3B it starts failing. So a single passing run tells you nothing about your description quality — tool selection is probabilistic and belongs in a measured dataset (lesson 7).

**Reasoning tokens dominate tool-calling turns too.** 88 of 119 output tokens (74%) on a turn that produced only a tool request. Two model calls to answer one question, and the second re-sends everything from the first.

## Phantom tool calls: models request tools you never gave them

The most surprising finding of this lesson, discovered by accident.

Asked for the current time with **no tools offered** but a system prompt saying *"use a tool whenever it can give you a fact you cannot know"*, gpt-oss-120b attempted to call tools from its training environment:

- `container.exec` with a bash command running a Python script to read the clock
- `browser.search` with the query "current time Mumbai"

Neither was ever published by us. Groq refused the generation with HTTP 400 `tool_use_failed`.

Change one thing — a system prompt saying *"you have no tools; if you cannot know something, say so"* — and the same model correctly answers "I'm not able to access real-time data," plus offers that Mumbai is UTC+05:30.

**Root cause: a contradiction between the prompt and the tool list**, not the absence of tools. The model was told to use a tool and given none, so it invented one.

Three takeaways:

1. **Prompt and tool list must agree.** If you disable tools at runtime (a feature flag, a degraded mode), update the system prompt in the same change.
2. **Tool-calling behaviour is baked into the weights** and doesn't switch off when you stop offering tools. Models have latent knowledge of tools from training.
3. **Hallucinated tool names are a real runtime event, not a thought experiment.** This is the strongest argument for the dispatcher rejecting unknown names by default.

Provider quirk worth noting: this arrives as HTTP 400, not as a normal response with a tool call, so it raises rather than returning something you can inspect. `llmkit` maps it to `PhantomToolCall` with the attempted call attached. Also, providers disagree about the error body shape — Groq puts the error dict at the top level of `.body` while others nest it under `"error"`, so parse defensively.

## Testing technique worth stealing

To test how your agent handles bad model output, **you don't need a bad model — you need a fake response.** Scenarios 1–4 hand-build the exact `ToolCall` a misbehaving model would emit and push it through `dispatch()`. Deterministic, free, instant, and it covers cases you can't reliably provoke from a real model. Lesson 6 builds this into a test suite.

## The wall this lesson hits

The `--multi-step` question needs three tools *in sequence*: the time, then a currency conversion, then a percentage of the conversion's result. The model can't ask for the percentage until it has seen the conversion.

One round can't do it. The fix isn't a better prompt or a bigger model — it's a `while` loop. That's lesson 3, and hitting the wall yourself is the best possible motivation for it.

## Carry forward

- Validate-and-repair, third appearance: JSON schemas (1), tool arguments (2), guardrails (11).
- The dispatcher's allowlist is the seed of lesson 11's sandboxing.
- Two model calls per question already; a loop multiplies that. Lesson 8 measures the cost.
