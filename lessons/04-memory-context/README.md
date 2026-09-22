# Lesson 4 — Memory and context management

**Time:** 1.5–2 hours
**You will end with:** an agent that keeps itself inside a token budget, a calibrated token estimator, and a session you can save and resume across processes
**Depends on:** lessons 0, 1, 2, 3

---

## Learn first

### The problem, already measured

Lesson 3 ended with a table rather than a cliffhanger. Prompt tokens across four steps of the research task:

| step | prompt tokens | growth |
|---|---|---|
| 1 | 824 | |
| 2 | 1,127 | +303 |
| 3 | 3,466 | +2,339 |
| 4 | 6,122 | +2,656 |

11,539 input tokens to answer one question. The cause is lesson 0's first mechanic: **the model is stateless, so the entire conversation is re-sent on every call.** Nothing accumulates on the server.

So cost grows with roughly the **square** of the step count. A 10-step agent is nearer 50x a 1-step agent on input tokens, not 10x. Two things eventually break — cost and latency immediately, and the context window as a hard wall, where crossing it gives you an error or, worse, silent truncation that looks like the agent suddenly forgetting its task.

### The shape of the fix

Context management is **a transformation of the message list, applied just before sending**. The loop does not change:

```python
if callable(compactor):
    messages = compactor(messages, step)     # <- the only new line
response = client.chat(messages, tools=registry.specs)
```

Lesson 3's `run_agent` gained exactly one optional hook for this. That is worth noticing: something that sounds architectural turns out to be a function applied to a list.

### The trap that catches everyone

A message list looks like a flat sequence. It is not.

An assistant turn carrying `tool_calls` and the `tool` messages answering it are **one indivisible unit**, paired by `tool_call_id`. Cut between them and the provider rejects the entire request. Verified against Groq:

```
HTTP 400: failed to template request: ... HarmonyError:
Message=render failed: Tools should have a name!
```

Here is what makes it genuinely dangerous: **naive trimming is not reliably broken.** Whether your cut lands on a group boundary is luck. So the bug appears intermittently, in production, and looks like a flaky provider rather than your own bookkeeping.

Two defences, both in `context.py`:

- **Group before trimming.** Partition the conversation into atomic units, then drop whole units.
- **Validate afterwards.** `validate()` checks that every `tool` message has a matching earlier `tool_call`. Catching this locally is far cheaper than a 400 three steps into a run.

Also never droppable: the **system prompt** (it defines behaviour) and the **first user message** (it *is* the task). An agent that forgets what it was asked is worse than one that runs out of budget.

### You cannot count tokens without the model's tokenizer

And this bites harder than it sounds. The first estimator in this lesson was `chars/3.7 + 4 per message`. Measured against what Groq actually charged, it was **91% too low** — 7 estimated against 79 billed.

Two causes, neither visible without measuring:

1. **A large fixed per-request cost.** gpt-oss is served with the "harmony" chat template, which injects its own preamble before your messages exist. About 70 tokens on every call, whatever you send.
2. **Density varies with content.** JSON, file paths and code tokenize roughly 40% denser per character than prose, because punctuation tokenizes badly.

Correcting both took the worst error to **+19%**, with most samples within 3%, and crucially the remaining error now *over*-estimates — the safe direction.

**An uncalibrated estimator is not conservative, it is just wrong, and wrong in the direction that overflows your context window.**

Accuracy hierarchy: the provider's `usage.prompt_tokens` is exact but only arrives *after* the call; the model's real tokenizer (tiktoken, `transformers.AutoTokenizer`) is accurate but must match the model or it lies confidently; a character estimate is free and needs calibrating. Use estimates to decide what to send, actuals to check yourself.

### Four strategies, and their prices

| Strategy | Result | Price |
|---|---|---|
| do nothing | 2,711 tokens | baseline |
| compress tool results | 1,429 (53%) | **no model call** |
| safe trim | 1,036 (38%) | forgets detail |
| summarise | 1,003 (37%) | an extra model call |

**Compress tool results first.** It halved the conversation for free and kept every step's structure intact. Look back at lesson 3's growth table — both big jumps were single `read_file` results. That is where the tokens are.

So `ContextManager` orders its policy cheapest-first: under budget, do nothing; over budget, compress tool results; still over, trim or summarise.

And summarisation has economics people get backwards. One measured instance **saved 45 tokens and cost 1,493 to produce.** You pay once and save on every later call, so it only pays off if many calls follow. Summarise near the end of a run and it is a straight loss.

---

## Then apply

### See where the tokens go

```powershell
uv run lessons/04-memory-context/agent.py
```

Per-message token accounting for a real run. The `tool` role almost always dominates, and usually one or two messages account for most of it.

Watch for one subtlety: assistant messages that request tools have `content: null` but are **not** free — their tool_call arguments are real tokens. A naive counter measuring only `content` scores them zero.

### The experiments

**1. Break it on purpose.**

```powershell
uv run lessons/04-memory-context/agent.py --orphan
```

Builds a real conversation, shows its atomic groups, then finds a naive cut that splits one and sends it to the provider. You get a genuine HTTP 400. Then the same conversation trimmed safely, which is accepted.

Three things to notice: the naive cut is only *sometimes* invalid; the trim reports a `FLOOR` when the budget is unreachable; and the safely-trimmed agent immediately re-requests a tool whose result was deleted.

**2. Calibrate the estimator.**

```powershell
uv run lessons/04-memory-context/agent.py --calibrate
```

Four samples, estimate versus what the provider charged. This is the experiment that caught the 91% error. Re-run it on any model change.

**3. Compare strategies.**

```powershell
uv run lessons/04-memory-context/agent.py --strategies
```

All four on one conversation, with token counts and trade-offs.

**4. The measured before/after.**

```powershell
uv run lessons/04-memory-context/agent.py --compare --budget 900
```

| | unmanaged | managed | change |
|---|---|---|---|
| input tokens | 9,726 | 6,747 | **−31%** |
| input tokens / step | 1,621 | 1,124 | **−31%** |

Two methodology points that matter more than the numbers.

Compare **input tokens per step**, not totals — totals move with step count, which varies run to run for unrelated reasons.

And **check the mechanism actually fired.** An earlier version of this experiment used a 2,500-token budget that was never crossed, and cheerfully printed a comparison table showing `0 compactions`. Every difference in it was ordinary variance. A harness that produces believable numbers when the thing under test never ran is how false conclusions get made. The script now refuses to interpret a void result.

**5. Save and resume.**

```powershell
uv run lessons/04-memory-context/agent.py --save runs/demo.json
uv run lessons/04-memory-context/agent.py --resume runs/demo.json
```

The second command runs in a **different process** and continues the conversation, answering a follow-up from work the first process did. It needs no tools to do it, because the findings are in the message list.

That is the whole of persistence: the message list is the entire state of an agent, so saving is `json.dump`.

### Make changes

1. **Set an impossible budget.** `--compare --budget 200`. The system prompt alone exceeds it. Watch the `FLOOR` note, and confirm the code tells you rather than silently returning something too big.
2. **Break the estimator deliberately.** Set `REQUEST_OVERHEAD = 0` in `context.py` and rerun `--calibrate`. You will see the original 91% error reappear. That constant is doing real work.
3. **Disable grouping.** Use `trim_naive` in `ContextManager` instead of `trim_safe` and run `--compare`. You should get intermittent 400s — which is exactly the production symptom.
4. **Make summarisation starve.** Set `max_summary_tokens=300` in `summarise_history` and run `--strategies`. It returns empty and the fallback fires. This is lesson 0's reasoning-token trap, and it caught me while writing this lesson.
5. **Resume with too small a budget.** `--resume` after editing `resume_budget` down to 800. The agent deletes the findings it just loaded and re-searches for what it already knew.
6. **Compress harder.** Drop `tool_result_max_chars` to 150 and read the answers. Find the point where saving tokens starts costing correctness — that boundary is the whole game.

---

## Checkpoint

You should be able to answer these without looking:

- Why does an agent's cost grow faster than linearly with step count?
- What makes a group of messages atomic, and what happens if you split one?
- Why is naive trimming *more* dangerous than reliably broken code?
- Why was the token estimator 91% wrong, and why is over-estimating safer?
- Which compaction strategy would you reach for first, and why?
- When does summarisation cost more than it saves?
- What is the entire state of an agent, and what does that make "resume"?
- Why must a resumed session's budget exceed the session's size?

Then read [NOTES.md](./NOTES.md).

## Files

| File | What it is |
|---|---|
| `context.py` | **The lesson.** Measuring, grouping, validating, and four strategies. |
| `session.py` | Save and resume. Small on purpose — it is `json.dump` plus validation. |
| `agent.py` | The deliverable: five experiments and the measured comparison. |
| `NOTES.md` | Revision notes. |

Lesson 3's `run_agent` gained two optional parameters for this lesson — `compactor` and `initial_messages` — both no-ops by default, so lesson 3 behaves exactly as before.

**Next:** lesson 05, retrieval. `search_files` in lesson 3 was keyword-only, so you had to guess the exact wording. Embeddings fix that, and the notes you have been accumulating become the corpus.
