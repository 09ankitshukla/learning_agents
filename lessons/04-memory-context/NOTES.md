# Lesson 4 — Notes

## The one idea

**Context management is a transformation of the message list, applied just before sending.** The agent loop does not change. Lesson 3's loop gained one optional hook (`compactor`) and nothing else.

```python
if callable(compactor):
    messages = compactor(messages, step)
response = client.chat(messages, tools=...)
```

## Why it is needed

The model is stateless (lesson 0), so the whole conversation is re-sent every call. Measured in lesson 3: prompt tokens 824 → 1,127 → 3,466 → 6,122 across four steps. **Cost grows with roughly the square of step count.** Two things eventually break: cost and latency immediately, and the context window as a hard wall.

## The trap: atomic groups

A message list is **not** a flat sequence. An assistant turn carrying `tool_calls` and the `tool` messages answering it are one indivisible unit, paired by `tool_call_id`.

Verified: cutting between them produced HTTP 400 from Groq —

```
failed to template request: ... HarmonyError: EncodingError:
Message=render failed: Tools should have a name!
```

**Naive trimming is not reliably broken, which is what makes it dangerous.** Whether a cut lands on a group boundary is luck, so the bug appears intermittently and looks like a flaky provider rather than your bookkeeping.

So: group first, then drop whole groups. And run a `validate()` pass that checks every `tool` message has a matching earlier `tool_call` — catching it locally beats a 400 three steps into a run.

Also never droppable: the **system prompt** (defines behaviour) and the **first user message** (the task itself). An agent that forgets what it was asked is worse than one that runs out of budget.

## Token estimation: calibrate or be wrong

The first estimator was `chars/3.7 + 4 per message`. Measured against Groq's reported `prompt_tokens` it was **91% too low** — estimated 7 where the provider charged 79.

Two causes:

1. **A large fixed per-request cost.** gpt-oss uses the "harmony" chat template, which injects its own preamble (date, reasoning config, channel markers) before your messages exist. ~70 tokens on every call regardless of content.
2. **Density varies by content type.** JSON, paths and code tokenize ~40% denser per character than prose, because punctuation tokenizes badly.

After correcting both: worst error **+19%**, most within 3%, and the error now *over*-estimates — the safe direction.

**An uncalibrated estimator is not conservative, it is simply wrong, and wrong in the direction that overflows your window.** Re-run `--calibrate` on any model or provider change.

Hierarchy of accuracy: provider `usage.prompt_tokens` (exact, but only after the call) > the model's real tokenizer (tiktoken / AutoTokenizer, and it must match the model) > character estimate (free, needs calibration). Use estimates to decide what to send, actuals to check yourself.

## Four strategies, measured

Same conversation, 2,711 estimated tokens:

| Strategy | Tokens | % of original | Trade-off |
|---|---|---|---|
| do nothing | 2,711 | 100% | baseline |
| compress tool results | 1,429 | 53% | **no model call** |
| safe trim | 1,036 | 38% | forgets detail |
| summarise | 1,003 | 37% | costs an extra model call |

**Compressing tool results is the best first move.** It halved the conversation with no model call and kept every step's structure intact. In lesson 3's growth table the two big jumps (+2,339 and +2,656) were both single `read_file` results — nothing else came close.

Policy, cheapest-first (what `ContextManager` does):

1. Under budget? Do nothing. Compaction is not free.
2. Compress oversized tool results.
3. Only then trim or summarise.

## Summarisation has a break-even point

Measured instance: saved 45 tokens, **cost 1,493 tokens to produce**. A net loss.

You pay once and save on every subsequent call, so summarisation only pays off if many calls follow. Summarise near the end of a run and it is pure loss. Compressing tool results has no break-even — it costs nothing but CPU.

Two more cautions:

- **Summaries compound.** Summarise a conversation that already contains a summary and detail decays geometrically. Fold the previous summary in rather than stacking.
- **The loss is not uniform.** Exact values, ids and quotes are precisely what a summary drops and precisely what an agent needs later. Ask for facts and identifiers explicitly.

## My own summariser starved

The first version requested `max_tokens=400` and reliably returned an **empty string** against gpt-oss. Cause: reasoning models spend the output budget on hidden deliberation first. Lesson 0 documented this exact trap and this function still walked into it — which shows how well it hides, since the call succeeds and tokens are billed.

Raised to 1,200 and it works. **Any model call you add to your infrastructure needs the same budget headroom as your agent's calls.**

The fallback was also broken: on empty summary it trimmed to a hardcoded 10,000-token budget, which was above the conversation size, so it dropped nothing and reported success. **A fallback that silently does nothing is worse than no fallback.** It now uses the caller's real budget.

## Trimming makes the agent redo work

The safe-trimmed conversation was accepted, and the model immediately re-requested a tool whose result had just been deleted. Correct behaviour, and the real price: **you trade context tokens for repeated tool calls.** Sometimes a good trade, sometimes a loop — one run ended `stalled` because compaction kept removing what the agent needed.

## Budgets have a floor

`trim_safe` can be given a budget it cannot reach. Protected content (system prompt + task + recent exchanges) may already exceed it. Observed: a 150-token budget against a 124-token system prompt.

It now reports a `FLOOR` note instead of silently returning something too big. The fix is a shorter system prompt, fewer protected exchanges, or summarisation — not more trimming.

## Measured before/after

Same task, budget 900:

| | unmanaged | managed | change |
|---|---|---|---|
| input tokens | 9,726 | 6,747 | **−31%** |
| total tokens | 10,283 | 7,147 | −30% |
| input tokens / step | 1,621 | 1,124 | **−31%** |

Compare **input tokens per step**, not totals. Totals move with step count, which varies run to run for reasons unrelated to context management.

**And check the mechanism fired.** An earlier run at a 2,500-token budget reported a plausible-looking table with `0 compactions` — the budget was never crossed, so the table compared nothing but ordinary variance. A harness that produces believable numbers when the thing under test never ran is how false conclusions get made.

Latency is not comparable on a free tier: wall-clock is dominated by queue waiting. Compare tokens, which are deterministic given the same messages.

## Persistence

**The message list is the entire state of an agent**, so "save a session" is `json.dump` and "resume" is read it back and keep appending. That is genuinely the whole idea.

Three things that go wrong:

1. **Tool calls must round-trip exactly.** `tool_call_id` pairs a result with its request. Messages are plain JSON-serialisable dicts throughout this repo specifically so this is free — but only if you never "helpfully" normalise on the way in or out.
2. **A saved session is untrusted input.** It is a file that a process reads and sends to a model. If something else can write it, something else controls your agent's instructions. `load_session` validates structure and refuses malformed files.
3. **A resumed session is already long.** It reloads at full size, so compaction belongs *before* the first new call.

**Sizing rule, learned by breaking it:** a resumed session's budget must exceed the session's own size, with headroom. A 1,200-token budget against a 2,645-token session compacted away the findings the follow-up was asking about; the agent re-ran searches to rediscover what it already knew and ran out of steps. Over-aggressive compaction does not merely lose detail — it destroys the premise of the question.

Saved sessions contain whatever tools returned, including file contents. Treat them as sensitive; `runs/` is gitignored.

## Carry forward

- Two optional parameters were added to lesson 3's `run_agent`: `compactor` and `initial_messages`. Both default to no-ops, so lesson 3 behaves identically.
- `validate()` belongs in the test suite — lesson 6.
- "Did quality survive compaction?" cannot be answered by eyeballing two answers. That is lesson 7.
- Token accounting here becomes cost accounting in lesson 8.
