# Lesson 3 — Notes

## The one idea

An agent is a `while` loop:

```
THINK   call the model
ACT     run the tools it asked for
OBSERVE append the results to the conversation
        repeat until it stops asking for tools
```

That's the whole definition. There is no planner, no reasoning engine, no orchestrator. **The loop doesn't make the model smarter — it lets the model see what happened and change its mind.**

Every agent framework is this loop plus conveniences.

## Why one round wasn't enough

Lesson 2 failed on sequentially dependent tools: you cannot ask for "8.25% of the INR amount" before you know the INR amount. The needed information doesn't exist when the first request is made. No prompt or model upgrade fixes that — only another turn does.

## The stopping condition

The loop ends when the model returns prose instead of a tool call. **The model decides when it's finished; you only decide when to give up.** That asymmetry shapes everything else.

## Four things that keep a loop safe

**1. A step cap with no "unlimited" option.** A confused model requests tools forever, and every iteration is a billed call re-sending the whole conversation. Default 8. The right value is a property of your task — measure your real workload and set the cap slightly above it.

**2. A `StopReason`, not just an answer.** The most important API decision in `loop.py`. `COMPLETED` / `MAX_STEPS` / `STALLED` / `NO_ANSWER` / `PHANTOM_TOOL` / `ERROR`. If the loop returned only a string, a caller couldn't distinguish a finished answer from a truncated one and would show a user half a result as complete.

**3. Stall detection.** A model repeating an identical `(tool, arguments)` call has stopped making progress and will burn every remaining step. Hash each step's signature, stop on a repeat.

**4. A trajectory record.** Steps, tool calls, results, tokens. Nearly free now; lesson 6 asserts on it, lesson 7 scores it, lesson 8 traces it. Retrofitting observability into a working agent is miserable.

## Assert on the tool sequence, not the prose

`trajectory.tool_sequence` → `["get_current_time", "convert_currency", "calculate"]`

Far more stable across runs than the final text, and it tells you whether the agent *reasoned* correctly even when the wording changes. This is the foundation of lesson 6's testing approach.

## Error recovery, observed live

Unscripted, on the research task:

```
step 2  read_file({"line_start": 90, ...})   FAILED:bad_signature
step 3  read_file({"start_line": ..., ...})  ok
```

The model used the wrong parameter name. The dispatcher returned `Error: 'read_file' rejected those arguments: unexpected keyword argument 'line_start'`. The model read it, corrected the name, continued.

**No recovery code exists for this.** It works because (a) lesson 2's dispatcher returns errors as observations instead of raising, and (b) the loop grants another turn. Two design decisions from two lessons combining into behaviour neither one implements.

## Honest limits

**`COMPLETED` does not mean correct.** It means "the model stopped asking for tools." Asked for a live share price it cannot fetch, the agent explains the limitation and stops — reported as `completed`. A graceful refusal and a correct answer are indistinguishable at this level. Unfixable inside the loop, because the loop has no notion of a good answer. Needs task-level scoring: lesson 7.

**The budget nudge corrupts the success metric.** Warning the model at one-step-remaining reliably converts "ran off the cliff" into "partial answer" — good product behaviour. But the run then reports `COMPLETED` even though the model stopped because we pushed it:

| | stop_reason | tools used |
|---|---|---|
| with nudge | `completed (under budget pressure)` | `get_current_time` |
| without | `max_steps` | `get_current_time → convert_currency` |

So the trajectory records `budget_warned`, and `COMPLETED` with that flag deserves less trust. General lesson: **a helpful mechanism can quietly corrupt your metrics.** Watch for it.

## Context growth: the cost surprise

Measured on the research task:

| step | prompt tokens | growth |
|---|---|---|
| 1 | 824 | |
| 2 | 1,127 | +303 |
| 3 | 3,466 | +2,339 |
| 4 | 6,122 | +2,656 |

11,539 input tokens for four calls.

The entire conversation, including every tool result, is re-sent on every call — nothing accumulates on the server (lesson 0). So **cost grows roughly with the square of the step count.** A 10-step agent isn't 10x a 1-step agent; on input tokens it's closer to 50x.

This is the biggest reason agents surprise people on their bill. Lesson 4 addresses it (trimming, summarising); lesson 8 measures it.

**Corollary: tool results are permanent context.** One unbounded file read can consume the window and push out the task. So `read_file` truncates *and says so*, letting the model request the next chunk. Silent truncation makes a model confidently answer from half a file.

## Sandboxing filesystem tools

Lesson 2's calculator was contained by construction. A path is not — `read_file("../../../../etc/passwd")` is a well-formed call.

```python
candidate = (SANDBOX / raw).resolve()
if not candidate.is_relative_to(SANDBOX):
    refuse()
```

**Resolve first, then check.** Resolving collapses `..` and follows symlinks, so traversal becomes an absolute path that fails containment. Checking the raw string is the classic path-traversal bug: `"../"` is easy to blocklist and easy to smuggle past a blocklist.

Verified refusals: `../../../../etc/passwd`, `..\..\..\Windows\...\hosts`, `C:/Windows/System32/config/SAM`, `.env`, `lessons/../../.env`. `README.md` allowed.

**Two separate controls, don't conflate them:**
- **Containment** = allowlist. Stops traversal. Primary.
- **Denylist** (`.env`, `.git`, `.venv`) = "this specific in-sandbox file is secret." `.env` holds your API key, which an agent must not read and must not paste into a prompt bound for a model provider.

## Multi-tool turns

A model can request several tools in one response. Every one needs a result appended before the next model call, each matched by its own `tool_call_id`. Return two results for three requests and the provider rejects the entire conversation.

## Carry forward

- `dispatch()` moved to `src/llmkit/tools.py`. Lesson 2 hand-rolled it (writing your own security boundary once is the point); it's now shared so later lessons don't rebuild it. Lesson 2's version stays as the teaching artifact.
- `Trajectory` is the input to lessons 6, 7 and 8.
- The context-growth table is lesson 4's problem statement.
- Sandboxing here is the seed of lesson 11's broader guardrails.
