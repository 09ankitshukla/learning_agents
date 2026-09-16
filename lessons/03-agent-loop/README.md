# Lesson 3 — The agent loop

**Time:** 1.5–2 hours
**You will end with:** an actual agent — one that plans across multiple steps, recovers from its own mistakes, refuses to escape its sandbox, and tells you honestly when it gave up
**Depends on:** lessons 0, 1, 2

---

## Learn first

### Where lesson 2 stopped

Lesson 2's assistant handled exactly one round of tool calls. So this question defeated it:

> What time is it in Mumbai right now, and if I invoice 2,450 USD today, how much is that in INR? Also what is 8.25% of that INR amount?

Three tools, and they're **sequentially dependent**. The model cannot ask for "8.25% of the INR amount" until it has seen what the INR amount is. No prompt fixes that. No larger model fixes it. The information doesn't exist yet at the moment the first request is made.

### The fix is a `while` loop

```python
while not done:
    reply = model(messages)              # THINK
    if not reply.tool_calls:             # the stopping condition
        return reply.text
    for call in reply.tool_calls:        # ACT
        result = dispatch(call)
        messages.append(tool_result(call.id, result))   # OBSERVE
```

That's it. That's an agent. Think, act, observe, repeat until the model stops asking for tools.

It's worth pausing on how little there is here, because the word "agent" carries a lot of mystique. There's no planner, no reasoning engine, no orchestrator. The model decides what to do next by looking at the conversation so far — and the conversation contains the results of everything it has already tried. **The loop doesn't make the model smarter; it gives the model the chance to look at what happened and change its mind.**

Every agent framework you'll meet is this loop plus conveniences. After reading `loop.py` you'll be able to look at any of them and identify which part is this and which part is decoration.

### The stopping condition is the model's decision

Notice what ends the loop: the model returns prose instead of a tool call. **The model decides when it's finished. You only decide when to give up.**

That asymmetry drives the rest of the design.

### What actually needs care

The loop is twenty lines. `loop.py` is 250, and the difference is everything that stops a loop from hurting you.

**A step cap, which is not optional.** An unbounded loop calling a paid API is how a bug becomes an invoice. A model that misunderstands a task will request tools forever, and each iteration re-sends the whole growing conversation. `max_steps` has no "unlimited" setting on purpose.

**A stop reason, not just an answer.** This is the most important API decision in the file:

```python
class StopReason(str, Enum):
    COMPLETED = "completed"        # model returned prose
    MAX_STEPS = "max_steps"        # hit the cap: incomplete
    STALLED = "stalled"            # repeating itself: incomplete
    NO_ANSWER = "no_answer"        # stopped without text
    PHANTOM_TOOL = "phantom_tool"  # asked for a tool never offered
    ERROR = "error"
```

If `run_agent` returned only a string, a caller could not tell a finished answer from a truncated one, and would show a user half a result as though it were complete. Returning *why* it stopped makes that impossible to get wrong by accident.

**Stall detection.** A model requesting an identical call twice running has stopped making progress — it won't learn anything new by repeating itself. Left alone it burns every remaining step. The heuristic is cheap: hash each step's `(tool, arguments)` and stop on a repeat.

**A trajectory.** Every step, every tool call, every result, every token count. This costs almost nothing now and is what lesson 6 asserts against, lesson 7 scores, and lesson 8 traces. Retrofitting observability into a working agent is miserable.

### Two things the loop makes newly dangerous

**Tool results are permanent context.** Every result stays in the conversation and is re-sent on every subsequent call. So one unbounded file read can consume the whole context window and push out the task itself. That's why `read_file` truncates — *and says it truncated*, so the model can ask for the next chunk. Truncating silently makes the model confidently answer from half a file.

**Paths are a way out of your process.** `read_file("../../../../etc/passwd")` is a perfectly well-formed tool call. Lesson 2's calculator was contained by construction; a filesystem tool isn't. The sandbox in `toolset.py` is worth reading closely:

```python
candidate = (SANDBOX / raw).resolve()
if not candidate.is_relative_to(SANDBOX):
    raise ToolError(...)
```

**Resolve first, then check.** Resolving collapses `..` and follows symlinks, so a traversal attempt becomes an absolute path that plainly fails containment. Checking the raw string instead is the classic path-traversal bug — `"../"` is easy to blocklist and easy to smuggle past a blocklist.

`.env` gets a second, separate denylist, and the distinction matters: containment is an *allowlist* that stops traversal, while the denylist expresses "this specific file is secret." It holds your API key, which an agent has no business reading and which you very much do not want pasted into a prompt and shipped to a model provider.

---

## Then apply

### Resolve the cliffhanger

```powershell
uv run lessons/03-agent-loop/agent.py
```

The same question lesson 2 couldn't answer. Observed run:

```
step 1  get_current_time({"timezone": "Asia/Kolkata"})   ok
step 2  convert_currency({2450, USD, INR})               ok
step 3  calculate({"expression": "204330 * 0.0825"})     ok
step 4  no tools requested -> finishing
stop_reason: completed
```

Three dependent steps, then an answer. Nothing changed but the control flow — the tools are imported unchanged from lesson 2.

### Watch it recover from its own mistake

```powershell
uv run lessons/03-agent-loop/agent.py --research
```

The agent searches this repo to answer a question about its own contents. An actual observed run:

```
step 1  search_files({"query": "eval"})                       ok
step 2  read_file({"line_start": 90, ...})    FAILED:bad_signature
step 3  read_file({"start_line": ..., ...})                   ok
step 4  no tools requested -> finishing
```

Step 2 used `line_start` instead of `start_line`. The dispatcher returned `Error: 'read_file' rejected those arguments: unexpected keyword argument 'line_start'`, the model read it, fixed the parameter name, and carried on. **Nobody wrote code to handle that.** It works because lesson 2's dispatcher returns errors as observations instead of raising, and because the loop gives the model another turn.

That is the whole payoff of both lessons in four lines of output.

### The experiments

**1. Step limits.**

```powershell
uv run lessons/03-agent-loop/agent.py --step-limits
```

The same task at `max_steps` 1, 2, 3 and 6. `max_steps=1` reproduces lesson 2 exactly. This is also the honest answer to "what should the cap be?" — it's a property of your task, not a universal constant. Measure your real workload, then set the cap a little above it.

**2. Give it an impossible task.**

```powershell
uv run lessons/03-agent-loop/agent.py --impossible
```

Asks for a live share price. None of the six tools can do it. The agent says so and stops — good behaviour, and it exposes a real limit: `stop_reason` comes back `completed`, because **`COMPLETED` means "the model stopped asking for tools", not "the answer is correct."** A graceful refusal and a correct answer are indistinguishable at this level. You can't fix that inside the loop, since the loop has no notion of what a good answer looks like. That needs task-level scoring — lesson 7.

**3. The budget nudge, and what it costs.**

```powershell
uv run lessons/03-agent-loop/agent.py --max-steps 2
uv run lessons/03-agent-loop/agent.py --max-steps 2 --no-nudge
```

By default the loop warns the model when it has one step left, which reliably turns "ran off the cliff" into "produced a partial answer." Compare:

| | stop_reason | tools used |
|---|---|---|
| with nudge | `completed (under budget pressure)` | `get_current_time` |
| `--no-nudge` | `max_steps` | `get_current_time → convert_currency` |

Better product behaviour, and a real cost: the run now reports `completed` even though the model stopped because we pushed it. So the trajectory records `budget_warned`, and a `COMPLETED` run with that flag deserves less trust than one without. Worth knowing that "helpful" mechanisms can quietly corrupt your success metric.

**4. Try to escape the sandbox.**

```powershell
uv run lessons/03-agent-loop/agent.py --sandbox
```

No model involved — these are hand-built tool calls, the deterministic technique from lesson 2. Five traversal attempts refused, one legitimate read allowed.

**5. Watch context grow.**

```powershell
uv run lessons/03-agent-loop/agent.py --growth
```

Measured on the research task:

| step | prompt tokens | growth |
|---|---|---|
| 1 | 824 | |
| 2 | 1,127 | +303 |
| 3 | 3,466 | +2,339 |
| 4 | 6,122 | +2,656 |

11,539 input tokens for four calls. Input grows every step because the entire conversation, including every tool result, is re-sent each time — nothing accumulates on the server (lesson 0).

So **cost grows roughly with the square of the step count, not linearly.** A 10-step agent isn't 10x a 1-step agent; on input tokens it's closer to 50x. This is the single biggest reason agents surprise people on their bill, and it's what lesson 4 (trimming and summarising) and lesson 8 (measuring) exist to address.

### Make changes

1. **Remove the step cap.** Set `--max-steps 100` and give it a vague, sprawling task. Watch the token count. Then put the cap back and never remove it again.
2. **Break stall detection.** Comment out the `seen_signatures` check and ask something that induces repetition (a search term that appears nowhere, phrased so the model keeps retrying). Compare steps consumed.
3. **Make a tool result enormous.** Raise `MAX_READ_LINES` to 5000 and read a long file. Watch prompt tokens explode and quality drop as the task gets buried.
4. **Assert on a trajectory.** Write five lines checking `trajectory.tool_sequence == ["search_files", "read_file"]`. Run it three times. That's the beginning of lesson 6, and you'll immediately feel why asserting on prose would be hopeless.
5. **Add a tool with a deliberately confusing name.** Add `get_time` alongside `get_current_time` and see which one gets chosen, and how often. Tool naming is part of the prompt.
6. **Widen the sandbox to `/`.** Then run `--sandbox` and see everything succeed. Put it back. Feeling the failure mode once is worth more than reading about it.

---

## Checkpoint

You should be able to answer these without looking:

- What ends the agent loop, and who decides?
- Why must `max_steps` have no "unlimited" option?
- Why return a `StopReason` instead of just the answer string?
- Why does `COMPLETED` not mean "correct"?
- Why does the sandbox resolve the path *before* checking containment?
- Why does agent cost grow faster than linearly with step count?
- Why did the agent recover from its `line_start` mistake without any recovery code?

Then read [NOTES.md](./NOTES.md).

## Files

| File | What it is |
|---|---|
| `loop.py` | **The lesson.** The loop, `StopReason`, `Step`, `Trajectory`, stall detection. |
| `toolset.py` | Lesson 2's three tools (imported, not copied) plus three sandboxed filesystem tools. |
| `agent.py` | The deliverable: live step output, trajectory view, five experiments. |
| `NOTES.md` | Revision notes. |

Note that `dispatch()` now lives in `src/llmkit/tools.py`. Lesson 2 hand-rolled it because writing your own security boundary once is the point; now it's shared infrastructure so later lessons don't rebuild it. Lesson 2's version stays put, unchanged, as the teaching artifact.

**Next:** lesson 04, where that context-growth table becomes the problem to solve.
