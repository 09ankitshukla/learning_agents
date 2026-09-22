# Key learnings index

The condensed version. Read this to revise quickly, or if you just want the ideas without running anything.

Each entry links to the lesson's full notes.

---

## Lesson 00 — Setup · [notes](../lessons/00-setup/NOTES.md)

**Three layers, often conflated:** the *model* is a file of weights (inert), the *inference server* runs it and consumes your CPU/GPU, the *client* is your code sending HTTP. Nearly every server copied OpenAI's API, which is why swapping models is config rather than code.

**Model calls are stateless.** No session exists on the server. You resend the whole conversation every turn. "Memory" is a list you maintain. This one fact explains growing cost, growing latency, and context-limit errors.

**Tokens are the unit of cost, latency, and memory.** ~¾ of a word each.

**Temperature 0 is not determinism.** Server batching changes floating-point order. So never assert exact model text in a test.

**`finish_reason`** tells you whether the model finished (`stop`) or got cut off (`length`). Truncation masquerades as model incompetence.

**Quantisation** (`Q4_K_M`) compresses weights to ~4 bits, cutting memory ~4x for a small quality loss. For a fixed memory budget, a bigger model at 4-bit usually beats a smaller one at 8-bit.

**Tool calling degrades faster than chat quality as models shrink.** Below ~7B, expect defensive parsing and retries.

**Reasoning models spend most of their output budget thinking.** GPT-OSS, DeepSeek-R1, o-series and Qwen3-thinking generate hidden deliberation before answering. It's billed as output, it eats `max_tokens`, and it's absent from the reply text. Measured: 141 of 151 output tokens (93%) for a one-word answer.

**Starvation is the failure mode to recognise.** Too small a budget and a reasoning model returns an *empty string* with `finish_reason="length"` — request succeeded, tokens billed, no answer. Looks like a refusal; is actually a budget. Keep `max_tokens` ≥ 500 even for one-word answers.

**`reasoning_effort` changes the answer, not just its length.** Different efforts produced different categories for the same input. Hold it fixed when comparing anything else.

**Hosted model IDs expire.** A model that was the sensible default while lesson 0 was written was retired before it was first run. Query the provider for the live list; never trust a model name in documentation.

**On Windows, model output can crash your script at print time.** Typographic quotes and non-breaking hyphens don't exist in cp1252 → `UnicodeEncodeError`, far from the model call. Fix needs both UTF-8 streams *and* console code page 65001, or you trade a crash for mojibake.

---

## Lesson 01 — Structured output · [notes](../lessons/01-structured-output/NOTES.md)

**The one idea: a model is a component with a failure rate, not a parser.** Don't prompt harder — validate the output and retry with feedback, the way you would with any flaky dependency.

**The pattern, reused for the rest of the course:**

```
DESCRIBE -> EXTRACT -> VALIDATE -> REPAIR (loop back)
```

Generate the schema from your type so prompt and validation can't drift. Extract JSON with a brace-depth scan, not a regex. Validate with Pydantic. On failure, append the bad reply *and* the specific error, then ask again — the model corrects itself when given precise feedback.

**Your schema is a prompt.** Field names, descriptions and enum values are all sent to the model. Weakening a description measurably reduces consistency.

**Constrain the output space.** Enums over free text.

**Optional beats required when data may be absent.** A required field pressures the model to invent a plausible value rather than admit ignorance.

**`json_mode` guarantees valid JSON, never correct JSON.** Wrong enums, missing fields and hallucinations all pass it. Keep the repair loop.

**Return metadata, not just the value.** Track attempts, errors and token usage from the start. An extractor that always succeeds on attempt 3 is a signal you can't see otherwise.

**Measured, and the reason the whole pattern exists:** on `gpt-oss-120b`, five identical runs produced identical final output — but only **3 of 5 succeeded on the first attempt**. A strong model failed schema validation 40% of the time and the repair loop hid it completely. `json_mode` cut that to 1 attempt and 1,716 tokens versus 2 attempts and 3,044.

---

## Lesson 02 — Tool calling · [notes](../lessons/02-tool-calling/NOTES.md)

**The one idea: the model never executes anything.** It emits a structured *request* and stops. Your code decides whether to run it. "Tool calling" is a misleading name for "the model can ask."

**So the model has no power; your dispatcher has all of it.** An agent's blast radius is a property of your code, never the model's intentions.

**The five steps:** send question + tool descriptions → receive a request (`finish_reason="tool_calls"`, `content=null`) → your code executes → result returns as a `role: "tool"` message paired by `tool_call_id` → model writes prose.

**`arguments` is a JSON string, not an object.** It's model-generated text, so it can be malformed. Represent that in your data model rather than raising, or your agent can't recover from it.

**Tool schemas are prompt text, re-sent in full on every call.** Ten verbose tools can outweigh the conversation. Adding a tool isn't free.

**`dispatch()` must never raise.** Every failure returns a string that goes back as an observation. Check order: known name → valid JSON → required args → signature → domain error → internal error.

**Error messages are prompt engineering.** `Error: invalid timezone` is a dead end; `Use an IANA name such as Asia/Kolkata` gets self-corrected next turn. Verified. Always name the valid options.

**Never `eval()` model output.** It's arbitrary code execution, reachable via text the model read. Parse to an AST and allowlist node types — that blocks attacks you didn't anticipate, because attribute access and imports simply aren't permitted. Code execution and resource exhaustion (`9**9**9`) are separate problems needing separate limits.

**`tool_choice`** decides who chooses: `auto` (model decides), `required` (must call something), `none` (visible, forbidden). Use `required` when a tool call is the only acceptable outcome.

**Phantom tool calls are real.** Given no tools but a prompt insisting on tool use, gpt-oss-120b tried to call `container.exec` and `browser.search` — tools from its training, never published by us. Root cause was the contradiction between prompt and tool list, not the missing tools. Keep the two in sync.

**When an agent behaves oddly, suspect the text you wrote.** The model used the calculator for `2 + 2` because our description said "use this for any calculation." It was obeying, not misjudging.

**A capable model masks a bad tool description.** The vague `"Does math."` variant still worked on a 120B model and fails on smaller ones. One passing run proves nothing about description quality.

**To test bad model output, you don't need a bad model — you need a fake response.** Hand-build the `ToolCall` a misbehaving model would emit and push it through your dispatcher. Deterministic and free.

## Lesson 03 — The agent loop · [notes](../lessons/03-agent-loop/NOTES.md)

**The one idea: an agent is a `while` loop.** Think (call the model) → act (run requested tools) → observe (append results) → repeat until it stops asking for tools. No planner, no orchestrator. **The loop doesn't make the model smarter; it lets the model see what happened and change its mind.** Every framework is this plus conveniences.

**The model decides when it's finished. You only decide when to give up.**

**`max_steps` has no "unlimited" setting.** A confused model requests tools forever and each iteration is a billed call re-sending the whole conversation. The right cap is a property of your task.

**Return a `StopReason`, not just an answer.** `COMPLETED` / `MAX_STEPS` / `STALLED` / `NO_ANSWER` / `PHANTOM_TOOL`. A caller receiving only a string can't tell a finished answer from a truncated one, and will show a user half a result as complete.

**`COMPLETED` means "stopped asking for tools", not "correct."** A graceful refusal completes too. Unfixable inside the loop — needs task-level scoring (lesson 7).

**Detect stalls.** An identical repeated `(tool, args)` call means no progress; left alone it burns every remaining step.

**Assert on `tool_sequence`, not prose.** Far more stable across runs, and it shows whether the agent reasoned correctly even when wording varies.

**Error recovery is emergent, not coded.** Observed: the model called `read_file(line_start=...)`, got `unexpected keyword argument`, and fixed it next step. No recovery code exists — it works because the dispatcher returns errors as observations (lesson 2) and the loop grants another turn.

**Cost grows ~quadratically with step count.** Measured: prompt tokens 824 → 1,127 → 3,466 → 6,122 across four steps, 11,539 input tokens total. The whole conversation is re-sent every call. A 10-step agent is nearer 50x a 1-step agent, not 10x. This is why agents surprise people on their bill.

**Tool results are permanent context.** One unbounded file read can push the task out of the window. Truncate *and say so* — silent truncation makes a model answer confidently from half a file.

**Sandbox paths by resolving first, then checking containment.** `resolve()` collapses `..` and follows symlinks, so traversal becomes an absolute path that fails `is_relative_to(SANDBOX)`. Checking the raw string is the classic bug. Keep containment (allowlist, stops traversal) separate from a secrets denylist (`.env` holds your API key — an agent must not read it or paste it into a prompt).

**A helpful mechanism can corrupt your metrics.** Warning the model near the step limit produces better answers *and* turns an honest `MAX_STEPS` failure into a `COMPLETED`. Record that it happened (`budget_warned`) so success stays trustworthy.

## Lesson 04 — Memory and context · [notes](../lessons/04-memory-context/NOTES.md)

**The one idea: context management is a transformation of the message list, applied just before sending.** The loop does not change — lesson 3's `run_agent` gained one optional hook.

**A message list is not flat.** An assistant turn with `tool_calls` plus the `tool` messages answering it is an **atomic group**, paired by `tool_call_id`. Split one and the provider returns 400. Verified: `HarmonyError: render failed: Tools should have a name!`

**Naive trimming is not reliably broken, which is what makes it dangerous.** Whether a cut lands on a group boundary is luck, so the bug is intermittent and looks like a flaky provider. Group before trimming, and `validate()` afterwards.

**Never droppable:** the system prompt (defines behaviour) and the first user message (*is* the task).

**Calibrate your token estimator or it will be wrong in the dangerous direction.** `chars/3.7 + 4` measured **91% too low** (7 estimated, 79 charged). Causes: a ~70-token fixed per-request cost from the chat template, and JSON/code tokenizing ~40% denser than prose. After correction: +19% worst case, and it now over-estimates, which is safe.

**Compress tool results first.** Measured on one conversation: 2,711 → 1,429 tokens (53%) with **no model call**. Trimming reached 38% but forgets; summarising reached 37% but costs a call. Order your policy cheapest-first.

**Summarisation has a break-even point.** One instance saved 45 tokens and cost 1,493 to produce. You pay once and save on every later call, so it only pays off if many calls follow. Summaries also compound — detail decays geometrically if you summarise a summary.

**Trimming makes the agent redo work.** A safely-trimmed agent immediately re-requested a tool whose result was deleted. You trade context tokens for repeated tool calls; sometimes that is a loop.

**Budgets have a floor.** Protected content can exceed the budget you asked for. Report that rather than silently returning something too big.

**Measured before/after** (budget 900): input tokens 9,726 → 6,747, **−31%**; per step 1,621 → 1,124. Compare per-step, not totals, since totals move with step count.

**Verify the mechanism fired.** An earlier run printed a plausible comparison table with `0 compactions` — the budget was never crossed, so it compared nothing but variance. A harness that looks credible when the thing under test never ran is how false conclusions get made.

**The message list is the entire state of an agent**, so saving a session is `json.dump` and resuming is reading it back. A resumed session is already full-size, so compact *before* the first new call — and its budget must exceed the session's own size, or you delete the work you just loaded.

**My own summariser starved on 400 max_tokens** and returned empty, walking straight into lesson 0's reasoning-token trap. Infrastructure model calls need the same budget headroom as the agent's. And its fallback silently did nothing because of a hardcoded budget — a fallback that quietly no-ops is worse than none.

## Lesson 05 — Retrieval

*Not yet written.*

## Lesson 06 — Testing

*Not yet written.*

## Lesson 07 — Evaluation

*Not yet written.*

## Lesson 08 — Judging and tracing

*Not yet written.*

## Lesson 09 — Iteration

*Not yet written.*

## Lesson 10 — Multi-agent

*Not yet written.*

## Lesson 11 — Guardrails and failure modes

*Not yet written.*

## Lesson 12 — Deployment

*Not yet written.*

## Lesson 13 — Framework comparison

*Not yet written.*

---

## Recurring themes

Things that keep showing up. Watch them accumulate.

**Validate and repair.** Lesson 1 for JSON, lesson 2 for tool arguments, lesson 3 for runtime tool errors, lesson 11 for guardrails. Same shape every time.

**Non-determinism is the defining engineering constraint.** It shapes testing (6), evaluation (7), and how you judge any change (9). One good run proves nothing.

**Measure from the beginning.** Tokens, latency and attempt counts are carried in the return types from lesson 1. Retrofitting observability into a working agent is much harder than building with it.

**The whole thing rests on one method.** `client.chat(messages, tools) -> LLMResponse`. Every capability in this course is that call plus ordinary Python. When a framework looks magical, ask what it's doing around this method.
