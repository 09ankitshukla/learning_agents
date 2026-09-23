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

## Lesson 05 — Retrieval · [notes](../lessons/05-retrieval/NOTES.md)

**The one idea: similarity search is a dot product followed by a sort.** Embed documents once, embed the query, multiply, take top k. No vector database needed for a few hundred chunks. Normalising vectors to unit length is what reduces cosine similarity to a plain dot product.

**Retrieval is the answer to lesson 4's problem.** Lesson 4 salvaged a full context window; retrieval avoids filling it. Measured: the corpus is ~33,000 tokens, retrieving top 3 sent 883 — **2.2%**. But stuffing cannot miss and retrieval can, so accuracy matters.

**Measured on 193 chunks, 10 paraphrased queries:**

| method | top-1 | recall@4 |
|---|---|---|
| keyword | 0/10 | 7/10 |
| semantic | 7/10 | 8/10 |
| hybrid | 7/10 | 9/10 |

**recall@k matters more than top-1 for an agent**, because it reads all k results. Keyword's 0/10 top-1 looks catastrophic until you see 7/10 recall@4.

**Absolute similarity scores are nearly meaningless.** Cosine sits in a 0.6–0.8 band even for bad matches. Only ranking and the first-to-second gap inform. A fixed threshold like "above 0.75" is almost impossible to tune.

**Two measurement bugs, and they are the real lesson.** Labels matched only file paths, so a correct hit on `docs/00-index.md > … > Lesson 02` scored as a miss — **your ground truth encodes assumptions, and an apparent miss is sometimes a better answer than the one you labelled.** Then the identical bug in a second experiment produced 3/10 where the first gave 7/10, caught *only* because two measurements of the same thing disagreed. **An eval that is subtly wrong is more dangerous than no eval.**

**A finding that did not replicate.** Embedding headings with chunk bodies gained 4/7→6/7 on a hand-written 8-chunk probe and **nothing** on the real 193-chunk corpus. Small hand-picked samples have no competing documents and uniformly good headings. **A result from a tiny sample is a hypothesis, not a finding.**

**Hybrid is not automatically better.** The two score scales are incompatible (cosine narrow, IDF unbounded), and min-max normalisation is outlier-sensitive, so one strong keyword match can drag an unrelated chunk above a correct semantic hit. Rank-based fusion is the standard fix.

**A summary document is an attractor.** `docs/00-index.md` condenses everything, so it matches most queries reasonably and crowds out detailed sources. Like an FAQ outranking the real docs.

**429 and 413 are different problems.** 429 means wait. **413 means this request will never fit** — waiting cannot help; reduce what you send. Retrieval plus seven tool schemas plus history hit 10,020 tokens against an 8,000 TPM ceiling. Fixed by using lesson 4's `ContextManager` unchanged: lessons compose.

**Over-compression destroys retrieval.** Lesson 4's 600-char default crushed a 6,195-token search result to 765 and deleted what the search had just found; the agent re-searched five times. **Compression thresholds depend on how much of a tool's output is signal.** Bound results at source instead.

**An agent with a search tool will search forever** when the answer is absent, rephrasing each time so stall detection never fires. Tell it explicitly when to stop.

**Keyword search is not obsolete.** It scored 0/10 only because every query was paraphrased. For a literal string — function name, error message, identifier — it is exact where semantic is approximate. Keep both.

## Lesson 06 — Testing · [notes](../lessons/06-testing/NOTES.md)

**The one idea: "agents can't be tested because they're non-deterministic" is half true, and believing it leads to testing nothing.** The dispatcher, sandbox, calculator, trimming, estimator, chunking and keyword search are all ordinary deterministic code — and all of the security and correctness properties live there.

**Substitute at the seam, not at HTTP.** Mocking `httpx` tests the SDK, not your agent. `LLMClient.chat()` is the seam; anything with that shape is a drop-in model. That Protocol was written in lesson 0 for provider-swapping and turns out to be what makes the project testable. **Designing a substitutable boundary before you need one is most of what makes code testable later.**

**Two doubles, two jobs.** *Scripted* clients let you write the responses — the only way to test token starvation, malformed tool arguments or a phantom tool on demand. *Cassettes* replay real recordings and preserve quirks you would not think to fake (`content: null` on tool turns, reasoning-token counts). Use both; when they disagree, the cassette is right.

**Assert on the trajectory, never on prose:** `stop_reason`, then `tool_sequence`, then **the requests your double received**, and almost never the final text. Many agent bugs live in what you *send* — so the double must record requests, and must copy them because the loop mutates its own list.

**Test your doubles.** Two bugs in mine: the cassette replayed entry 0 twice (key lookups and sequential fallback used separate counters) producing a false `STALLED`; and a double that silently repeats its last response lets a runaway 40-iteration loop pass as green. **A lying fake is worse than no test.**

**Live tests opt-in, and few.** 111 offline tests run in under a second; 8 live tests are deselected by default. **A suite you avoid running because it's expensive provides no safety.** Live tests should check provider assumptions a double cannot — does the model still exist, does tool calling still work — and be tolerant, catching structural breaks rather than a few percent drift.

**Property-shaped beats example-shaped.** `for budget in range(...): assert validate(trim_safe(...)) == []` would have caught lesson 4's orphan bug directly.

**Keep counterexample tests.** One test asserts that the deliberately-broken `trim_naive` *is* broken, so nobody "fixes" it and silently destroys lesson 4's demonstration.

**What the suite found immediately: two classes named `ToolError`.** Lesson 2 declared its own; lesson 3's promoted dispatcher caught `llmkit`'s. So every timezone and currency error in lessons 3–5 was reported as `failed unexpectedly` instead of its actionable message, silently undoing the self-correction lesson 2 demonstrates. **Invisible to inspection** — same name, every call site reads correctly. Found by asserting on error text.

**Then I broke this lesson's own rule.** A live test asserted `"6319" in final_answer` after stripping commas; the model wrote LaTeX `6{,}319` → `6{}319`. Arithmetic perfect, assertion wrong. Now checks the tool result instead.

**The regression suite is the honest documentation.** One test per bug actually shipped. Each cost real debugging time; each test costs milliseconds and runs forever.

## Lesson 07 — Evaluation · [notes](../lessons/07-evaluation/NOTES.md)

**The one idea: lesson 6 asks "does the machinery work?", lesson 7 asks "is the agent any good?".** `COMPLETED` only means the model stopped asking for tools. Telling a correct answer from a graceful refusal or a confident fabrication needs cases with **expected outcomes**.

**An eval case needs a `why`.** A dataset without rationale rots — six months on nobody knows if a case is load-bearing, and nobody dares delete it either.

**Include cases the agent should refuse.** A set of only solvable tasks rewards confident guessing. `does_not_contain` / `does_not_match` are the fabrication guards most eval sets lack, and a case that only checks for the right answer cannot distinguish "declined correctly" from "invented something plausible".

**Deterministic scorers before judges.** Free, instant, reproducible, unbiased. Exhaust them before letting a model grade a model (lesson 8).

**Normalise aggressively or you measure formatting.** A scorer checking `"6319"` fails on LaTeX `6{,}319`. Strip separators, braces, typographic punctuation; extract *all* numbers and ask if the right one is present. **A scorer that is too strict measures formatting instead of correctness, and you will not notice, because the failures look real.**

**Substring matching is too blunt for fabrication checks.** `"$1"` is a substring of `"$185"`, so forbidding it also fails a good refusal saying "it is not $1 or any other figure". Forbid a *shape* with a regex instead.

**Cache the execution, never the score.** Got this wrong first: caching the scored result meant a newly added scorer never ran, and the suite reported a pass for a case that should have failed — confidently wrong and silent. Cache the expensive stable thing (the agent run), recompute the cheap volatile thing (the score) every time. After the fix, changing a scorer re-scored all 16 cases for **zero tokens**.

**Never cache errors.** A rate limit is not a finding about the agent; caching one poisons every later run.

**Caching is a correctness feature, not an optimisation.** Without it, re-running a baseline gives *different* baseline numbers, so you attribute model variance to your change.

**Measured: the "better" prompt was worse.** Baseline 15/16 (94%); a stricter prompt saying "always use a tool, never guess" scored 14/16 (88%) and broke a passing case. Verdict `REGRESSION`. Without the harness I would have shipped it, because it reads better.

**Right answer, wrong process.** `currency_unsupported` fails because the agent declined *without calling the tool* — correct outcome, unjustified reasoning, and the same reasoning would wrongly refuse a supported currency. **High success with low tool-choice accuracy means the agent is right by accident.**

**A saturated eval has no resolving power.** The first dataset scored 13/13 and could not rank two configurations at all. **If everything passes, the eval is too easy, not the agent too good.**

**Report the confidence interval.** 15/16 = 94% has a Wilson interval of 72–99%. With 16 cases one case is 6%: this suite distinguishes working from broken, not 85% from 92%. Treat a net change of one case as noise.

**Report what broke, not just the average.** An aggregate can rise while security or fabrication cases regress, and one such loss is not offset by two wins elsewhere. Warn when two runs differ in more than one variable.

## Lesson 08 — Judging and tracing · [notes](../lessons/08-judging-tracing/NOTES.md)

**Two ideas. A judge is a measuring instrument, so calibrate it before trusting it.** And **tracing is a view, not a collection problem** — lesson 3's `Trajectory` already held every step, token count and latency, so `build_trace` is a pure transformation. That is what building observability in from the start actually buys.

**Calibrate first, not last.** Measured 9/10 agreement (90%) against lesson 7's deterministic verdicts. **A too-lenient judge is the dangerous one** — it inflates scores and hides regressions; a too-strict one is merely annoying.

**A judge cannot see the trajectory.** The single disagreement was structural, not an error: code failed `currency_unsupported` because the agent never called the tool, while the judge passed the *answer*, which was fine. A judge cannot know an agent was right by luck, took nine steps instead of two, or ignored its tools. **So judges complement deterministic scorers; they do not replace them.**

**Verbosity bias is real and three sentences fix it.** Naive rubric preferred the padded answer in both orders; the mitigated rubric preferred the terse one. Same model, same two correct answers — the only difference being an instruction to ignore length, confidence and fluency. **A judge without it rewards an agent for being more expensive.**

**Run the A/B, not just the careful version.** Testing only the mitigated rubric would have shown "no bias found" — the wrong conclusion, since it was the mitigation working invisibly.

**Absolute scoring beat pairwise.** Both answers passed under both rubrics against an explicit factual criterion. Pairwise forces a preference even when both are correct, and that is when style decides. If you must use pairwise, run both orders and discard unmirrored verdicts.

**Test the judge with a confidently wrong answer.** The cheapest check there is. A judge rewarding fluency approves every plausible mistake.

**Judge design rules:** structured validated output (not regex over prose), reasoning field before verdict field, a failed judge must never pass, never show it the expected answer, rubric in the cache key, temperature 0.

**Cost per success is the number nobody reports.** The strict prompt was 10.2% cheaper per case and only 3.8% cheaper per success, because it failed more — so quoting "tokens down 10%" overstates the benefit by ~2.7x for what was, on accuracy, a regression. **Cost per success discounts a misleading saving rather than reversing it.** (Corrected in lesson 9: this originally claimed cost per success "went the wrong way", which the run files refute. The commentary was hardcoded prose that ignored its own numbers; it is now computed and tested.) 92% of tokens are input, so context management is a cost lever.

**Another silent measurement bug.** The trace rollup double-counted, making every cost figure exactly 2x — nothing crashed, no verdict changed, and a doubled report looks plausible. Third such bug in the project after lesson 5's label matcher and lesson 7's score cache. **The arithmetic in a measurement tool deserves a test even when it is obviously right.**

**What a trace makes obvious:** tool time 5ms against 1.06s of model time (optimising tools is pointless), input outnumbering output 12:1, and half of output tokens being invisible reasoning.

## Lesson 09 — Iteration · [notes](../lessons/09-iteration/NOTES.md)

**One idea. Measure the ruler before you measure the thing, and write the prediction down first.** The code is thin; the discipline is the content.

**Predictions were right 1 time in 4 (25%).** Graded strictly: calling the win but missing a regression is a miss. The single hit was confirming a diagnosis already established by measurement — every genuine guess about what a prompt would do was wrong, including *where* the collateral damage would land. **That number is the argument for the whole harness.** If prompt intuition were reliable, you could reason your way to a better agent and skip the measuring.

**Read the failure before theorising. It is free and it is the step that gets skipped.** The obvious diagnosis for `currency_unsupported` — "the tool description lists the supported currencies, so the eval is unfair" — was refuted for zero tokens by reading one answer already sitting in `baseline.json`. The agent refused because it believed Bitcoin's price was unknowable, not because of anything it read. The case was right and the agent was wrong.

**Removing the tool's currency list changed nothing.** 0/3 with the supported set hidden entirely, so the tool description was never the cause. And the same "call the tool rather than assuming" sentence worked in the *system prompt* and did nothing in a *tool description*. **Instruction placement matters more than instruction wording.**

**One case many times beats one suite once.** A full run is 33k tokens and tells you the average moved; four repeats of one case is ~5k and tells you whether the case is a signal at all. At n=16 a result usually turns on one case. Three of this lesson's most decisive findings were single-case probes, together costing less than one suite run.

**Two repeats cannot establish a noise floor.** Measured 0/16 flips, which set the detectable threshold to 1 case — then found a case flaking at ~20% hours later. A case failing one run in seven looks perfectly stable across two runs. **A clean result at R=2 is weak evidence of stability, not evidence of determinism.**

**The decision rule reverted a real fix, so the rule changed.** `verify_first` fixed its target case and broke one nobody predicted; the break turned out to be the flake. A net-zero result hanging on an *unpredicted* break now returns `inconclusive` with a re-test instruction. **At one case, a flake and a regression are indistinguishable**, and re-testing costs ~5k tokens against losing the fix.

**Prompt changes reintroduce old bugs at a distance.** "Attempt the most relevant tool" re-triggered lesson 2's phantom tool call — the model invents a tool when the six real ones do not fit — aborting the loop with an empty answer, three lessons later.

**A real improvement can be invisible to the suite.** A prompt change moved a refusal from "I can't look up the price" to "the tool only supports USD, EUR, GBP, INR, JPY, AUD, CAD", and no scorer could see it. **"The metric did not move" and "nothing improved" are different statements.** Combined with a matching scorer change it gave 16/16 for zero tokens — some fixes are only visible in combination, which is the blind spot in strict one-variable discipline.

**Caching and variance are in tension.** Caching makes comparisons reproducible and therefore hides variance. Measuring noise means turning it off and paying full price. One mechanism cannot do both.

**Commentary that ignores its own data is a confident caption on the wrong photograph.** Lesson 8's `--cost-compare` panel was hardcoded prose about one comparison, so it announced "cost per success went the wrong way" for every pair of runs — including one showing an 11% improvement, and including the very comparison it was written for, where cost per success had actually improved 3.8%. Found by pointing the tool at a run that did not exist when it was written. Now derived in `cost_verdict()` and tested. **Narrative text in a measurement tool needs the same tests as the arithmetic.**

**Two more silent measurement bugs, both in lesson 7's harness.** The cache key was blind to tool descriptions, so a tool experiment would have replayed the baseline and reported "no change" with total confidence — the same class as lesson 7's cache-the-score bug. And the dataset fingerprint hashed the *number* of scorers, so swapping one for another left it unchanged and `compare()` claimed apples to apples across runs graded differently. **A cache key that omits a variable turns a measurement tool into a confident liar.**

**Four attempts, nothing kept: 2 revert, 2 inconclusive.** For a lesson about improving an agent that is the honest outcome. `inconclusive` is the most useful verdict and the one teams refuse to say — it means the suite cannot resolve the change, and the follow-up is a bigger dataset, not a bigger opinion.

## Lesson 10 — Multi-agent · [notes](../lessons/10-multi-agent/NOTES.md)

**One idea. A sub-agent is a tool whose implementation is another agent loop.** `run_agent` unchanged, wire protocol unchanged, lesson 2's dispatcher still the security boundary. It needed exactly one previously-unused thing — `ToolRegistry.subset` — and nothing else. Every multi-agent framework is this plus naming, which means the content is not *how* to delegate but the four things that break once you do, all silently.

**Given its own tools, the coordinator does not delegate.** Full team available on a two-part question: zero delegations, zero sub-agent tokens. It used `search_files` and `calculate` itself, which was the correct decision. **A capable agent routes around your team.**

**The tax for a team you never call is ~30%.** Two cases where nothing was delegated still cost 30% and 33% more than solo — three extra tool schemas and a longer coordinator prompt, re-sent every step. Forcing delegation with a pure router cost **+56%** on the same question.

**76% of a delegating run's tokens are invisible to `Trajectory`.** Sub-agent model calls happen inside `registry.dispatch()`, which lesson 3's `Trajectory` does not look at — it was written before sub-agents existed. Parent accounting showed 2,265 of 9,438 actual tokens, so **read off `Trajectory.usage` the expensive architecture looks 62% cheaper than the solo one** when it is 56% dearer. Lesson 7's per-case counts and lesson 8's `CostReport` inherit the error. Fifth silent measurement bug in the project, and the first predicted rather than discovered.

**The same blind spot breaks the scorers, which is worse.** `arith_precision` failed under delegation on both models, reproducibly — and the answer was correct. What failed was `used_tools(["calculate"])`, because the parent's sequence reads `["ask_calculator"]`. **10 of 16 eval cases assert `used_tools` and an 11th asserts `answered_without_tools`**, so most process checks silently measure the wrong thing once work is delegated, and a full suite run would have reported a regression manufactured by the harness. Fixed by expanding delegations into the tools actually used — and nothing new had to be recorded, because the names were already in the log. **A view problem, not a collection problem**, exactly like lesson 8's traces.

**A sub-agent's failure looks exactly like its answer** — both are a string returned from a tool. Lesson 6's "COMPLETED ≠ correct", one level deeper and invisible. So every non-completion is raised as an error with partial output marked UNVERIFIED, and a failed pipeline stage stops the pipeline rather than laundering a failure into a confident answer three stages later.

**If you know the sequence in advance, a pipeline beats a delegating agent** — otherwise you pay a model to make a decision you already made. Delegation earns its cost only when the route depends on what is found. The price of that predictability: **a pipeline cannot recover.** The first run died in stage one when the researcher stalled, and produced nothing.

**Sharing the whole conversation doubled the cost and bought nothing.** relay → shared was +99% tokens and +160% time for the same critic verdict. It is the only way a critic can verify a quotation rather than trust a paraphrase, so not worthless — but point at a reason before paying for it.

**Recursion needs its own guard.** Lesson 3's step cap does not catch it: a nested delegation is one tool call that takes a while. Depth is enforced by *withholding* the delegation tools — a tool the model cannot see is a tool it cannot be talked into using. Breadth needs a separate counter.

**Two descriptions, two audiences.** A sub-agent's `description` is read by the parent to decide whether to delegate; its `system_prompt` is read by the sub-agent to decide how to behave. Writing one and reusing it for both is the most common multi-agent mistake.

**Five probes at ~25k tokens beat an 82k suite run.** `--evaluate` is implemented and deliberately unrun: the probes had already shown delegation fixed nothing, cost 30–90% more, and produced one apparent regression that was a measurement artifact. Lesson 9's method applied rather than described. The prediction's *direction* was confirmed and its *specifics* were wrong — it named two cases that passed and misattributed the third. Same pattern as lesson 9: **predicting that something will break is far easier than predicting what.**

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
