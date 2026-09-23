# Glossary

Terms in the order you'll meet them, not alphabetical. Definitions are practical rather than academic.

## Models and inference

**Token** — the unit a model reads and writes, roughly ¾ of an English word. Cost, latency and memory limits are all measured in tokens.

**Context window** — the maximum tokens a model can consider at once, covering the system prompt, the whole conversation, tool schemas, tool results and the reply. A hard wall: exceed it and you get an error or silent truncation.

**Compaction** — shrinking a conversation to fit a budget, applied just before sending. Umbrella term for trimming, summarising and compressing tool results.

**Atomic group** — an assistant turn with `tool_calls` plus the `tool` messages answering it. Indivisible: splitting it makes the provider reject the whole request.

**Orphaned tool result** — a `tool` message whose matching `assistant` tool_call was trimmed away. The most common bug in hand-written context management, and intermittent because whether a cut splits a group is luck.

**Trimming** — dropping whole old exchanges to fit a budget. Cheap; forgets.

**Summarisation** — replacing old turns with a model-written précis. Remembers the gist, costs an extra model call, and has a break-even point: it only pays off if many calls follow.

**Chat template** — the provider-side formatting that wraps your messages before the model sees them. Often injects its own preamble, which is a fixed per-request token cost you pay on every call.

**Inference** — running a model to produce output. The expensive part.

**Inference server** — the process that loads weights and generates tokens. Ollama, llama.cpp, vLLM, or a hosted provider.

**Quantisation** — compressing weights to fewer bits (`Q4_K_M` ≈ 4 bits) to cut memory and increase speed, at some accuracy cost. Why a 7B model runs on a laptop.

**Parameters / model size** — `7B` means 7 billion weights. Bigger is generally more capable, slower, and hungrier for memory.

**Temperature** — randomness of sampling. Near 0 for structured output and tool calling; higher for creative text. Not a determinism guarantee even at 0.

**`max_tokens`** — cap on generated tokens. Too low truncates output mid-structure.

**`finish_reason`** — why generation stopped. `stop` = model finished; `length` = hit `max_tokens`; `tool_calls` = wants to call a tool.

**Instruct / chat model** — a base model fine-tuned to follow instructions and hold conversations. Always what you want for agents.

**Reasoning model** — a model that generates hidden deliberation before its visible answer (GPT-OSS, DeepSeek-R1, o-series, Qwen3 in thinking mode). Better at multi-step problems and schema compliance; costs several times more output tokens.

**Reasoning tokens** — the hidden deliberation, billed as output tokens and counted against `max_tokens`, but absent from the reply text. Frequently 70–90% of output.

**Reasoning effort** — a `low`/`medium`/`high` dial on how much deliberation to spend. It changes the answer, not just its length, so treat it as a variable to control when comparing prompts or models.

**Starvation** — a reasoning model consuming its entire token budget on reasoning and returning empty text with `finish_reason="length"`. The request succeeded and you were billed; there is simply no answer. Fix with a larger budget or lower effort, never with a better prompt.

## Prompting and messages

**System prompt** — instructions framing the whole conversation, conventionally the first message. The cheapest and most effective control surface you have.

**Message list** — the full conversation, resent on every call. Since models are stateless, this list *is* the agent's memory.

**Roles** — `system` (instructions), `user` (input), `assistant` (model output), `tool` (result of running a tool).

**Few-shot prompting** — including worked examples in the prompt to demonstrate the desired output.

**Structured output** — constraining a model to emit machine-readable data (usually JSON) matching a schema.

**JSON mode** — a server feature constraining output to syntactically valid JSON. Guarantees valid, not correct.

**Constrained decoding** — restricting token sampling so output must match a grammar or schema. Stronger than JSON mode; still can't prevent hallucinated-but-well-typed values.

**Hallucination** — confidently generating something false. Under a required-field schema, a model will often invent a value rather than admit absence.

## Tools and agents

**Tool (function calling)** — a function you expose to the model. You send a name, description and JSON Schema; the model may reply asking for it to be called with specific arguments. **The model never executes anything.** It emits a request; your code runs it and feeds the result back.

**Tool schema** — the description sent to the model. The model's only basis for deciding whether to use your tool, which makes it the highest-leverage text in an agent.

**Tool call** — the model's request: a name plus arguments plus an id.

**Tool result** — what your code sends back, paired by id.

**`tool_choice`** — who decides whether a tool is used: `auto` (the model), `required` (a tool call is mandatory), `none` (tools visible but forbidden). Anthropic calls these `auto` / `any` / `none`.

**Dispatcher** — a function *you* write that takes the tool name the model requested and routes it to the matching function in your code. A switchboard. Not a provider feature or a library component: the provider never sees it. It is your security boundary, because it decides what can run at all, so an agent's blast radius lives here.

**Registry** — the mapping from tool name to implementation that the dispatcher looks up. Because reaching a function requires a successful lookup, the registry *is* an allowlist by construction.

**Phantom tool call** — a model requesting a tool that was never offered, usually one from its training environment (a code interpreter, a browser). Common when a system prompt insists on tool use but no tools are supplied.

**Allowlist** (in tool inputs) — enumerating what's permitted and refusing everything else. Blocks attacks you didn't anticipate, unlike a blocklist of known-bad patterns.

**Agent** — a program that calls a model in a loop, executing the tools the model asks for and feeding results back, until the task is done or a limit is hit. That's the whole definition. Everything else is engineering around it.

**Agent loop** — the `while` loop at the centre of an agent.

**Trajectory** — the actual sequence of steps a run took: which tools, in what order, with what arguments. Often more informative than the final answer when debugging or evaluating.

**ReAct** — reason then act; interleaving model reasoning with tool calls. Most modern agents are a variant of this.

**Iteration cap / max steps** — the guard that stops a confused agent looping forever. Never omit it.

**Multi-agent** — several agents with distinct roles collaborating. Sometimes better than one agent, often just slower and harder to debug.

**Handoff** — one agent transferring control, plus context, to another.

## Retrieval

**Embedding** — a vector representing text's meaning, so similar text lands nearby in vector space.

**Vector store** — a database of embeddings supporting similarity search.

**Chunking** — splitting documents into retrievable pieces. Chunk size and overlap materially affect retrieval quality.

**RAG (retrieval-augmented generation)** — fetching relevant text and adding it to the prompt so the model answers from real sources rather than memory.

**Semantic vs. keyword search** — meaning-based vs. literal matching. Hybrid *often* beats either alone, but only if the scores are combined well; measured in lesson 5, a naive blend was no better than semantic alone.

**Cosine similarity** — the standard measure of how close two embeddings are. If vectors are normalised to unit length it is just their dot product. Absolute values are compressed into a narrow band and carry little meaning; ranking and the gap between results are what inform.

**recall@k** — was the correct document anywhere in the top k results. The metric that matters for an agent, since it reads all k. Distinct from top-1 accuracy, which matters when showing one answer to a user.

**Rank fusion** — combining rankings by position (`1/(k + rank)`) rather than by raw score. Avoids the scale-mismatch problem that makes naive hybrid blending unstable.

**Chunk boundary problem** — an answer straddling two chunks is retrievable by neither half. Overlap between chunks mitigates it but never fixes it.

## Testing and evaluation

**Deterministic vs. stochastic parts** — your tools and parsing are deterministic and testable normally; model output isn't. Separating them is the core testing strategy.

**Fixture / cassette** — a recorded model response replayed in tests, making them fast, free and repeatable. Preserves quirks a hand-written fake would miss.

**Test double** — any stand-in for a real dependency. Here: a *scripted* client whose responses you write, versus a *cassette* client that replays recordings. Scripted doubles test specific paths; cassettes keep the scripted ones honest.

**Seam** — the boundary where a real dependency can be swapped for a double. In this project it is the `LLMClient` protocol's `chat()` method. Designing a substitutable seam before you need one is most of what makes code testable.

**Property-shaped test** — asserts an invariant across many inputs ("whatever trimming produces is structurally valid") rather than one example ("trimming produces exactly this"). Catches the cases you did not think of.

**Live test** — a test that calls a real API. Slow, costs tokens, fails on rate limits, so it must be opt-in. Closer to monitoring than to unit testing.

**Eval dataset** — inputs paired with expected outcomes, used to score an agent. Distinct from a test suite: tests check the machinery works, evals check the agent is good.

**Scorer** — a function turning an agent's output into pass/fail. *Deterministic* scorers are plain code (numeric match, required phrase, forbidden pattern); a *judge* uses a model. Exhaust the former first.

**Fabrication guard** — a scorer asserting something is *absent*, catching an invented answer. Without one, a case cannot distinguish a correct refusal from a plausible lie.

**Saturated eval** — one where everything passes, so it can no longer rank configurations or detect regressions. Means the eval is too easy, not the agent too good.

**Confidence interval** — the honest range around a success rate. 15/16 reads as 94% but spans roughly 72–99%; on small sets, one case is several percent.

**Execution cache** — storing what the agent *did* so scoring can be recomputed for free. Caching the *score* instead means a new scorer never runs.

**Task success rate** — the fraction of eval cases the agent actually completed. The metric that matters most.

**Tool-choice accuracy** — how often the agent picked the right tool.

**LLM-as-judge** — using a model to grade another model's output against a rubric. Reaches what code cannot (clarity, faithfulness) but cannot see the trajectory, so it complements deterministic scorers rather than replacing them.

**Judge calibration** — measuring a judge's agreement with checks already known to be correct, before trusting it where nothing can check it. A too-lenient judge is dangerous (it hides regressions); a too-strict one is merely annoying.

**Verbosity bias** — a judge preferring longer answers for the same content. Measured in lesson 8: a naive rubric preferred a padded answer, and three sentences telling the judge to ignore length reversed the preference.

**Position bias** — in pairwise comparison, preferring whichever answer is presented first. Detect it by running both orders and discarding results that are not mirrored.

**Self-preference bias** — a model favouring output from its own family. Testing it honestly requires a genuinely different model.

**Span** — one timed unit of work in a trace (a model call, a tool execution). Nesting spans makes a trace a tree, which shows which model turn triggered which work.

**Cost per success** — total cost divided by *successful* answers. The figure that matters and the one nobody reports: an agent at half the price that fails twice as often costs more per usable answer.

**Regression** — a change that improves one case while breaking others. The reason you need a dataset rather than one example.

**Golden set** — a small, hand-checked set of cases you never let regress.

## Iteration

**One-variable experiment** — a run that differs from the baseline in exactly one respect. Change two and a result cannot tell you which one moved the score, so you can neither ship half of it nor explain the rest. Lesson 9 enforces this in code rather than trusting discipline.

**Recorded prediction** — which cases you expect to move, written down *before* the run and never edited. Turns "I had a feeling" into a hit rate. Graded strictly: calling the win but missing a regression is a miss, because in production the regression is the part that matters.

**Noise floor** — how much the suite moves when nothing changes, measured by running an identical configuration repeatedly with caching off. If N cases flip on their own, a net change of N is indistinguishable from doing nothing. **Two repeats cannot establish one** — a case failing one run in seven looks perfectly stable at R=2.

**Flake** — a case whose verdict varies across identical runs. A decision that turns on a flaky case was decided by chance. Distinguish it from a regression by re-running *that one case* several times, which costs a fraction of a suite run.

**Minimum detectable effect** — the smallest change your suite can distinguish from noise. Sets what counts as a result, and is usually larger than anyone assumes.

**Reproducible vs. generalisable** — two different uncertainties, routinely confused. Repeats tell you whether a change reproduces; sample size tells you whether it generalises. Zero flips across repeats says nothing about the second, and at 16 cases the confidence intervals overlap regardless.

**Protected case** — one where a regression is never traded away, whatever the net. Fabrication and sandbox-escape checks: an agent that starts inventing prices has not got slightly worse, it has acquired a different and worse failure mode.

**Inconclusive** — the verdict meaning "the suite cannot resolve this change", as distinct from "the change is bad". The most useful of the three and the one teams refuse to say. The follow-up is a bigger dataset, not a bigger opinion.

**Attempt log / changelog of experiments** — an append-only record of what was tried, including what was rejected and why. A log of only what shipped is worse than none: it lets a measured-and-lost idea be proposed again with nothing in the repo to contradict it.

**Optimising the metric** — changing the system to make the score rise without making the product better. Lesson 9's example: hiding a tool's supported-currency list so the agent is forced to call it. The measured refutation of the underlying diagnosis is what stopped it being shipped.

## Multi-agent

**Sub-agent** — an agent invoked as a tool by another agent. Not a special construct: the parent sees a name, a description and a parameter, exactly as with any tool, and cannot tell the difference.

**Coordinator** — the agent holding the delegation tools. Keeps its own tools too, unless you deliberately strip them; a coordinator with no tools of its own pays a full agent run for a two-digit multiplication.

**Delegation** — asking a sub-agent and getting control back. The model chooses whether to delegate, so the route adapts to what is found.

**Handoff** — passing control to the next agent, which does not report back. The sequence is fixed in code, making it cheaper and more predictable and unable to recover when a stage fails.

**Router** — a coordinator with no tools of its own, so every piece of work costs a delegation. Sounds clean, measures badly.

**Least privilege (for agents)** — giving each specialist only the tools its job needs. A critic with tools becomes a second researcher; a writer that can read files will read files.

**Delegation budget** — caps on nested agent calls. **Depth** stops recursion and is best enforced by withholding the delegation tools rather than refusing the call. **Breadth** caps total calls, because six specialists once each costs the same as one specialist six times.

**Hidden cost** — tokens spent inside a tool call, where the parent's trajectory cannot see them. A delegating agent's real cost is its own usage plus its sub-agents', and any tool reading only the first under-reports — in this project by 76%, enough to make the expensive architecture look cheaper.

**Effective tool sequence** — a parent's tool sequence with delegations expanded into the tools the sub-agents actually used. Needed because process assertions like `used_tools` otherwise measure the delegation rather than the work, and report a regression that is an artifact of the instrument.

**Relay vs shared state** — passing only the previous stage's output, versus continuing the whole message list. Relay is cheap and lossy: facts a later stage needs must be plumbed to it explicitly. Shared lets a stage verify rather than trust, and costs about twice as much.

## Operations

**Trace / span** — a structured record of a run and its individual steps. How you debug something non-deterministic.

**Observability** — being able to answer "why did this run do that?" after the fact.

**Guardrail** — a check on input or output that blocks unsafe or invalid behaviour.

**Prompt injection** — untrusted text (a web page, a retrieved document, a user message) containing instructions that hijack the agent. The defining security problem of agents, and not fully solvable by prompting.

**Sandboxing** — restricting what tools can reach, so a hijacked agent can't do real damage.

**Human-in-the-loop** — requiring human approval before consequential actions.

**Allowlist** — an explicit set of permitted operations. Safer than trying to enumerate everything forbidden.

**Idempotency** — an operation safe to repeat. Valuable because agents retry.

**Streaming** — sending tokens as they're generated instead of waiting for the full reply. Mostly a perceived-latency improvement.
