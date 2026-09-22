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

**Semantic vs. keyword search** — meaning-based vs. literal matching. Hybrid usually beats either alone.

## Testing and evaluation

**Deterministic vs. stochastic parts** — your tools and parsing are deterministic and testable normally; model output isn't. Separating them is the core testing strategy.

**Fixture / cassette** — a recorded model response replayed in tests, making them fast, free and repeatable.

**Eval dataset** — inputs paired with expected outcomes, used to score an agent.

**Task success rate** — the fraction of eval cases the agent actually completed. The metric that matters most.

**Tool-choice accuracy** — how often the agent picked the right tool.

**LLM-as-judge** — using a model to grade another model's output against a rubric. Scalable, and biased in known ways (toward verbosity, and toward its own outputs).

**Regression** — a change that improves one case while breaking others. The reason you need a dataset rather than one example.

**Golden set** — a small, hand-checked set of cases you never let regress.

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
