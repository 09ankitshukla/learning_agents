# Lesson 0 — Notes

Revision sheet. Concepts only; commands live in the README.

## The three layers

| Layer | What it is | Examples |
|---|---|---|
| Model | A file of weights. Inert. | `qwen2.5:7b-instruct` |
| Inference server | Loads weights, generates tokens. Consumes your CPU/GPU. | Ollama, llama.cpp, vLLM, Groq |
| Client | Your code, sending HTTP requests. | `llmkit` in this repo |

Most local model providers expose OpenAI's HTTP API. That convergence is why swapping a model on your laptop for a hosted API is a config change, not a rewrite.

## Five mechanics that explain most agent behaviour

**1. Model calls are stateless.** There is no session on the server. You resend the entire conversation on every turn. "Memory" is a list you maintain and grow. Consequences: cost grows superlinearly across a conversation, latency grows with history, and eventually you hit the context limit. Lesson 4 is entirely about managing this list.

**2. Tokens are the unit of everything.** Cost, latency, and the context window are all measured in tokens (~¾ of a word). Both directions count: input tokens are usually cheaper than output tokens but you send far more of them.

**3. The context window is a hard wall.** Exceed it and you get an error or silent truncation. Truncation is worse, because the agent appears to "forget" mid-task and looks like a logic bug.

**4. Temperature trades reproducibility for variety.** Near 0 for tool calling and extraction; higher for open-ended generation. Temperature 0 is *not* a determinism guarantee — servers batch requests and floating-point reduction order shifts. So never assert on exact model text in a test. This is the root of why agent testing needs its own approach (lesson 6).

**5. `finish_reason` tells you how generation ended.** `stop` = the model finished. `length` = it hit `max_tokens` and got cut off. A truncated tool call produces invalid JSON, which looks like a model failure but is really a budget you set too low. Check this field first when output looks mangled.

## Reasoning models

Modern open models (GPT-OSS, DeepSeek-R1, o-series, Qwen3 in thinking mode) generate hidden deliberation before their visible answer.

- Reasoning tokens are **billed as output tokens** and **consume your `max_tokens` budget**.
- They are **invisible in the reply text** — they arrive in a separate `reasoning` field.
- They routinely dominate. Measured on `gpt-oss-120b`: 141 of 151 output tokens (93%) were reasoning, for a one-word answer.

**The starvation failure.** With `max_tokens=40`, the same request spent 38 tokens reasoning and returned an empty string with `finish_reason="length"`. The call succeeded, you were billed, and there is no answer. This looks exactly like a refusal or a broken model, and it is neither. `LLMResponse.starved` detects it.

**`reasoning_effort` changes the answer, not just the length.** At one budget, `low` and `medium` effort produced *different categories* for the same support ticket. So it's a variable to hold fixed when comparing prompts or models, and one worth testing deliberately.

Rules: keep `max_tokens` ≥ 500 even for one-word answers; check `finish_reason` before blaming the model; track reasoning tokens separately in cost accounting.

## Windows console encoding

Models emit typographic quotes, em dashes, non-breaking hyphens and emoji. Windows consoles default to cp1252, which cannot encode them, so printing one raises `UnicodeEncodeError` and kills the script — at print time, nowhere near the model call. Milder cases just mangle text (`table's` → `tableÆs`).

Fix needs both halves: reconfigure Python's streams to UTF-8 *and* set the console code page to 65001, otherwise you trade a crash for mojibake. Handled once in `llmkit/terminal.py`; import `console` from `llmkit` instead of building your own.

## Quantisation

`Q4_K_M` and similar tags mean weights are compressed from 16 bits to ~4 bits. Memory drops roughly 4x, quality drops a little, speed improves. It's why a 7B model runs on a laptop. Rough ordering for a fixed memory budget: **a larger model at 4-bit usually beats a smaller model at 8-bit.**

## Model size vs. capability, for agents specifically

Instruction-following degrades gracefully as models shrink. **Tool calling does not** — it needs exact structured output, so it degrades sharply. A 3B model will chat acceptably and then fail to emit valid tool JSON a third of the time.

Practical rule: below ~7B, expect to write defensive parsing and retries. That's why `ToolCall` in this repo carries a `malformed_arguments` field instead of raising — bad JSON from a model is a normal runtime condition, not an exception.

## Hosted vs. local, measured

Hosted (`gpt-oss-120b` on Groq): **~240 tok/s**. A lesson-1 extraction completes in under 2 seconds.

Local on the machine this repo was built on (AMD Ryzen 5 PRO 8540U, 31 GB RAM, integrated Radeon 740M with no usable ROCm path on Windows → CPU-only):

- 7B at 4-bit: ~5–8 tok/s → ~10s per agent turn, ~30s for a 3-step loop
- 3B at 4-bit: ~15–20 tok/s

Roughly a 30x gap. Local is workable for lessons 1–6 and genuinely instructive — small models fail in ways that teach you why the defensive patterns exist. It is not workable for lessons 7–9, where a 50-case eval suite at 10s per call runs over an hour.

**Model names on hosted providers expire.** `llama-3.3-70b-versatile` was a reasonable default when this lesson was written and was already retired by the time it was first run. Treat model IDs in any documentation as stale; query the provider (`check_env.py` does) for the live list.

## The cost you don't see at first

One run of `hello_model.py` used 3,868 tokens across ~12 calls. An *agent* loops, and each iteration resends the whole history, so token use grows quadratically with conversation length. On a free tier that surfaces as rate limits; on a paid one, as a bill. Lesson 8 measures it properly.

## The one abstraction the whole repo rests on

```python
client.chat(messages, tools=None) -> LLMResponse
```

Everything later — tool use, memory, retrieval, multi-agent orchestration, evaluation — is built from this single call plus ordinary Python. When a framework looks magical, ask what it's doing around this method. Usually the answer is "a while loop and some string formatting."

## Open questions to revisit

- How much does 3B → 7B actually improve tool-call reliability? Measure it in lesson 7 instead of guessing.
- Does prompt-only JSON coaxing hold up as well as native constrained decoding? Lesson 1 tests both.
