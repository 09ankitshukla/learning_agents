# Lesson 0 — Setup, and what your model is actually doing

**Time:** 15 minutes on the hosted path, 45 on the local path
**You will end with:** a working Python environment, a model you can call, and measured answers to "how fast is it", "can it call tools", and "is it a reasoning model"

No agent code yet. The goal is to remove every environment excuse before you start learning concepts, and to surface a few properties of your model that will otherwise confuse you later.

---

## Concept first: what are we installing?

Three separate things, often conflated.

**The model** is a file of weights — billions of numbers. It does nothing on its own.

**The inference server** loads those weights and runs the maths that turns your prompt into tokens. Ollama, llama.cpp and vLLM are inference servers you run yourself; Groq and OpenAI run theirs for you. This is where the compute goes.

**The client** is your Python code sending HTTP requests to that server.

The useful accident of history: nearly every inference server copied OpenAI's HTTP API. So the same client code talks to Qwen on your laptop and GPT-OSS on Groq's cloud. That's why this repo has one small `llmkit` layer and every lesson runs on any provider.

Two numbers govern everything:

- **Context window** — how many tokens the model can see at once, covering the system prompt, the whole conversation, tool schemas, tool results and the reply. A hard wall. This is the constraint behind lessons 4 and 5.
- **Tokens per second** — generation speed. The difference between a pleasant lesson and a frustrating one.

A token is roughly ¾ of an English word.

---

## Step 1 — Install `uv` (Python toolchain)

`uv` installs Python itself, creates the virtual environment, and resolves dependencies. One tool instead of three.

```powershell
winget install --id=astral-sh.uv -e
```

Close and reopen PowerShell so `uv` lands on your `PATH`, then confirm with `uv --version`.

<details>
<summary>No winget, or it fails</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS or Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Or install Python 3.12 from [python.org](https://www.python.org/downloads/) (tick **Add python.exe to PATH**) and substitute `python -m venv .venv` + `pip install -e .` for the `uv sync` below.
</details>

## Step 2 — Create the environment

From the repository root:

```powershell
uv python install 3.12
uv sync
```

This creates `.venv/` and installs the pinned dependencies. Prefix commands with `uv run` and you never activate anything manually.

## Step 3 — Pick a model: Path A or Path B

Both paths use open-source models. They differ only in whose hardware runs them.

### Path A — Hosted open weights (recommended start)

Open models on someone else's GPUs. No download, no local compute, free tier, and fast enough for every lesson including the evaluation-heavy ones.

1. Get a free key at [console.groq.com](https://console.groq.com).
2. Create your config:

```powershell
Copy-Item .env.example .env
```

3. Edit `.env` and set your key:

```ini
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=gsk_your_key_here
```

`.env` is gitignored. Keys never belong in code or in a committed file.

**A caution about model names:** hosted providers retire models on a regular cadence. `llama-3.3-70b-versatile` was a sensible default while this lesson was being written and was already gone by the time it was tested. If a model name errors, `check_env.py` prints what the provider currently serves — trust that over any documentation, including this file.

### Path B — Local models (fully offline, no key)

Slower without a GPU, but free, private, and it works on a plane.

```powershell
winget install Ollama.Ollama
```

Reopen PowerShell. On Windows the installer registers Ollama as a background service, so there's nothing to leave running in a terminal; confirm with `ollama list`. On macOS and Linux, run `ollama serve` and leave it open.

```powershell
ollama pull qwen2.5:7b-instruct   # ~4.7 GB - better at tool calling
ollama pull qwen2.5:3b-instruct   # ~1.9 GB - faster, weaker
```

Then in `.env`:

```ini
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
LLM_TIMEOUT=180
```

Why Qwen2.5 Instruct: openly licensed, and trained with tool-calling support, which most small models handle badly. Tool calling is the foundation of lesson 2 onward, so a model that can't do it makes the whole course harder.

**Expect it to be slow on CPU.** Without a usable GPU, a 7B model at 4-bit runs at roughly 5–8 tokens/sec, so one agent turn takes ~10 seconds and a 3-step loop ~30. Fine for lessons 1–6, painful for lessons 7–9 where an eval suite makes hundreds of calls.

<details>
<summary>Path B variant: llama.cpp directly, one layer below Ollama</summary>

Ollama wraps llama.cpp and adds model management. Running `llama-server` yourself shows what's underneath — you choose the GGUF file, the quantisation, the context length, the thread count. Grab a release from [llama.cpp](https://github.com/ggml-org/llama.cpp/releases):

```powershell
.\llama-server.exe -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
```

```ini
LLM_PROVIDER=llamacpp
LLM_MODEL=local-model
```

Worth doing once. `Q4_K_M` means weights compressed to about 4 bits each — roughly a quarter of the memory for a small accuracy cost. That trade is why a 7B model runs on a laptop at all.
</details>

## Step 4 — Verify and measure

```powershell
uv run lessons/00-setup/check_env.py
```

Eight checks, ending with three measured facts about your setup:

| Check | Why you care |
|---|---|
| **tokens/sec** | Sets your expectations, and tells you whether lessons 7–9 are viable locally |
| **tool calling** | An agent is useless without it. Small models fail here first. |
| **reasoning model** | If most of your output tokens are hidden reasoning, token budgets behave in a way that will otherwise baffle you |

If `Tool calling` reports `WARN`, the model replied with prose instead of a structured call. On Path B, try the 7B rather than the 3B. Persistent failure isn't fatal — lesson 2 covers the fallback — but it makes lessons 2 and 3 noisier.

Speed guide for Path B:

| tok/s | Meaning |
|---|---|
| 25+ | Comfortable everywhere. You have a real GPU. |
| 8–25 | Fine for lessons 1–6. Switch to Path A for 7–9. |
| < 8 | Use the 3B model while iterating, and Path A for anything eval-heavy. |

## Step 5 — Run the experiments

```powershell
uv run lessons/00-setup/hello_model.py
```

Five experiments. Read the commentary it prints; that's the lesson.

1. **One call, one reply** — and what `finish_reason` tells you
2. **The model has no memory** — the same question with and without history
3. **Temperature** — the same prompt four times, and why tests can't assert exact text
4. **The system prompt** — one question, three personas, no model change
5. **Reasoning tokens** — the budget trap, covered below because it's the one that will bite you

---

## The reasoning-model trap

Worth reading even if you're on Path B, because you'll meet one soon.

Modern open models — GPT-OSS, DeepSeek-R1, Qwen3 in thinking mode — generate hidden deliberation before their visible answer. You're billed for those tokens at the output rate, they consume your `max_tokens` budget, and they don't appear in the reply text.

Measured on `openai/gpt-oss-120b`, asking for a one-word answer:

| `max_tokens` | reasoning tokens | visible answer |
|---|---|---|
| 800 | 141 | `bug` |
| **40** | **38** | **`''` (empty)** |

The second row is the trap. The request succeeded, tokens were billed, `finish_reason` was `length`, and the answer is an empty string. It looks like the model refused or broke. It didn't — it spent the whole budget thinking and had nothing left to say out loud. `LLMResponse.starved` exists to detect exactly this.

There's a second surprise. `reasoning_effort` is a quality dial, and it **changes the answer**, not just its length:

| effort | reasoning tokens | answer |
|---|---|---|
| low | 53 | `bug` |
| medium | 211 | `bug` |
| high | 798 | *(empty — budget exhausted)* |

In an earlier run at a different budget, `low` said `bug` while `medium` said `billing` for the same ticket. So reasoning effort is a variable you must hold fixed when comparing anything else, and one worth A/B testing deliberately in lesson 9.

Practical rules:

- Keep `max_tokens` ≥ 500 with reasoning models, even for one-word answers.
- Check `finish_reason` before concluding a model "failed".
- Track `reasoning_tokens` separately in cost accounting. They're often 70–90% of output.

---

## Troubleshooting

**`HTTP 401 - API key rejected`** — check `LLM_API_KEY` in `.env` for typos or trailing whitespace, and that the key is still active in the provider console.

**`groq does not serve 'X'`** — the model was retired. `check_env.py` lists current models; pick one and update `.env`.

**`Cannot reach ollama at http://localhost:11434/v1`** — the service isn't running. On Windows, launch Ollama from the Start menu (it lives in the system tray) or run `ollama serve`. Elsewhere, `ollama serve`.

**Empty model responses** — almost always the reasoning trap above. Raise `max_tokens`.

**`UnicodeEncodeError: 'charmap' codec can't encode character`** — a Windows console encoding problem, not a model problem. Already handled by `llmkit.terminal`, which forces UTF-8; if you write your own script, import `console` from `llmkit` rather than constructing your own Rich `Console`.

**Everything is slow (Path B)** — switch to `qwen2.5:3b-instruct`, and close Chrome. CPU inference wants all your cores.

**Timeouts** — raise `LLM_TIMEOUT` in `.env`.

**`uv: command not found` after installing** — reopen your terminal. PATH changes don't reach already-open shells.

---

## Checkpoint

You're done when `check_env.py` is all green and you can state, for your setup: tokens/sec, whether tool calling works, and whether you're on a reasoning model.

Then read [NOTES.md](./NOTES.md) and go to [lesson 01](../01-structured-output).
