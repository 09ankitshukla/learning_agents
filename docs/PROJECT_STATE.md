# Project state — read this first when resuming

Last updated: end of lesson 03, pushed to GitHub, paused before starting lesson 04.

This file exists so you (or an AI assistant in a fresh session) can resume without re-deriving context. If you're an assistant reading this: everything here is current and verified. Don't re-explore the basics; skim this, then read the file list at the bottom.

---

## Restart in two minutes

Everything is already installed and committed. Nothing is half-finished.

```powershell
# confirm the environment still works (9 checks, ~5 seconds)
uv run lessons/00-setup/check_env.py

# see the most recent working state: an agent solving a 3-step task
uv run lessons/03-agent-loop/agent.py
```

If `uv` is not found in a fresh shell, refresh PATH:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

If `check_env.py` reports the model is unavailable, Groq retired it. The script prints the currently served models; pick one and update `LLM_MODEL` in `.env`. This has already happened once during the project.

To carry on building: **lesson 04, memory and context management.** The full plan is in "Where lesson 04 picks up" below, and its problem statement is already measured.

---

## What this project is

A learn-by-doing course on building LLM agents, written as a public repo. The user is starting from zero experience with agents. Constraints agreed up front:

- **Learn by doing.** No long theory phase. Each lesson teaches a concept, then applies it immediately.
- **Every lesson ships a runnable agent** anyone can clone and run locally.
- **Every lesson documents itself** — a `README.md` (concept first, then exercises) and a `NOTES.md` (revision sheet).
- **1–2 hour sittings.** Lessons are sized to finish in one sitting.
- **No frameworks until the end.** Hand-roll the agent loop so the mechanics are visible; compare frameworks at lesson 13 once there's a basis for judgement.

## Current status

| Lesson | State |
|---|---|
| 00 — Setup | **Complete, verified against a live model** |
| 01 — Structured output | **Complete, verified against a live model** |
| 02 — Tool calling by hand | **Complete, verified against a live model** |
| 03 — The agent loop | **Complete, verified against a live model** |
| 04 — Memory and context | **Next up. Not started.** |
| 05–13 | Planned only. See the curriculum table in the root README. |

## Git

Published at **https://github.com/09ankitshukla/learning_agents** on branch `main`, six commits, local and remote in sync.

```
docs: Add revision index, glossary and project handoff
feat(lesson-03): Add the agent loop with caps, stop reasons and sandboxing
feat(lesson-02): Add hand-wired tool calling and six failure modes
feat(lesson-01): Add structured output via validate-and-repair
feat(lesson-00): Add environment setup and model mechanics
chore: Add project scaffold and shared model layer
```

Things to know before committing again:

- Identity is set **repo-locally** (`09ankitshukla` / `09ankitshukla@gmail.com`). Global git config is deliberately untouched.
- Auth is HTTPS via Git Credential Manager. The first push opened a browser; later pushes should use cached credentials.
- `.env` is gitignored and verified absent from all history. A regex scan across every commit found no key material. Keep it that way — the Groq key lives only in `.env`.
- **PowerShell breaks multi-line commit messages.** Bash heredocs (`<<'MSG'`) are a parse error, and piping a here-string to `git commit -F -` silently prepends a UTF-8 BOM to the subject. Use repeated `-m` flags, with backtick-n for newlines inside a paragraph. Verify with `git log -n 1 --pretty=%s | Format-Hex`.
- The six commits are organised by concern, not as a replay of how the code evolved: `llmkit` gained reasoning support, `tool_choice` and `ToolRegistry` while lessons 0, 2 and 3 were being built, but it all lands whole in commit 1. Every commit is still a working state.

Suggested next commit style: one per lesson, `feat(lesson-04): ...`.

## Environment (already set up, don't redo)

- **`uv` 0.12.13** installed via winget → `C:\Users\shhankit\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe`
- **Python 3.12.14** installed via `uv python install`
- **`.venv`** created, 23 pinned dependencies installed via `uv sync`
- **`.env`** exists and works. Provider is **Groq**, model **`openai/gpt-oss-120b`**.
- No Ollama, no Docker. Local models are **deliberately deferred** by user instruction — the code path is complete and documented, just unused.

Run everything from the repo root with `uv run`:

```powershell
uv run lessons/00-setup/check_env.py
uv run lessons/00-setup/hello_model.py
uv run lessons/01-structured-output/extract.py
```

**PowerShell note:** if `uv` isn't on PATH in a fresh shell, refresh it:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

## Decisions made, and why

**Hosted API over local models.** User has no GPU (AMD iGPU, no usable ROCm on Windows), so local inference is CPU-only at ~5–8 tok/s for a 7B model. A free Groq key gives ~100–240 tok/s instead. Local support is fully built and documented as "Path B" in lesson 0 for anyone cloning the repo, and for the user to revisit later.

**Frameworks moved from lesson 5 to lesson 13.** Original plan introduced one mid-course. Changed so the eval harness built in lessons 7–9 is hand-rolled and portable, and so frameworks can be judged against the user's own implementation rather than adopted blind.

**Testing at lesson 6, not near the end.** Build five agents without tests and adding tests later means rewriting all five.

**One `llmkit` layer, one method.** Every lesson goes through `client.chat(messages, tools) -> LLMResponse`. Provider swaps are a `.env` edit. Seven of eight supported providers are OpenAI-API-compatible; only Anthropic needs an adapter.

## Hard-won findings (do not rediscover these)

**Hosted model IDs expire.** `llama-3.3-70b-versatile` was the sensible Groq default when lesson 0 was written and was already retired by first run. Always query the provider for live models — `check_env.py` does this.

**GPT-OSS is a reasoning model.** It spends most of `max_tokens` on hidden deliberation (measured: 74–93% of output tokens). Too small a budget returns an **empty string** with `finish_reason="length"` — request succeeded, tokens billed, no answer. Keep `max_tokens >= 500` even for one-word answers. `LLMResponse.starved` detects this. Also: `reasoning_effort` changes the *answer*, not just its length, so hold it fixed when comparing anything else.

**Windows consoles crash on model output.** Typographic quotes and non-breaking hyphens don't exist in cp1252 → `UnicodeEncodeError` at print time, far from the model call. Fixed once in `src/llmkit/terminal.py`; it needs *both* UTF-8 stream reconfiguration *and* console code page 65001, or you swap a crash for mojibake. **Always import `console` from `llmkit`** rather than constructing a Rich `Console`.

**Pydantic docstrings leak into prompts.** Class and enum docstrings are serialised into `model_json_schema()` as `description` and sent to the model. Keep human commentary in `#` comments. Verify with `extract.py --show-schema`.

**Health checks must be authenticated.** A bare `GET /v1/models` returns 401 on hosted providers, which looks like "server down" and produces a wrong diagnosis. Use the authenticated SDK client.

**PowerShell fakes failures.** Exit code 1 from `uv` often just means it wrote to stderr, and piping to `Select-Object -First` also produces a non-zero exit. Don't trust it; check the actual output.

## Measured results worth keeping

On `openai/gpt-oss-120b` via Groq:

- Speed: ~107–240 tok/s (varies with free-tier queue time)
- Reasoning share: 74–93% of output tokens
- Lesson 1 reliability, 5 identical runs at temperature 0: **3/5 succeeded first attempt**, all 5 final outputs identical. The repair loop silently absorbed a 40% failure rate.
- Lesson 1, `json_mode` vs prompt-only: 1 attempt / 1,716 tokens vs 2 attempts / 3,044 tokens.

## Where lesson 04 picks up

Lesson 3 ends with a measured problem statement rather than a cliffhanger. Run:

```powershell
uv run lessons/03-agent-loop/agent.py --growth
```

Prompt tokens across four steps of the research task: **824 → 1,127 → 3,466 → 6,122**, totalling 11,539 input tokens for four calls. The whole conversation is re-sent every call, so cost grows roughly with the square of the step count. A 10-step agent is nearer 50x a 1-step agent on input tokens.

Lesson 04 should build:

1. **Measuring context** — token counting per message, and where the budget actually goes (tool results usually dominate).
2. **Trimming strategies** — drop oldest turns, keep the system prompt, keep the most recent N, and the failure mode of each. Note that naive trimming can orphan a `tool` message from its `assistant` tool call, which providers reject.
3. **Summarisation** — compress older turns into a running summary when the window fills, and what gets lost.
4. **Tool-result compression** — the biggest single win, given the growth table above.
5. **Persistence** — save and resume a session from disk, so an agent survives a restart.
6. **A measured before/after** — same task, same model, with and without context management. Show the token reduction and any quality cost.

Reuse `lessons/03-agent-loop/loop.py` and `toolset.py`; the loop should stay recognisably the same, with a context-management step inserted before each model call.

## Where lesson 03 picked up (done)

Lesson 02 ends by hitting a wall on purpose, and that wall is lesson 03's entire motivation. Run this to see it:

```powershell
uv run lessons/02-tool-calling/agent.py --multi-step
```

The question needs three tools **in sequence** — the current time, then a USD→INR conversion, then a percentage of the converted amount. The model cannot request the percentage until it has seen the conversion result. `agent.py` handles exactly one round, so it stops and says so explicitly.

The fix is not a better prompt or a bigger model. It is a `while` loop.

Lesson 03 should build:

1. **The loop itself** — keep calling the model, executing requested tools, and appending results until it returns prose instead of a tool request.
2. **A max-iteration cap.** Non-negotiable. Without it a confused model loops forever, burning tokens.
3. **A stopping condition** and the distinction between "finished" and "gave up".
4. **Multi-tool turns** — a model can request several tools in one response, and results must be appended in matching order with correct `tool_call_id`s.
5. **Error recovery inside the loop** — a failed tool becomes an observation and the loop continues, which is where lesson 02's `dispatch()` design pays off.
6. **A trajectory record** — what was called, in what order, with what results. This becomes the thing lesson 6 asserts on and lesson 8 traces.
7. Ideally a genuinely multi-step task: a small filesystem or research tool where the second call depends on the first's output.

Reuse `lessons/02-tool-calling/tools.py` (or a superset) so the difference between lessons 2 and 3 is purely the control flow.

## Open questions to resolve later

- How much does model size affect tool-call reliability? Placeholder in `lessons/01-structured-output/NOTES.md` for the user to fill in by rerunning `--reliability 5` on `openai/gpt-oss-20b`.
- Does the user's team standardise on a specific agent framework? If so, lesson 13 should target it. Not yet answered.
- The Anthropic and Ollama adapters have unit-verified logic but have never been run against a live endpoint.

## Repo map

```
README.md                      curriculum table, quickstart, provider tiers
docs/PROJECT_STATE.md          this file
docs/00-index.md               condensed key learnings across all lessons (the revision sheet)
docs/glossary.md               terminology, ordered by when you meet it
pyproject.toml                 pinned deps, src layout
.env                           ACTIVE CONFIG - gitignored, holds the Groq key
.env.example                   committed template, three provider tiers

src/llmkit/
  __init__.py                  public surface; exports get_client, console, message helpers
  config.py                    9 provider presets, env resolution, ConfigError with fix instructions
  types.py                     Message/ToolSpec/ToolCall/Usage/LLMResponse
  openai_compat.py             covers 7 providers; reasoning extraction; friendly error mapping
  anthropic_client.py          adapter for Anthropic's different wire format
  factory.py                   get_client() + the LLMClient protocol
  terminal.py                  the Windows UTF-8 fix; exports the shared `console`

lessons/00-setup/
  README.md                    Path A (hosted) / Path B (local), the reasoning-model trap
  check_env.py                 9 checks incl. speed, tool calling, reasoning detection
  hello_model.py               5 experiments: statelessness, temperature, system prompt, reasoning
  NOTES.md                     revision notes

lessons/01-structured-output/
  README.md                    concept, 3 experiments, 5 suggested modifications
  structured.py                THE core pattern: describe -> extract -> validate -> repair
  extract.py                   CLI deliverable, Ticket schema, --compare / --reliability
  sample_ticket.txt            deliberately ambiguous ticket
  NOTES.md                     revision notes + measured results

lessons/02-tool-calling/
  README.md                    the five steps, security section, 5 experiments
  tools.py                     3 tools + registry + dispatch() -- the security boundary
  agent.py                     deliverable; --no-tools/--show-wire/--arithmetic/--multi-step
  failures.py                  6 failure modes; scenarios 1-4 are deterministic
  NOTES.md                     revision notes incl. the phantom-tool-call finding

lessons/03-agent-loop/
  README.md                    the loop, safety mechanisms, 5 experiments
  loop.py                      THE lesson: run_agent + StopReason/Step/Trajectory
  toolset.py                   imports lesson 2's 3 tools + 3 sandboxed fs tools
  agent.py                     deliverable; --research/--step-limits/--sandbox/--growth
  NOTES.md                     revision notes incl. measured context growth
```

Note: `dispatch()` was promoted from lesson 2 into `src/llmkit/tools.py` as
`ToolRegistry.dispatch()` so lessons 3+ don't rebuild it. Lesson 2's hand-rolled
version stays in place, unchanged, as the teaching artifact. This duplication is
deliberate — mention it rather than "fixing" it.

## Reading order for a fresh session

1. This file
2. `docs/00-index.md` — the accumulated key learnings
3. `src/llmkit/types.py` — the core data model, and it's commented as teaching material
4. `lessons/01-structured-output/structured.py` — the pattern lessons 2, 3 and 11 all reuse
