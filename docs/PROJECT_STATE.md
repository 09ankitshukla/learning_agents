# Project state — read this first when resuming

Last updated: end of lesson 04, before starting lesson 05.

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

**Two decisions are open and should be settled before writing lesson 04.** They were put to the user and not yet answered, so ask once and don't re-derive them:

1. **Does session persistence belong in lesson 04, or its own lesson?** Lesson 04 is already the fullest one so far. Recommendation: include a minimal JSON save/resume and do not gold-plate it.
2. **Should the lesson demonstrate a real context-window overflow?** `gpt-oss-120b` has a large window, so filling it naturally is slow and expensive. Recommendation: set an artificially small budget (~2,000 tokens), label clearly that the number is synthetic, and show the trimming and summarisation machinery working against it. The alternative is to skip the demo, which is weaker.

If the user says "go" or "use your defaults", take both recommendations above.

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
| 04 — Memory and context | **Complete, verified against a live model** |
| 05 — Retrieval | **Complete, verified against a live model** |
| 06 — Testing | **Complete. 111 offline tests, all passing** |
| 07 — Evaluation | **Complete. 15/16 baseline, measured regression demonstrated** |
| 08–13 | Planned only. See the curriculum table in the root README. |

**There is a test suite and an eval harness now. Run both before and after any change:**

```powershell
uv sync --all-extras                      # NOTE: --all-extras, or pytest disappears
uv run pytest lessons                     # 153 tests, offline, ~2 seconds
uv run pytest lessons -m live             # 8 more, costs tokens

uv run lessons/07-evaluation/evaluate.py --show baseline        # free, from a saved run
uv run lessons/07-evaluation/evaluate.py --compare baseline strict
```

Gotcha worth knowing: `uv sync --extra retrieval` **removes** pytest, because uv syncs to exactly the extras you name. Always use `--all-extras`.

Lesson 3's `run_agent` gained two optional parameters while building lesson 4:
`compactor` (a hook called before each model call) and `initial_messages` (start
from an existing conversation, which is what makes resume work). Both default to
no-ops, and lessons 0–3 were re-verified afterwards.

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
- **Python 3.12.14** installed via `uv python install`. `requires-python` is now **>=3.12**, raised from 3.11 when lesson 5 arrived (fastembed needs numpy>=2.3 needs 3.12). 3.11 was only ever claimed, never tested.
- **`.venv`** created via `uv sync`. Lesson 5 needs `uv sync --extra retrieval`, which adds fastembed + numpy + onnxruntime.
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

**The Groq free tier has a 200,000 tokens-per-day cap, and agent loops burn it fast.** Hit while building lesson 4: 198,405 of 200,000 used, several experiments left unrunnable. Every loop step re-sends the whole conversation, so a handful of multi-step runs consumes a daily allowance. Two workarounds that both worked: switch to `openai/gpt-oss-20b`, which has its own quota, or wait for the window. `llmkit` now surfaces the provider's own 429 text (which limit, how long to wait) instead of a generic message.

**This is a hard constraint on lessons 7–9.** An eval suite over a few dozen cases, each a multi-step agent run, will exceed 200k tokens per day easily. Plan for it: use `gpt-oss-20b` for eval runs, keep datasets small, cache results so a re-run does not re-spend tokens, or budget across days.

**There is also a per-minute ceiling: 8,000 TPM, and exceeding it returns HTTP 413, not 429.** Hit while building lesson 5 — retrieval results plus seven tool schemas plus history reached 10,020 tokens in one request. The distinction matters: 429 means wait, 413 means this request will never fit and waiting cannot help. `llmkit` now handles both with specific advice. The fix for 413 is lesson 4's `ContextManager`, plus bounding tool output at source.

**Groq serves no embedding models** (verified: chat, safety classifiers, speech only). Lesson 5 embeds locally with `fastembed`, chosen over `sentence-transformers` because it uses ONNX and needs no PyTorch.

## Measured results worth keeping

On `openai/gpt-oss-120b` via Groq:

- Speed: ~107–240 tok/s (varies with free-tier queue time)
- Groq's served model count drifts (14 at first use, 13 on a later check). Nothing we depend on has been retired so far, but `check_env.py` is the source of truth.
- Reasoning share: 74–93% of output tokens
- Lesson 1 reliability, 5 identical runs at temperature 0: **3/5 succeeded first attempt**, all 5 final outputs identical. The repair loop silently absorbed a 40% failure rate.
- Lesson 1, `json_mode` vs prompt-only: 1 attempt / 1,716 tokens vs 2 attempts / 3,044 tokens.

## Where lesson 08 picks up

Lesson 7's scorers are all deterministic, and that was deliberate: exhaust the free, unambiguous checks before letting a model grade a model. What they cannot reach is anything subjective — "is this explanation clear?", "did it cite the right source?", "is this refusal appropriately worded rather than merely containing the word 'cannot'?"

Lesson 7 left two explicit hooks for this:

- `declined()` scores refusals by keyword and says so in its own docstring: an agent could refuse in words it misses, or say "cannot" while still fabricating. That is a judge's job.
- `mentions()` is a fragile proxy for "conveys this fact".

Lesson 08 should build:

1. **LLM-as-judge with a rubric** — a scorer that calls a model with explicit criteria and returns pass/fail plus a reason. Reuse lesson 1's structured-output pattern so the verdict is parsed and validated, not regex-scraped.
2. **Judge calibration against the deterministic set.** Run the judge on the 16 cases where the answer is already known and measure agreement. A judge that disagrees with `numeric_answer` is wrong, and you need to know that before trusting it on subjective cases.
3. **Known judge biases, demonstrated** — verbosity preference (longer answers score higher), self-preference (a model favours its own output), position bias in pairwise comparison. Each is measurable with the existing harness.
4. **Tracing** — structured spans for every model call and tool call, written to disk. `Trajectory` already holds the data; this is about making a run inspectable after the fact.
5. **Cost and latency accounting** — per-case token counts already exist in `CaseResult`. Add price tables and report cost per case, per run, and per successful answer.
6. **A trace viewer** — even a simple terminal tree. The goal is answering "why did this run fail?" without re-running it.

Reuse: `harness.py`'s caching (a judge call should be cached too, and is subject to the same cache-the-execution-not-the-score rule), and lesson 6's `CassetteClient` so judge behaviour can be tested offline.

## Where lesson 07 picked up (done)

Lesson 6 tests whether the machinery *works*. Lesson 7 asks whether the agent is any *good* — a different question, and the gap is already documented:

`stop_reason == COMPLETED` means "the model stopped asking for tools", not "the answer is correct". `test_completed_does_not_mean_correct` in lesson 6 pins this with a real recording: asked for a share price it cannot fetch, the agent declines gracefully and the loop reports success. No mechanical test can tell that apart from a correct answer.

Pieces already in place:

- `lessons/05-retrieval/queries.py` — 10 labelled queries with an `is_correct` matcher, plus a `tests` field explaining why each case exists
- `lessons/06-testing/fakes.py` — `CassetteClient` for replaying runs without spending tokens, which matters enormously for evals
- `lessons/03-agent-loop/loop.py` — `Trajectory` with `tool_sequence` and per-step usage, so tool-choice accuracy and cost are already measurable
- Lesson 5's `--measure` is effectively a single-metric eval harness; generalise it

Lesson 07 should build:

1. **An eval dataset with task-level expectations**, not just retrieval targets: input, expected outcome, and how to judge it.
2. **Metrics that matter**: task success rate, tool-choice accuracy, steps taken, tokens spent. Lesson 5 established `top-1` versus `recall@k`; extend that thinking.
3. **Deterministic scorers first** — exact match, numeric tolerance, required-substring, expected tool sequence. Cheap and unambiguous.
4. **A scorecard** that is honest about sample size. Lesson 5's repeated finding: 10 cases cannot resolve a 1-case difference, and a subtly wrong eval is more dangerous than none.
5. **Regression detection** — run the suite against two configurations and report what improved *and what broke*, since an average can rise while specific cases fail.
6. **Caching by default.** Re-running an eval must not re-spend tokens. This is both a cost and a correctness issue: cached results make comparisons reproducible.

Token budget is the hard constraint here: 200k/day and 8k/minute. Design for `gpt-oss-20b`, small datasets, and cached runs. Read the rate-limit notes above before starting.

## Where lesson 06 picked up (done)

Lesson 6 is testing, and two pieces already exist:

- `lessons/05-retrieval/queries.py` — 10 labelled queries with an `is_correct` matcher. An eval dataset in embryo.
- `lessons/03-agent-loop/loop.py` — `Trajectory`, with `tool_sequence`, `stop_reason` and per-step records. Designed from the start to be asserted against.
- `lessons/04-memory-context/context.py` — `validate()`, which catches orphaned tool results locally instead of as an HTTP 400.

Lesson 06 should build:

1. **Separating deterministic from stochastic.** Tools, chunking, trimming, `validate()` and the safe calculator are all deterministic and testable normally. Model output is not. Establishing that boundary is the core idea.
2. **Recorded fixtures (cassettes).** Save real model responses to disk and replay them, so tests are fast, free and repeatable. Lesson 2's `failures.py` already hand-builds fake `ToolCall`s — generalise that.
3. **Trajectory assertions.** Assert on `tool_sequence` and `stop_reason`, never on prose. Lesson 3 already makes this point; lesson 6 makes it routine.
4. **Property tests for the risky parts.** The AST calculator and the path sandbox both have adversarial inputs already enumerated in lessons 2 and 3 — turn those tables into tests.
5. **A regression test for the bugs this project actually hit.** The orphaned-tool-result 400, the path-only label matcher, the starved summariser, the 91%-low token estimator. Each was a real bug; each should now be caught automatically.
6. **pytest wiring** — `pyproject.toml` already has a `[tool.pytest.ini_options]` section and a `dev` extra with pytest pinned, both unused so far.

Keep tests offline by default so they cost no tokens, with live tests behind an opt-in marker.

## Where lesson 05 picked up (done)

Lesson 3 gave the agent `search_files`, which is **literal keyword search**: you must guess the exact wording used in the file. Its own tool description admits this. That limitation is lesson 5's motivation — ask "why not use eval?" and a keyword search for "eval" works, but ask "how do we stop the model running dangerous code?" and it finds nothing.

Lesson 05 should build:

1. **Embeddings** — text to vectors, and why nearby vectors mean similar meaning. Needs a local embedding model or a hosted embedding API; check what Groq serves, and note that `sentence-transformers` runs on CPU acceptably even without a GPU since embedding is far cheaper than generation.
2. **Chunking** — splitting the lesson notes into retrievable pieces, and how chunk size and overlap change results. The repo's own `NOTES.md` files are the corpus, which keeps it self-referential and honest.
3. **A vector store** — start with numpy and cosine similarity so the mechanism is visible, then note what a real store adds.
4. **Retrieval as a tool** — a `search_notes` tool beside the existing `search_files`, so the agent chooses, and you can compare keyword against semantic on the same question.
5. **Why retrieval beats stuffing** — connect to lesson 4: putting whole documents in context is what blew the token budget, and retrieval is how you send only the relevant part.
6. **Honest failure modes** — retrieval returning confidently irrelevant chunks, the chunk-boundary problem where an answer straddles two chunks, and why hybrid keyword-plus-semantic usually beats either.

Watch the token budget: embedding the corpus is cheap, but comparison experiments that run the agent repeatedly are not.

## Where lesson 04 picked up (done)

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

lessons/04-memory-context/
  README.md                    the problem, atomic groups, 4 strategies, 5 experiments
  context.py                   THE lesson: measuring, grouping, validate(), 4 strategies
  session.py                   save/resume; small on purpose
  agent.py                     deliverable; --orphan/--calibrate/--strategies/--compare/--save/--resume
  NOTES.md                     revision notes incl. the 91% estimator error

lessons/05-retrieval/
  README.md                    embeddings, measured comparison, 6 exercises
  chunking.py                  heading-aware markdown splitting; the non-replicating finding
  store.py                     THE lesson: embeddings, cache, keyword/semantic/hybrid search
  queries.py                   10 labelled queries -- the seed of lesson 7's eval set
  agent.py                     deliverable; --build/--search/--measure/--chunking/--stuffing/--ask
  NOTES.md                     revision notes incl. two measurement bugs
  .cache/                      embedding cache (gitignored)

lessons/06-testing/
  README.md                    deterministic vs stochastic, doubles, 6 exercises
  fakes.py                     THE lesson: ScriptedClient, CassetteClient, RecordingClient
  conftest.py                  sys.path for lessons 1-5; live marker + skip logic
  test_deterministic.py        56 tests: sandbox, dispatcher, trimming, chunking
  test_loop.py                 the loop under scripted doubles
  test_regressions.py          one test per bug this project shipped
  test_cassettes.py            replays recorded runs; real model quirks
  test_live.py                 8 opt-in tests (provider contract, estimator drift)
  record.py                    records cassettes; the only token-spending script here
  cassettes/*.json             committed fixtures (4 scenarios)

lessons/07-evaluation/
  README.md                    eval design, scorers, the measured regression, 6 exercises
  dataset.py                   16 cases with expected outcomes and rationale
  scorers.py                   THE lesson: deterministic scorers + normalisation
  harness.py                   running, caching (Execution vs score), comparison
  evaluate.py                  CLI: --run/--show/--compare/--case/--list
  conftest.py + test_scorers.py  42 tests for the scorers and dataset integrity
  runs/baseline.json           committed: 15/16 on gpt-oss-20b
  runs/strict.json             committed: 14/16, the measured regression
  .cache/                      cached executions (gitignored, model-specific)
```

Note lesson 5 imports lesson 4's `ContextManager` and lesson 3's `run_agent` and
`build_registry` via explicit `sys.path` inserts. Lesson folders are not importable
packages (digit-leading names), and lesson 5's tool module is `store.py` rather
than `tools.py` to avoid colliding with lesson 2's.

Note: `dispatch()` was promoted from lesson 2 into `src/llmkit/tools.py` as
`ToolRegistry.dispatch()` so lessons 3+ don't rebuild it. Lesson 2's hand-rolled
version stays in place, unchanged, as the teaching artifact. This duplication is
deliberate — mention it rather than "fixing" it.

## Reading order for a fresh session

1. This file
2. `docs/00-index.md` — the accumulated key learnings
3. `src/llmkit/types.py` — the core data model, and it's commented as teaching material
4. `lessons/01-structured-output/structured.py` — the pattern lessons 2, 3 and 11 all reuse
