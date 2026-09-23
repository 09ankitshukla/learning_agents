# Project state — read this first when resuming

Last updated: end of lesson 10, before starting lesson 11.

This file exists so you (or an AI assistant in a fresh session) can resume without re-deriving context. If you're an assistant reading this: everything here is current and verified. Don't re-explore the basics; skim this, then read the file list at the bottom.

---

## Restart in two minutes

Everything is installed, committed and pushed. Nothing is half-finished.

```powershell
# 1. dependencies. --all-extras matters: naming fewer extras REMOVES the others
uv sync --all-extras

# 2. the fastest, cheapest confidence check -- no tokens spent
uv run pytest lessons                    # expect 281 passed, 8 skipped, ~2s

# 3. confirm the live model still works (9 checks, a few seconds)
uv run lessons/00-setup/check_env.py

# 4. see measured results without spending tokens
uv run lessons/07-evaluation/evaluate.py --compare baseline strict
uv run lessons/09-iteration/iterate.py --log
uv run lessons/09-iteration/iterate.py --scorecard
uv run lessons/09-iteration/iterate.py --replay
```

If `uv` is not found in a fresh shell, refresh PATH:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

If `check_env.py` says the model is unavailable, Groq retired it. The script prints what is currently served; pick one and update `LLM_MODEL` in `.env`. This has already happened once, and the served model count has drifted from 14 to 11 over the project.

**To carry on building: lesson 11, guardrails and failure modes.** The full plan is in "Where lesson 11 picks up" below. No decisions are outstanding — proceed unless the user wants to change direction.

**Three habits now expected of every change, because the tooling exists:**

1. Run `uv run pytest lessons` before and after. It is free and takes two seconds.
2. If the change could affect agent behaviour, re-run the eval: `evaluate.py --run something --model openai/gpt-oss-20b`, then `--compare baseline something`. Cached executions mean a re-score costs nothing.
3. If a result turns on one or two cases, **recheck those cases before believing it**: `iterate.py --recheck CASE --experiment EXP --repeats 4`, and again with `--experiment baseline` as the control. Lesson 9 measured a case flaking at ~20% after a two-repeat noise floor said nothing flipped at all. A suite-level noise floor does not tell you about the one case in front of you.

Every bug fixed from here should also gain a test in `lessons/06-testing/test_regressions.py`. That file is the project's honest changelog.

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
| 08 — Judging and tracing | **Complete. Judge calibrated at 90%, verbosity bias measured** |
| 09 — Iteration | **Complete. 4 attempts logged, predictions 1/4, nothing kept** |
| 10 — Multi-agent | **Complete. Delegation measured as 30–90% dearer and no more accurate** |
| 11–13 | Planned only. See the curriculum table in the root README. |

**There is a test suite, an eval harness and an experiment log now. Run the first before and after any change:**

```powershell
uv sync --all-extras                      # NOTE: --all-extras, or pytest disappears
uv run pytest lessons                     # 281 tests, offline, ~2 seconds
uv run pytest lessons -m live             # 8 more, costs tokens

uv run lessons/07-evaluation/evaluate.py --show baseline        # free, from a saved run
uv run lessons/07-evaluation/evaluate.py --compare baseline strict
uv run lessons/09-iteration/iterate.py --list                   # free
uv run lessons/09-iteration/iterate.py --log                    # free
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

- **`uv` 0.12.13** installed via winget â†’ `C:\Users\shhankit\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe`
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

**Windows consoles crash on model output.** Typographic quotes and non-breaking hyphens don't exist in cp1252 â†’ `UnicodeEncodeError` at print time, far from the model call. Fixed once in `src/llmkit/terminal.py`; it needs *both* UTF-8 stream reconfiguration *and* console code page 65001, or you swap a crash for mojibake. **Always import `console` from `llmkit`** rather than constructing a Rich `Console`.

**Pydantic docstrings leak into prompts.** Class and enum docstrings are serialised into `model_json_schema()` as `description` and sent to the model. Keep human commentary in `#` comments. Verify with `extract.py --show-schema`.

**Health checks must be authenticated.** A bare `GET /v1/models` returns 401 on hosted providers, which looks like "server down" and produces a wrong diagnosis. Use the authenticated SDK client.

**PowerShell fakes failures.** Exit code 1 from `uv` often just means it wrote to stderr, and piping to `Select-Object -First` also produces a non-zero exit. Don't trust it; check the actual output.

**The Groq free tier has a 200,000 tokens-per-day cap, and agent loops burn it fast.** Hit while building lesson 4: 198,405 of 200,000 used, several experiments left unrunnable. Every loop step re-sends the whole conversation, so a handful of multi-step runs consumes a daily allowance. Two workarounds that both worked: switch to `openai/gpt-oss-20b`, which has its own quota, or wait for the window. `llmkit` now surfaces the provider's own 429 text (which limit, how long to wait) instead of a generic message.

**This is a hard constraint on lessons 7–9.** An eval suite over a few dozen cases, each a multi-step agent run, will exceed 200k tokens per day easily. Plan for it: use `gpt-oss-20b` for eval runs, keep datasets small, cache results so a re-run does not re-spend tokens, or budget across days.

**There is also a per-minute ceiling: 8,000 TPM, and exceeding it returns HTTP 413, not 429.** Hit while building lesson 5 — retrieval results plus seven tool schemas plus history reached 10,020 tokens in one request. The distinction matters: 429 means wait, 413 means this request will never fit and waiting cannot help. `llmkit` now handles both with specific advice. The fix for 413 is lesson 4's `ContextManager`, plus bounding tool output at source.

**Groq serves no embedding models** (verified: chat, safety classifiers, speech only). Lesson 5 embeds locally with `fastembed`, chosen over `sentence-transformers` because it uses ONNX and needs no PyTorch.

**Temperature 0 is not determinism, and two repeats will not prove otherwise.** Lesson 9 measured 0 of 16 cases flipping across two identical uncached runs, concluded the suite could resolve a single-case change, and then found a case flaking at roughly 20% within the same session. A case failing one run in seven has a good chance of looking perfectly stable at R=2. Use at least three repeats for a suite-level floor, and **when a decision turns on one case, recheck that case directly** — four repeats of one case costs ~5,000 tokens against ~33,000 for a suite run and answers a sharper question.

**The cheapest useful measurement is one case, many times.** Three of lesson 9's most decisive findings came from single-case probes, together costing less than one suite run: two refuted diagnoses and one exposed flake. Reach for `iterate.py --recheck` before `--try`.

**A prompt change can reintroduce a bug from five lessons earlier.** Lesson 9's `verify_first` prompt ("attempt the most relevant tool before refusing") re-triggered lesson 2's phantom tool call: the model requests a tool that was never offered when the real ones do not fit, and `run_agent` aborts with an empty answer. Nothing about the loop changed; only the prompt did.

**A sub-agent's work is invisible to `Trajectory`, and that breaks cost AND correctness.** A sub-agent runs inside `registry.dispatch()`, which lesson 3's `Trajectory` does not look at — it was written before sub-agents existed. Two consequences, both silent:

- *Cost*: 76% of a delegating run's tokens go unrecorded, so lesson 7's per-case counts and lesson 8's `CostReport` under-report by most of the bill. Lesson 10's `DelegationLog` works around it from outside; anything reading `Trajectory.usage` directly is still wrong.
- *Correctness*: the parent's `tool_sequence` reads `["ask_calculator"]`, not `["calculate"]`, so `used_tools(["calculate"])` fails on a correct answer. **10 of the 16 eval cases assert `used_tools` and an 11th asserts `answered_without_tools`**, so most process checks silently measure the wrong thing once work is delegated. A full suite run would have reported a large regression manufactured by the harness rather than by the architecture.

The fix (`DelegationLog.effective_tool_sequence` plus lesson 7's new `correct_execution` hook) needed no new data collection — the tool names were in the log already. Same shape as lesson 8's tracing insight: **a view problem, not a collection problem.** If you ever make `Trajectory` aware of nested usage, both problems go away at the source and lessons 7 and 8 get fixed for free.

**Hardcoded commentary in a measurement tool is a bug.** Lesson 8's `--cost-compare` panel was fixed prose describing the baseline-vs-strict result, so it printed "cost per success went the wrong way" for every comparison — including one that showed an 11% improvement, and including baseline-vs-strict itself, where cost per success had actually improved 3.8% (not worsened). Found in lesson 9 by pointing the tool at a run that did not exist when the prose was written. Now computed by `cost_verdict()` in `tracing.py` with four tests, and the claim is corrected in lesson 8's README/NOTES and `docs/00-index.md`. **The narrative in a measurement tool needs tests as much as its arithmetic does** — this is the fourth silent measurement bug in the project.

**Instruction placement beats instruction wording, at least on this model.** The same sentence — "call this tool to check rather than assuming" — changed behaviour in the system prompt and did nothing in a tool description (0/3). Before spending an afternoon rewording tool descriptions, test whether this model reads them as instructions at all.

## Measured results worth keeping

On `openai/gpt-oss-120b` via Groq:

- Speed: ~107–240 tok/s (varies with free-tier queue time)
- Groq's served model count drifts (14 at first use, 13 on a later check). Nothing we depend on has been retired so far, but `check_env.py` is the source of truth.
- Reasoning share: 74–93% of output tokens
- Lesson 1 reliability, 5 identical runs at temperature 0: **3/5 succeeded first attempt**, all 5 final outputs identical. The repair loop silently absorbed a 40% failure rate.
- Lesson 1, `json_mode` vs prompt-only: 1 attempt / 1,716 tokens vs 2 attempts / 3,044 tokens.
- Lesson 3 context growth: prompt tokens 824 â†’ 1,127 â†’ 3,466 â†’ 6,122 over four steps (11,539 input tokens for 4 calls). Cost grows ~quadratically with step count.
- Lesson 4 compaction, budget 900: input tokens 9,726 â†’ 6,747 (âˆ’31%); per step 1,621 â†’ 1,124. Compressing tool results alone took a conversation from 2,711 â†’ 1,429 tokens with no model call.
- Lesson 4 token estimator: originally **91% too low** (7 estimated vs 79 charged) before accounting for a ~70-token chat-template preamble and JSON's higher density. Now +19% worst case, erring high.
- Lesson 5 retrieval, 193 chunks / 10 paraphrased queries: keyword 0/10 top-1 but 7/10 recall@4; semantic 7/10 and 8/10; hybrid 7/10 and 9/10. Retrieval sends ~2.2% of the corpus versus stuffing it.

On `openai/gpt-oss-20b` (used for evals, separate quota):

- Lesson 7 baseline: **15/16 (94%)**, Wilson interval 72–99%, tool-choice accuracy 91%, ~33,000 tokens cold.
- Lesson 7 strict-prompt variant: **14/16 (88%)** — a *regression*. The stricter prompt reads better and is worse.
- `currency_unsupported` fails in both: the agent declines without calling the tool. Right answer, unjustified process.

Lesson 9, on the same model and against the same baseline (~165,000 tokens across seven experiments and five rechecks):

- **Prediction accuracy: 1 in 4.** Graded strictly — calling the win but missing a regression is a miss. The one hit was confirming a diagnosis already established by measurement, so every genuine guess about prompt behaviour was wrong.
- **Noise floor, 2 uncached repeats: 0 of 16 cases flipped.** Both runs 15/16, every per-case verdict identical. **Then contradicted the same session**: `files_quote_definition` under the `verify_first` prompt measured 8/10 (~20% failure, `stop_reason=phantom_tool`) against 4/4 clean on the control. Two repeats cannot establish a noise floor.
- `verify_conversion` (narrow prompt): no metric change, 830 tokens cheaper, but the refusal moved from "I can't look up the Bitcoin price" to "the tool only supports USD, EUR, GBP, INR, JPY, AUD, CAD". A real improvement no scorer could see.
- `verify_first` (broad prompt): fixed `currency_unsupported`, tool-choice accuracy 91% â†’ **100%**, 1,943 tokens cheaper, and broke `files_quote_definition` via a phantom tool call. Unresolved on purpose.
- `combined` (prompt + rescored case, forced, **zero tokens** because both halves were cached): **16/16**, the only 100% in the project, reproducible 4/4, cost per success 0.89x. Still recorded `inconclusive` — +1 case is below the provisional threshold.
- `hide_currency_list` and `loud_currency_list`: **0/3 each** on single-case probes costing ~5,000 tokens instead of 33,000. Removing the tool's supported-currency list entirely did not make the agent call the tool, so the obvious diagnosis was simply wrong. The same instruction worked in the system prompt and did nothing in a tool description.
- **Final: 4 attempts, 2 revert, 2 inconclusive, 0 kept.**

Lesson 10, multi-agent (both models, against the same dataset and tools — the only difference is the wiring):

- **Given its own tools, the coordinator does not delegate.** Full team available on a two-part question: zero delegations, zero sub-agent tokens. It used `search_files` and `calculate` itself, which was correct. A capable agent routes around your team.
- **The tax for a team you never call is ~30%.** `no_tool_definition` 882 → 1,151 (+30%) and `files_quote_definition` 5,735 → 7,620 (+33%), both with *zero* delegations — the cost is three extra tool schemas and a longer coordinator prompt re-sent every step.
- Forcing delegation with `--router`: 6,034 → **9,438 tokens (+56%)** on the same question.
- **76% of a delegating run's tokens are invisible to `Trajectory`** (parent saw 2,265 of 9,438). Read off `Trajectory.usage`, the expensive architecture looks **62% cheaper** than solo when it is 56% dearer.
- `arith_precision` solo → team: 1,807 → 3,395 (+88%), and **reproducibly "failed"** (20b once, 120b 0/2) — but the answer was correct. See the findings below.
- Pipeline `relay` → `shared`: 7,491 → **14,896 tokens (+99%)**, +113% prompt tokens, +160% wall time, and the same critic verdict.
- Pipeline on a keyword-friendly topic: 7,671 tokens across three stages, critic approved. On a paraphrased topic: the researcher **stalled** in stage one and the pipeline correctly produced nothing.
- **Hit the 200,000/day ceiling** (`Used 199252`) on 20b mid-probe, in a lesson that ran fewer live commands than lesson 9.
- **`--evaluate` is implemented and deliberately unrun.** Five probes at ~25,000 tokens answered the question more sharply than the 82,000-token suite would, which is lesson 9's own finding applied.

## Where lesson 11 picks up

Lesson 11 is guardrails and failure modes, and it is the first lesson where the adversary is a person rather than a model's limitations. Most of the machinery exists; what is missing is a threat model and the checks that follow from it.

What is already available, and mostly already load-bearing:

- **`ToolRegistry.dispatch()`** — the security boundary since lesson 2, with its check order documented as the security model. It blocks hallucinated tool names by construction, because the registry *is* the allowlist.
- **Lesson 3's path sandbox** — `read_file` refuses to escape the project root, and lesson 7's `files_refuse_escape` case proves it while also checking the agent does not fabricate file contents when refused. That case is in the protected category for exactly this reason.
- **Lesson 2's AST calculator** — no `eval`, adversarial inputs already enumerated, property-tested in lesson 6.
- **Lesson 10's `ToolRegistry.subset`** — per-agent privilege, currently used for tidiness. Lesson 11 is where it becomes a boundary that matters: a sub-agent processing untrusted text should not hold the tools that can act on it.
- **Lesson 9's decision rule** — the `impossible` category is already protected, so a guardrail change that regresses a fabrication or sandbox case reverts regardless of the net score. The rule is there; lesson 11 supplies more cases for it to protect.

Lesson 11 should build:

1. **A threat model first, written down.** Who is the adversary, what can they control, and what would count as a breach. Skipping this produces a pile of string filters that block yesterday's attack.
2. **Prompt injection, demonstrated on this project's own agent.** The natural vector already exists: `read_file` and `search_files` put *file contents* into the conversation, and lesson 3's loop treats a tool result as trustworthy observation. Plant a file in the sandbox containing instructions and see whether the agent follows them. **This should be built as an eval case, not a demo**, so it is checked forever rather than once.
3. **The trust boundary, stated as a rule.** Tool results are untrusted data. The loop currently makes no distinction between what the user said and what a file said, which is the root cause of every injection.
4. **Output guardrails.** Lesson 7's `does_not_contain` and `does_not_match` are already fabrication guards — generalise them into a check that runs *in the loop* rather than only in the eval, and be honest about the cost: a guardrail is a per-step tax.
5. **Approval gates.** A tool that needs human confirmation before it acts. `dispatch()` is the single place to put it, which is the payoff for having one dispatcher.
6. **Failure modes with no clean fix**, named as such. The phantom tool call (lessons 2 and 9), the sub-agent whose refusal reads as an answer (lesson 10), and the fact that a model cannot reliably distinguish instructions from data no matter how the prompt is worded.

A caution: guardrails invite security theatre. The lesson should end with a reader who can say which of their checks would actually stop an attacker and which merely raise the effort — and the honest answer for prompt injection is that nothing here solves it, it only narrows the blast radius. That is worth stating plainly rather than shipping a filter and implying otherwise.

Budget note: injection cases are cheap (short prompts, few steps), so this lesson should be affordable even after lesson 10 exhausted a day's quota. Add the cases to lesson 7's dataset and the whole suite grows, which lesson 9 has been asking for since it was written.

## Where lesson 10 picked up (done)

Lesson 10 is multi-agent patterns, and for the first time in a while there is a real question to answer rather than a capability to add: **is a second agent ever worth it?** Lesson 9's machinery can answer that, and it should be pointed at the question from the start rather than bolted on afterwards.

What is already available:

- **`ToolRegistry.subset(names)`** — written in lesson 2 and never used. It exists precisely so different agents get different powers, which is the whole premise of delegation.
- **`run_agent`'s `initial_messages`** — lesson 4's resume hook. A handoff is a conversation continued by someone else, so this is most of what a handoff needs.
- **Lesson 7's harness, lesson 9's experiment runner** — a two-agent pipeline is just another config. It should be an `Experiment` with a recorded prediction, measured against the same 16-case baseline, and judged by the same keep/revert rule.
- **Lesson 8's cost per success** — the number that decides this. Two agents means at least two full context re-sends per step, so the cost question is not a footnote.
- **Lesson 3's `Trajectory` and lesson 8's spans** — nesting a sub-agent's trace inside the parent's is the natural shape, and `Span` already supports children.

Lesson 10 should build:

1. **A delegating agent** — one agent that can call another as a tool. The cleanest framing available: a sub-agent *is* a tool whose implementation happens to be another loop, so lesson 2's dispatcher already covers the boundary.
2. **Handoff versus delegation.** Handoff passes control and does not come back; delegation gets an answer and continues. They fail differently and the distinction is usually blurred.
3. **Shared state, and why it is the hard part.** Two agents with separate message lists cannot see each other's work; two agents sharing one list re-send everything twice. Both are bad in different ways, and lesson 4's context management is the only reason either is affordable.
4. **A research / write / critique pipeline** — the canonical example, and a chance to see the critique step catch something a single agent would have shipped.
5. **The honest comparison.** Run the multi-agent version through lesson 7's suite as a lesson 9 experiment, with the prediction written first. **The expected answer is that it is worse and more expensive on this dataset**, because the 16 cases are mostly single-tool questions that need no delegation. Getting a null or negative result here is the useful outcome: it is the evidence for when *not* to reach for multiple agents, which is most of the time.
6. **Failure modes specific to multi-agent** — infinite delegation (agent A asks B, B asks A), a sub-agent's refusal being misread as an answer, and error messages losing their origin as they pass up through layers.

Budget note: a delegating run costs roughly the sum of its agents, so an eval run could be two or three times the usual ~33,000 tokens. Prefer single-case probes (`iterate.py --recheck`) while developing, and run the full suite once at the end.

A caution worth stating up front: multi-agent is the most over-applied pattern in this space. The lesson should end with a reader who can say *why* a second agent is not the answer to their problem, and that is more valuable than a working pipeline.

## Where lesson 09 picked up (done)

Lesson 9 is iteration, and every tool it needs now exists. The point is to close the loop: change one variable, measure, keep or revert — and to show that doing this properly is unglamorous and works.

What is already available:

- **Lesson 7's harness** — `evaluate.py --run NAME`, `--compare A B`, cached executions so a re-score is free, and a `Comparison` that reports what *broke* rather than only the average.
- **Lesson 8's cost report** — `--cost-compare A B`, including cost per success, which already reframed one decision.
- **Lesson 8's judge** — calibrated at 90%, for criteria code cannot check.
- **Lesson 8's traces** — for diagnosing *why* a specific case failed without re-running it.
- **Two committed runs** — `baseline` (15/16) and `strict` (14/16), a real regression to work from.

Lesson 09 should build:

1. **A disciplined loop, written down.** Hypothesis â†’ change exactly one variable â†’ measure â†’ read the failures â†’ keep or revert. The discipline is the content; the code is thin.
2. **A real improvement, earned.** `currency_unsupported` is the obvious target: it fails because the agent declines without calling the tool. Try a prompt change, measure, and see whether it fixes that case *without breaking others* — which is exactly what `--compare` exists to catch.
3. **A demonstration that intuition loses.** Lesson 7 already has one: the "better" strict prompt was worse. Add one or two more variables (tool description wording, `max_steps`, `reasoning_effort`) and show the hit rate of guessing.
4. **Variance versus signal.** Run the same configuration twice and compare. If two identical runs differ by a case, then a one-case "improvement" is noise — and that number is the floor on what the suite can detect.
5. **A changelog of attempts**, including the failures. What was tried and rejected is more useful than a list of what shipped, and it stops the same idea being retried.
6. **Cost-aware decisions.** An accuracy gain that doubles tokens may not be worth taking. Lesson 8's cost per success is how you decide.

Watch the token budget: each full eval run is ~33k tokens on `gpt-oss-20b`, and lesson 9 is inherently several runs. Cached executions mean prompt variants cost once each.

## Where lesson 08 picked up (done)

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

Prompt tokens across four steps of the research task: **824 â†’ 1,127 â†’ 3,466 â†’ 6,122**, totalling 11,539 input tokens for four calls. The whole conversation is re-sent every call, so cost grows roughly with the square of the step count. A 10-step agent is nearer 50x a 1-step agent on input tokens.

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

- **Does the user's team standardise on an agent framework?** If so, lesson 13 should target it. Asked in the first session, never answered. Worth asking again before lesson 13.
- **The Anthropic and Ollama adapters have never run against a live endpoint.** Their message translation is unit-tested, but nothing has exercised them end to end. A reader who clones this and uses Anthropic is the first real test.
- **The eval set is too small at 16 cases.** Lesson 7 says so explicitly: one case is 6%, so it cannot resolve small differences. Lesson 9 hit this from every direction — every Wilson interval overlapped, and the one confirmed 16/16 result is still recorded `inconclusive` because of it. Growing the dataset is the single highest-value improvement available and it is cheap, because scoring is free and only new questions cost tokens.
- **`currency_unsupported` has a confirmed fix that has not been adopted.** `verify_conversion` + the `documented_refusal` scorer gives 16/16, reproducible 4/4, cheaper than the baseline — but it is two variables, one of which changes the eval, and +1 case is below the detection threshold. Left as a candidate rather than shipped, which is the honest state.
- **`verify_first` is an unresolved trade-off.** It fixes the target case and takes tool-choice accuracy to 100%, and it flakes ~20% on `files_quote_definition` via a phantom tool call. The better product fix is probably to handle `PhantomToolCall` in `run_agent` as a recoverable observation ("that tool does not exist, here are the ones that do") instead of aborting the run. Untested.
- **Only one case has a measured flake rate.** The other fifteen are assumed stable on the strength of two repeats, which lesson 9 demonstrated is not enough.
- **The noise floor on disk (`lessons/09-iteration/noise.json`) is provisional**, at 2 repeats. `--noise --repeats 3` costs two full runs and would replace it with something usable.
- **`terse_prompt` and `more_steps` are defined, predicted and unrun** — the day's token budget went on rechecks instead. Both are one command away.
- **Lesson 10's `--evaluate` is implemented and unrun.** ~82,000 tokens. The probes already answered the architecture question, but running it would confirm whether the naive-vs-effective scoring gap appears across the suite as it did on `arith_precision`.
- **`Trajectory` still under-reports nested cost.** `DelegationLog` patches it from outside, for lesson 10 only. Teaching `Trajectory` about nested usage would fix lessons 7 and 8 at the source, and is the highest-value refactor left in the project.
- **Process assertions assume a fixed topology.** The effective-sequence fix handles delegation. A differently-shaped architecture would break `used_tools` again, and the general problem is unsolved.
- **Lesson 10's researcher has no semantic retrieval**, so it stalls on paraphrased questions — the exact failure lesson 5 was built to fix. The two were never wired together; doing so is a small change with a measurable outcome.
- Model size vs tool-call reliability (lesson 1's `--reliability 5` on a smaller model) still has an unfilled placeholder in `lessons/01-structured-output/NOTES.md`. Optional.

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
  runs/exp_*.json, noise_1.json  committed: lesson 9's experiment runs live here too
  .cache/                      cached executions (gitignored, model-specific)

lessons/08-judging-tracing/
  README.md                    judge design, bias A/B, cost per success, 6 exercises
  judge.py                     THE lesson: rubric judge, naive vs mitigated prompts
  tracing.py                   spans, trace tree, price table, CostReport
  observe.py                   CLI: --calibrate/--bias/--trace/--cost/--cost-compare
  conftest.py + test_judge_tracing.py   33 tests using lesson 6's ScriptedClient
  traces/, .cache/             gitignored (illustrative runs, regenerable verdicts)

lessons/09-iteration/
  README.md                    the loop, the decision rule, what actually happened
  experiments.py               THE lesson: Config, Experiment, one-variable rule,
                               7 experiments with predictions recorded before running
  iteration.py                 noise floor, decide(), Changelog, prediction scoring
  iterate.py                   CLI: --list/--try/--noise/--recheck/--log/--scorecard/--replay
  conftest.py + test_iteration.py   53 tests, mostly pinning the decision rules
  attempts.json                committed: 4 attempts, 2 revert 2 inconclusive, 0 kept
  noise.json                   committed: the (provisional, 2-repeat) noise floor

lessons/10-multi-agent/
  README.md                    delegation vs handoff, the measured verdict, 6 exercises
  team.py                      THE lesson: SubAgent, as_tool(), DelegationBudget,
                               DelegationLog (incl. effective_tool_sequence)
  pipeline.py                  handoff: Stage/PipelineRun, relay vs shared, the
                               research/write/critique sequence
  multi.py                     CLI: --team/--ask/--router/--pipeline/--compare/
                               --recursion/--probe/--evaluate
  conftest.py + test_team.py   42 tests, aimed at the silent failure modes
```

Lesson 10 changed two files outside its own folder:

- `lessons/07-evaluation/harness.py` — `run_case`/`run_eval` gained an optional
  `correct_execution(trajectory, execution) -> Execution` hook, so a caller that knows
  something the harness cannot see may repair the record *before* it is scored or
  cached. It corrects what the agent did, never the verdict, which keeps lesson 7's
  cache-the-execution-not-the-score rule intact.
- `src/llmkit/tools.py` — nothing new; `ToolRegistry.subset` finally has a caller.

Lesson 9 changed three files outside its own folder, all small and all justified in
comments at the point of change:

- `lessons/07-evaluation/harness.py` — `cache_key` gained a `variant` component (the
  cache was blind to tool descriptions), `_dataset_fingerprint` became v2 and tagged
  (it was blind to scorer swaps), `run_eval` gained `variant_key` / `extra_config` and
  records which cases came from cache, and `EvalRun` gained `tokens_actually_spent`.
- `lessons/07-evaluation/scorers.py` — every scorer factory now attaches a `label` to
  the function it returns, so a scorer can be identified before it has been run. New
  helper `scorer_label`. No `ScoreResult.name` values changed.
- `src/llmkit/tools.py` — `ToolRegistry.tools` property, so a variant registry can be
  built with the same functions and different descriptions without touching `_tools`.

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
2. `docs/00-index.md` — the accumulated key learnings across all lessons
3. `src/llmkit/types.py` — the core data model, commented as teaching material
4. `lessons/03-agent-loop/loop.py` — the agent loop everything after it builds on
5. `lessons/06-testing/test_regressions.py` — the bugs this project actually shipped, which is the fastest way to learn where the sharp edges are

## Recurring lessons, in case only one thing gets read

Seven lessons in, the same handful of ideas keep reappearing. They are worth more than any individual technique:

- **Measure rather than reason.** Almost every finding in this project contradicted an expectation: the token estimator was 91% low, headings-in-chunks did not replicate, the stricter prompt was worse, `--compare` once reported a plausible table with zero compactions.
- **A small hand-picked sample is a hypothesis, not a finding.** 8 chunks said headings help; 193 said they do not.
- **Measurement tools fail silently, which makes them the most dangerous code.** A cache that stored scores meant a new scorer never ran. A label matcher scored correct retrievals as misses. Both looked authoritative.
- **Two measurements that disagree are a gift.** That is the only reason the duplicated label bug was caught.
- **Errors should be data, not exceptions.** Tool failures as observations is what makes self-correction work at all.
- **Write down why, not just what.** The `why` on eval cases and the docstrings on regression tests are what make this repo resumable.

