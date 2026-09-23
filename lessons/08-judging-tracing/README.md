# Lesson 8 — Judging and tracing

**Time:** 1.5–2 hours
**You will end with:** a calibrated LLM judge, a measured demonstration of verbosity bias, span traces you can read, and cost reported per *successful* answer
**Depends on:** lessons 0–7

---

## Learn first

### Two ideas

**A judge is a measuring instrument, so calibrate it before you trust it.** An uncalibrated judge produces numbers that look like measurement and are not.

**Tracing is a view, not a collection problem.** Lesson 3's `Trajectory` already recorded every step, tool call, token count and latency. `build_trace` is a pure transformation of data that has existed since lesson 3 — no new instrumentation at all.

That second point is what "build observability in from the start" actually buys. Had lesson 3's loop returned only a string, this lesson would have begun by rewriting it.

### Where deterministic scorers stop

Lesson 7's scorers are all plain code, and that ordering was deliberate: free, instant, reproducible, unbiased checks should be exhausted before a model grades a model. But some things code cannot read:

> is this explanation clear?
> did it cite the source it actually used?
> is this refusal honest, or does it merely contain the word "cannot" while still asserting a made-up number?

Lesson 7 left that gap open explicitly. `declined()` says so in its own docstring — it detects refusal *language* by keyword, and an agent could refuse in words it misses.

### Calibrate before you trust

The first thing to build, not the last. Run the judge on cases whose verdict is already known deterministically, and measure agreement.

```powershell
uv run lessons/08-judging-tracing/observe.py --calibrate --model openai/gpt-oss-20b
```

Measured: **9/10 agreement (90%)**, one case too lenient, none too strict.

**Read the two error types differently.** A judge that is *too lenient* is the dangerous one — it inflates scores and hides regressions. A judge that is *too strict* is merely annoying: you investigate a failure and find the agent was fine.

### The disagreement is the most useful part

They differed on `currency_unsupported`. Deterministic scorers failed it because the agent declined **without calling the tool**. The judge passed it because the *answer* was fine — it named no Bitcoin rate and said clearly it couldn't convert.

Both are right about different things.

**A judge cannot see the trajectory.** It sees the output. It cannot tell you that an agent reached the right answer by luck, took nine steps where two would do, or never used the tool it was given. Those are exactly the failures lesson 7's `used_tools` and lesson 3's `tool_sequence` catch.

So a judge **complements** deterministic scorers rather than replacing them. Use it for what code cannot read; keep code for everything checkable.

### Judge design, and why each rule exists

**Structured output, validated.** A judge whose prose you regex for "PASS" is a second unreliable component stacked on the first. This one reuses lesson 1's describe → extract → validate → repair loop.

**Reasoning before verdict**, in the schema field order. A decision written first gets rationalised afterwards rather than derived.

**A judge that fails to produce a verdict must not pass.** Defaulting an unparseable verdict to `True` silently inflates every score.

**Never show the judge the expected answer.** Given the key, a model agrees with the key rather than assessing the work. `build_prompt` takes no expected value at all, so the mistake is unrepresentable rather than merely discouraged — and a test asserts that.

**temperature = 0.** A judge that varies its verdict run to run cannot compare two agents; you would be measuring the judge's noise.

---

## Then apply

### The experiments

**1. Calibration** — covered above. Run it first.

**2. Bias, measured as an A/B.**

```powershell
uv run lessons/08-judging-tracing/observe.py --bias --model openai/gpt-oss-20b
```

The headline result of the lesson. Two answers to "what is 71 times 89?" — one is `6319.`, the other is ~600 characters of correct-but-padded explanation. Judged under two rubrics differing by three sentences:

| rubric | terse first | padded first | reading |
|---|---|---|---|
| naive | winner B (padded) | winner A (padded) | consistent, **prefers padded** |
| mitigated | winner A (terse) | winner B (terse) | consistent, **prefers terse** |

Same model, same answers, both correct. The only difference is an instruction to ignore length, confidence and fluency.

**A judge with no such instruction quietly rewards padding — which means it rewards an agent for being more expensive.**

The A/B was necessary. Testing only the careful rubric would have shown "no bias found", which is the wrong conclusion: it would have been the mitigation working, invisibly.

Two more findings from the same run. **Absolute scoring beat pairwise** — both answers passed under both rubrics when scored against an explicit factual criterion, because pairwise forces a preference even when both answers are correct, and that is exactly when the model falls back on style. And a **confidently wrong answer** ("I used the calculator and can confirm 71 x 89 = 6,419, verified exactly") was correctly failed, which is the cheapest judge check that exists.

**3. Traces.**

```powershell
uv run lessons/08-judging-tracing/observe.py --trace "What time is it in Tokyo, and what is 71 times 89?"
```

```
What time is it in Tokyo, and what is 71 times 89?   (completed, 1.06s model time)
├── step 1   0.55s, 872 tok
│   ├── model  requested 1 tool(s)   0.54s, in 806 / out 66, 39 reasoning
│   └── tool   get_current_time      5ms -- 2026-09-24 02:37:16 (Asia/Tokyo)
├── step 2   0.17s, 897 tok
│   ├── model  requested 1 tool(s)   0.17s, in 861 / out 36, 14 reasoning
│   └── tool   calculate             0ms -- 71 * 89 = 6319
└── step 3   0.35s, 1,000 tok
    └── model  returned text (stop)  0.35s, in 896 / out 104, 50 reasoning
```

Three things the tree makes obvious that a flat log would not:

- **Tool time is 5ms against 1.06s of model time.** Optimising a tool is almost always pointless.
- **Input tokens outnumber output 12:1.** Lesson 4's quadratic growth, now visible per step.
- **Half the output tokens are reasoning** and invisible in the answer.

**4. Cost, per successful answer.**

```powershell
uv run lessons/08-judging-tracing/observe.py --cost baseline
uv run lessons/08-judging-tracing/observe.py --cost-compare baseline strict
```

| | baseline | strict | change |
|---|---|---|---|
| total cost | $0.0044 | $0.0040 | −10% |
| cost per case | $0.0003 | $0.0002 | −10% |
| successes | 15/16 | 14/16 | **−1** |
| cost per success | $0.0003 | $0.0003 | −4% |

The strict prompt is **10% cheaper per case and worse value**, because it fails more. Reporting only "tokens down 10%" would have made lesson 7's regression look like an optimisation.

**An agent at half the price that fails twice as often costs more per answer you can use.** Also note 92% of tokens are input, which makes context management a cost lever, not just a context-window one.

The dollar figures are **illustrative**. The price table in `tracing.py` was plausible when written, prices change, and Groq's free tier bills nothing. Read them as relative comparisons.

**5. Inspect one judgement.**

```powershell
uv run lessons/08-judging-tracing/observe.py --judge-case currency_unsupported
```

Shows the question, the agent's answer, the criteria, and the judge's reasoning side by side with the deterministic verdict. This is the view to use when a judge and your code disagree.

### The bug worth knowing about

`Trace.prompt_tokens` originally walked every span and summed. But a step span mirrors its `model_call` child's token counts so the tree can show a per-step total — so **every step was counted twice and every cost figure was exactly 2x.**

Nothing crashed. No verdict changed. A doubled cost report looks entirely plausible.

Caught by a test asserting the rollup equalled the sum of the step spans. **The arithmetic in a measurement tool deserves a test even when it is obviously right** — this is the third silent, plausible measurement bug in this project, after lesson 5's label matcher and lesson 7's score cache.

### Make changes

1. **Break the rubric and re-calibrate.** Remove the anti-bias sentences from `JUDGE_SYSTEM_PROMPT`, run `--calibrate --no-cache`, and see whether agreement drops. This is how you'd evaluate a rubric change for real.
2. **Judge something subjective.** Add criteria for a case about clarity or citation quality, where no deterministic scorer could work. That's the actual reason judges exist.
3. **Find a judge failure.** Write an answer that is wrong but persuasive enough to pass. When you find one, tighten the criteria — that loop is judge development.
4. **Test self-preference properly.** It's deliberately untested here because both candidate models are GPT-OSS, so any result would be confounded. Add a genuinely different model (`qwen/qwen3.8-27b` is served) and have each judge both its own and the other's answer.
5. **Add reasoning tokens to `CostReport`.** `EvalRun` doesn't record them, so `cost_from_eval_run` reports zero. Traces have them. Wiring it through would show what share of spend is invisible deliberation.
6. **Trace a failure.** Run `--trace` on something that hits `max_steps` and read the tree. Debugging from a trace rather than by re-running is the habit this lesson exists to build.

---

## Checkpoint

You should be able to answer these without looking:

- Why calibrate a judge against deterministic scorers first?
- Why is a too-lenient judge more dangerous than a too-strict one?
- What can a judge never tell you, no matter how good the rubric?
- Why must the judge not see the expected answer?
- Why did testing only the careful rubric risk the wrong conclusion about bias?
- Why is absolute scoring more robust than pairwise comparison?
- Why is cost per success different from cost per run, and which matters?
- Why was tracing almost free to add?

Then read [NOTES.md](./NOTES.md).

## Files

| File | What it is |
|---|---|
| `judge.py` | **The lesson.** Rubric judge, structured verdicts, naive vs mitigated prompts, caching. |
| `tracing.py` | Spans, the trace tree, the price table, `CostReport`. |
| `observe.py` | The CLI: `--calibrate`, `--bias`, `--trace`, `--cost`, `--cost-compare`, `--judge-case`. |
| `conftest.py` + `test_judge_tracing.py` | 28 offline tests, using lesson 6's `ScriptedClient`. |
| `NOTES.md` | Revision notes. |

`traces/` and `.cache/` are gitignored — a trace is one illustrative run rather than a measured baseline, and judgements are model-specific and regenerable.

**Next:** lesson 09, iteration. Everything needed to improve the agent deliberately now exists: change one variable, measure with lesson 7's harness, check value with lesson 8's cost report, and diagnose failures from a trace.
