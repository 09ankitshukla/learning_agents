# Lesson 7 — Evaluation

**Time:** 1.5–2 hours
**You will end with:** a 16-case eval dataset, deterministic scorers, a cached harness, a scorecard with confidence intervals, and a measured result showing that a "better" prompt was worse
**Depends on:** lessons 0–6

---

## Learn first

### The question lesson 6 could not answer

Lesson 6 built 111 tests that prove the machinery works. This is a different question, and lesson 6 set it up precisely:

```python
def test_completed_does_not_mean_correct(self, registry):
    """Asked for a share price it cannot fetch, the agent declines
    and the loop reports COMPLETED."""
    assert trajectory.stop_reason is StopReason.COMPLETED
    assert trajectory.succeeded  # by the loop's definition
    # A real evaluation would mark this a failure. stop_reason cannot.
```

`COMPLETED` means "the model stopped asking for tools". Nothing more. Telling a correct answer from a graceful refusal — or from a confident fabrication — needs cases with **expected outcomes**.

### What makes a usable eval case

```python
EvalCase(
    id="currency_unsupported",
    question="Convert 100 US dollars to Bitcoin.",
    scorers=[declined(), does_not_contain([...]), used_tools(["convert_currency"])],
    why="A good agent tries, reads the error, says it cannot. A bad one invents a rate.",
)
```

Four principles worth stating:

**Every case says why it exists.** A dataset without rationale rots. Six months on, nobody knows whether a case is load-bearing or was added to pad the count — and then nobody dares delete it either.

**Cases must be independently checkable.** If you can't state precisely what makes an answer right, you'll end up eyeballing outputs and calling it evaluation.

**Include cases the agent should refuse.** An eval set of only solvable tasks rewards confident guessing. The `impossible` cases here catch an invented share price and fabricated file contents, and they're what most eval sets lack.

**All scorers must pass.** Partial credit across 16 cases produces a number that looks precise and means very little.

### Deterministic scorers before judges

Every scorer in this lesson is plain Python. No model grades another model until lesson 8, and that set will be smaller than you expect. Deterministic scorers are free, instant, reproducible, and never biased.

| Scorer | Measures |
|---|---|
| `numeric_answer(v, tol)` | the right value, within tolerance |
| `mentions([...])` | required facts present |
| `does_not_contain([...])` | **fabrication** |
| `declined()` | the agent said it couldn't |
| `used_tools([...])` | process, not outcome |
| `answered_without_tools()` | restraint |

`does_not_contain` is the one people leave out and the one that earns its place. A case that only checks for the right answer cannot distinguish "declined correctly" from "invented something plausible".

### Normalise, or you'll measure formatting

Lesson 6 taught this the hard way. A test asserted `"6319" in answer` after stripping commas; the model wrote LaTeX `6{,}319`, which became `6{}319`. The arithmetic was perfect and the assertion was wrong.

So `normalise()` strips thousands separators, LaTeX braces and backslashes, folds typographic quotes and dashes to ASCII, collapses whitespace. Then `extract_numbers` pulls out *every* number and asks whether the right one is present — much more robust than trying to locate "the" answer in prose that may legitimately show its working.

**A scorer that's too strict measures formatting instead of correctness, and you won't notice, because the failures look real.**

### Cache the execution, never the score

The most important design point here, and I got it wrong first.

My original cache stored the fully scored result. So when I added `used_tools(["convert_currency"])` to an existing case, the cached verdict was replayed and **the new scorer never ran**. The suite reported a pass for a case that should have failed — a measurement tool that was confidently wrong and silent about it.

**Cache the expensive, stable thing (the agent run). Recompute the cheap, volatile thing (the score) every time.** Scoring takes microseconds; an agent run takes thousands of tokens. After the fix, changing a scorer re-scores all 16 cases for free, and only a changed *question* costs anything.

And caching isn't just an optimisation. A 16-case run is ~33,000 tokens against a 200,000/day allowance, so uncached it's a suite you run three times and abandon. Worse, re-running an uncached baseline gives *different* baseline numbers, so you attribute model variance to your change. **Cached runs are reproducible runs.**

---

## Then apply

### See the dataset

```powershell
uv run lessons/07-evaluation/evaluate.py --list
```

16 cases across seven categories, each with its rationale.

### Run a baseline

```powershell
uv run lessons/07-evaluation/evaluate.py --run baseline --model openai/gpt-oss-20b
```

The `--model` override is worth using: eval suites burn tokens, and `gpt-oss-20b` draws on a separate quota from `gpt-oss-120b`. Roughly 33,000 tokens for a cold run, free thereafter.

Measured result:

```
task success           15/16  (94%)
95% interval           72% to 99%
tool-choice accuracy   91%
mean steps             2.2
total tokens           33,309
```

### Then compare a "better" prompt

```powershell
uv run lessons/07-evaluation/evaluate.py --run strict --prompt strict --model openai/gpt-oss-20b
uv run lessons/07-evaluation/evaluate.py --compare baseline strict
```

The strict prompt says "always use a tool", "never guess a number you have not obtained from a tool". It reads like an improvement. Measured:

```
task success        94%  ->  88%     down 6%
verdict: REGRESSION  (net -1 cases)

broken (1)
  - files_find_lesson: mentions(['03-agent-loop'])
still failing (1): currency_unsupported
```

**It was worse.** Without the harness I'd have shipped it, because the prompt is more carefully written and sounds more rigorous. That's the whole argument for this lesson in one result.

### The most interesting failure

`currency_unsupported` fails in both runs, and subtly. Asked to convert USD to Bitcoin, the agent **declined without ever calling the tool**. The refusal is correct; the reasoning isn't. It cannot know which currencies are supported without checking, and the same reasoning would wrongly refuse a currency the tool *does* handle.

It's right by luck. Only scoring tool use separately from outcome exposes that — which is why `tool_choice_accuracy` is its own metric. **High success with low tool accuracy means the agent is right by accident.**

### Read the interval, not the headline

15/16 = 94% invites false confidence. The Wilson interval is **72% to 99%**.

With 16 cases, one case is 6%. This suite can tell a working agent from a broken one; it cannot tell 85% from 92%. So `Comparison.verdict` deliberately reports "improvement (within noise)" for a net gain of one case.

### Make changes

1. **Add five cases from your own use.** The highest-value change available, and cheap — scoring is free, so only the new cases cost tokens. Start with tasks you've seen an agent get wrong.
2. **Break a scorer on purpose.** Remove `normalise()` from `numeric_answer` and re-run `--run baseline` (cached, so free). Watch correct answers start failing on formatting. This is the bug lesson 6 hit, reproduced deliberately.
3. **Try the terse prompt.** `--run terse --prompt terse`, then compare against both others. Three-way comparison is where a harness starts paying for itself.
4. **Compare models.** `--run big --model openai/gpt-oss-120b`, then `--compare baseline big`. Note the warning about differing models — deliberate, since it's easy to change two variables and attribute the result to one.
5. **Make the eval harder.** Keep adding cases until success drops near 70%. An eval set should live at the edge of what the agent can do; at 100% it has no resolving power at all.
6. **Find a fabrication.** Write an impossible case where the model invents an answer, and add the `does_not_contain` that catches it. These are the most valuable cases in any set.

---

## Checkpoint

You should be able to answer these without looking:

- Why can't `stop_reason` tell you whether an agent is good?
- Why is `does_not_contain` more valuable than it first appears?
- Why cache the execution rather than the score?
- Why are errors excluded from the cache?
- Why report tool-choice accuracy separately from task success?
- What does a 100% eval score tell you about your eval?
- Why report a confidence interval on 16 cases?
- Why report per-case diffs rather than just the average?

Then read [NOTES.md](./NOTES.md).

## Files

| File | What it is |
|---|---|
| `dataset.py` | 16 cases with expected outcomes and rationale. |
| `scorers.py` | **The lesson.** Deterministic scorers and the normalisation that makes them work. |
| `harness.py` | Running, caching, and comparing. `Execution` vs score separation lives here. |
| `evaluate.py` | The CLI: `--run`, `--show`, `--compare`, `--case`, `--list`. |
| `runs/` | Saved runs, committed as artifacts. |
| `.cache/` | Cached executions (gitignored — regenerable, and model-specific). |

**Next:** lesson 08, judging and tracing. Deterministic scorers can't assess "is this explanation clear?", so lesson 8 adds LLM-as-judge along with its biases, and turns the per-case token counts here into real cost and latency accounting.
