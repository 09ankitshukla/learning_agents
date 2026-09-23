# Lesson 7 — Notes

## The one idea

**Lesson 6 asks "does the machinery work?". Lesson 7 asks "is the agent any good?".** Those need different tools.

`stop_reason == COMPLETED` only means the model stopped asking for tools. Lesson 6 pins a real recording where the agent gracefully declines an impossible task and the loop reports success. Only a case with an **expected outcome** can tell that apart from a correct answer.

## What an eval case needs

```python
EvalCase(
    id="currency_unsupported",
    question="Convert 100 US dollars to Bitcoin.",
    scorers=[declined(), does_not_contain([...]), used_tools(["convert_currency"])],
    why="A good agent tries, reads the error, says it cannot. A bad one invents a rate.",
)
```

Four principles:

- **Every case states why it exists.** A dataset without rationale rots — six months on nobody knows if a case is load-bearing. The `why` field is what lets you decide whether a failure matters.
- **Cases must be independently checkable.** If you cannot say what makes an answer right, you will eyeball outputs and call it evaluation.
- **Include cases the agent should refuse.** A set of only solvable tasks rewards confident guessing. The `impossible` cases catch a fabricated share price, and they are what most eval sets lack.
- **All scorers must pass.** Partial credit on 16 cases produces a number that looks precise and means little.

## Deterministic scorers first

Every scorer here is plain Python. No model grades another model until lesson 8, and that set will be smaller than you expect. Deterministic scorers are free, instant, reproducible and unbiased.

| Scorer | What it measures |
|---|---|
| `numeric_answer(v, tol)` | the right value appears, within tolerance |
| `mentions([...])` | required facts present |
| `does_not_contain([...])` | **fabrication** — the most valuable and most often missing |
| `declined()` | the agent said it could not |
| `used_tools([...])` | process, not outcome |
| `answered_without_tools()` | restraint |

`does_not_contain` deserves emphasis. A case that only checks for the right answer cannot distinguish "declined correctly" from "invented something plausible."

## Normalise aggressively, or you measure formatting

Lesson 6 taught this painfully: a test asserted `"6319" in answer` after stripping commas, the model wrote LaTeX `6{,}319`, and it became `6{}319`. Arithmetic perfect, assertion wrong.

So `normalise()` strips thousands separators, LaTeX braces and backslashes, folds typographic quotes and dashes to ASCII, and collapses whitespace. `extract_numbers` then pulls out *every* number and asks whether the right one is present — far more robust than locating "the" answer in prose that may legitimately show working.

**A scorer that is too strict measures formatting instead of correctness, and you will not notice, because the failures look real.**

## Cache the execution, never the score

The most important design decision here, and I got it wrong first.

The original cache stored the fully scored `CaseResult`. So when I added `used_tools(["convert_currency"])` to an existing case, the cached verdict was replayed and **the new scorer never ran**. The suite reported a pass for a case that should have failed.

That is the worst failure mode a measurement tool has: confidently wrong, and silent about it.

**Cache the expensive, stable thing (the agent run). Recompute the cheap, volatile thing (the score) every time.** Scoring is microseconds; an agent run is thousands of tokens. After the fix, changing a scorer re-scores all 16 cases for free, and only a changed *question* costs anything.

`Execution` satisfies the same `Trajectory` protocol the scorers expect, so a replayed run is scored by identical code to a live one.

Two related details: the cache key includes question, model, step cap and system prompt — **not** the scorers, since they do not affect the run. And **errors are never cached**, because a rate limit is not a finding about the agent, and caching one would poison every later run.

## Caching is a correctness feature, not an optimisation

A 16-case run costs ~33,000 tokens against a 200,000/day allowance. Uncached, that is a suite you run three times and abandon.

Worse, without caching you cannot compare fairly: re-running the baseline produces *different* baseline numbers, so you attribute model variance to your change. **Cached runs are reproducible runs.**

## Measured results

`openai/gpt-oss-20b`, 16 cases:

| run | task success | tool-choice | tokens |
|---|---|---|---|
| baseline (lesson 3 prompt) | **15/16 (94%)** | 91% | 33,309 |
| strict prompt | **14/16 (88%)** | 91% | 30,302 |

**The "stricter" prompt was worse.** It explicitly said "always use a tool", "never guess a number you have not obtained from a tool" — and it broke `files_find_lesson`, which the baseline passed. Verdict: `REGRESSION (net -1 cases)`.

That is the entire argument for having a harness. The strict prompt reads better. Without measurement I would have shipped it.

## Right answer, wrong process

`currency_unsupported` fails in both runs, and the failure is subtle. Asked to convert USD to Bitcoin, the agent **declined without calling the tool**. The refusal is correct; the reasoning is not — it cannot know which currencies are supported without checking, and the same reasoning would wrongly refuse a currency the tool does handle.

It is right by luck. Only scoring tool use separately from outcome exposes that, which is why `tool_choice_accuracy` is reported as its own metric. **High success with low tool accuracy means the agent is right by accident.**

## A saturated eval has no resolving power

The first version of this dataset scored **13/13 = 100%**. Useless: it could not rank two configurations or detect a regression, because it had no headroom.

Three harder cases were added (a three-step currency chain, a document-quoting task, six-decimal precision) plus a stricter requirement on the refusal case. That restored some discrimination — and 15/16 is still close to saturated.

**If everything passes, the eval is too easy, not the agent too good.** An eval set should live at the edge of what the agent can do.

## Report the confidence interval

15/16 = 94% invites false confidence. The Wilson interval is **72% to 99%**.

With 16 cases, one case is 6%. This suite can tell a working agent from a broken one. It cannot tell 85% from 92%. So the scorecard prints the interval, and `Comparison.verdict` deliberately says "improvement (within noise)" for a net gain of one.

**Printing the interval is the cheapest available defence against over-reading your own numbers.**

## Report what broke, not just the average

An aggregate can rise while specific cases regress, and the regressions are usually what you care about — especially security and fabrication cases, where one loss is not offset by two wins elsewhere.

So `compare()` returns `fixed`, `broken`, `still_failing` and `unchanged_pass` per case. A harness that printed only `94% -> 88%` would hide *which* case broke.

It also warns when the two runs differ in more than one variable: different models, different prompts, or a changed dataset fingerprint. Comparing runs scored against different datasets is a silent route to a wrong conclusion, and it happens the moment you add a case mid-session.

## Per-category breakdown

Aggregates hide structure. An agent can look 75% accurate overall while failing every arithmetic case — a completely different problem from failing a scattered quarter of everything. The scorecard groups by category for that reason.

## Errors are results, not crashes

A suite that dies on case 3 of 16 tells you nothing about cases 4–16, and rate limits make that common. An exception becomes a `CaseResult` with `error` set, marked distinctly from a failure, and excluded from the cache so a re-run retries it.

## Carry forward

- Lesson 8 adds LLM-as-judge for what deterministic scorers cannot reach ("is this explanation clear?"), plus its biases — verbosity preference, self-preference. The judge is graded against these same cases.
- `does_not_contain` is the seed of lesson 11: fabrication is a safety property, not just a quality one.
- Token accounting per case is already here; lesson 8 turns it into cost.
- 16 cases is too few. Growing the set is the single highest-value thing to do next, and it is cheap because scoring is free.
