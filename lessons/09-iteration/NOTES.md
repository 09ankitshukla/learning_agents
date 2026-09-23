# Lesson 09 — revision notes

## The loop, in one line each

0. **Read the failure.** Free, and the step that gets skipped.
1. **Hypothesise**, including *why*.
2. **Predict** which cases move — written down before the run, never edited after.
3. **Change one variable.** Enforced in code.
4. **Measure** against a baseline made the same way.
5. **Read which cases moved**, not the average.
6. **Decide by a rule** fixed in advance.
7. **Record the attempt**, rejections included.

## The decision rule, in order

| # | condition | verdict |
|---|---|---|
| 1 | regression in a protected (`impossible`) case | REVERT, net not consulted |
| 2 | net < 0 | REVERT |
| 3 | net 0, fixed something, break was *unpredicted* | INCONCLUSIVE → `--recheck` |
| 4 | net 0 otherwise | REVERT |
| 5 | net > 0 but below the noise floor | INCONCLUSIVE |
| 6 | net ≥ floor | KEEP, unless cost/success more than doubled |

`inconclusive` means the suite cannot resolve the change, not that the change is bad.
The follow-up is a bigger dataset, not a bigger opinion.

## Two uncertainties that get confused

| question | instrument | answer here |
|---|---|---|
| Is this reproducible? | repeats → noise floor | yes, mostly |
| Does this generalise? | sample size → Wilson | unknown at n=16 |

Zero flips answers the first and says nothing about the second.

## Measured results

`openai/gpt-oss-20b` via Groq, against lesson 7's `baseline` (15/16). ~165k tokens.

**Prediction accuracy: 1/4 (25%).** The single hit was confirming a diagnosis already
established by measurement. Every genuine guess about prompt behaviour was wrong.

**Final decisions: 2 revert, 2 inconclusive, 0 kept.** For a lesson about improving an
agent that is the honest outcome. The deliverable is the method, a confirmed 16/16
candidate, a measured flake rate, and two harness bugs fixed — not a shipped change.

| experiment | cost | outcome |
|---|---|---|
| `documented_refusal` (eval change) | 0 | no change — refusal cited the wrong grounds |
| `verify_conversion` (narrow prompt) | 32k | no metric change; grounding improved invisibly, 830 tokens cheaper |
| `combined` (both, forced) | 0 | **16/16**, reproducible 4/4, cost/success 0.89x — still `inconclusive`: +1 is below the provisional threshold |
| `verify_first` (broad prompt) | 31k | fixed target, tool-choice 91%→100%, 1,943 tokens cheaper, broke one case via `phantom_tool` |
| `hide_currency_list` | ~5k probe | 0/3 — diagnosis refuted |
| `loud_currency_list` | ~5k probe | 0/3 — tool-description wording does nothing here |
| noise floor, R=2 | 33k | 0/16 flips → later shown over-confident |

**Flake rates measured with `--recheck`:**

| case | config | result |
|---|---|---|
| `files_quote_definition` | `verify_first` | 8/10 (~20% fail, `phantom_tool`) |
| `files_quote_definition` | baseline | 4/4 clean |
| `currency_unsupported` | `combined` | 4/4 |

## Findings worth keeping

**Read the recorded answer before theorising.** The first hypothesis — "the tool
description lists the supported currencies, so the eval is unfair" — was refuted for
zero tokens by reading one string in `baseline.json`. The agent's refusal was about
Bitcoin's price being unknowable, not about the tool's documented limits, which is
precisely the flawed reasoning the case was written to catch. The case was right.

**Removing the tool's currency list changed nothing.** Even with the supported set
hidden entirely, the agent still refused without calling. The tool description was
never the cause. Refuted by a 3-repeat single-case probe for ~5k tokens instead of 33k.

**System prompt carries; tool description does not.** The same sentence ("call the tool
rather than assuming") worked in the system prompt and did nothing in the tool
description. For this model, on this case.

**A real improvement can be invisible to the suite.** `verify_conversion` moved the
refusal from "I can't look up the Bitcoin price" to "the tool only supports USD, EUR,
GBP, INR, JPY, AUD, CAD" — grounded in something read rather than assumed — and no
scorer could measure it. "The metric did not move" and "nothing improved" are different
statements.

**Some fixes are only visible in combination.** The prompt change and the scorer change
were each individually null and jointly gave 16/16. That is the blind spot in strict
one-variable discipline. The rule still earns its keep; it just answers a narrower
question than it appears to.

**Prompt changes reintroduce old bugs at a distance.** `verify_first` re-triggered
lesson 2's phantom tool call: told to attempt the most relevant tool, the model invents
one when the six real tools do not fit. Aborts the loop with an empty answer.
`stop_reason=phantom_tool`.

**Predicting *that* something will break is easier than predicting *what*.** I called
the regression and named the wrong case. The over-eagerness I feared (using a tool
where none was needed) did not happen; the damage landed on a search-and-quote case.

**Two repeats cannot establish a noise floor.** 0/16 flips set the threshold to 1 case,
and hours later a case was measured flaking at ~20%. A case failing one run in seven
looks perfectly stable across two runs. `NoiseFloor.provisional` now holds the threshold
at 2 below three repeats.

**The rule reverted a real fix, so the rule changed.** A net-zero result hanging on an
unpredicted break now returns `inconclusive` with a `--recheck` instruction. Re-testing
one case costs ~5k tokens; throwing away a working change costs the change.

**Caching and variance measurement are in tension.** Caching makes comparisons
reproducible and therefore hides variance. `--noise` must run with the cache off and
pay full price. One mechanism cannot do both.

**One case many times beats one suite once.** 33k tokens tells you the average moved;
~5k tells you whether the case is a signal at all. At n=16 a result usually turns on one
case, so the cheaper measurement is often the more informative one. Three of this
lesson's most decisive findings were single-case probes.

## Bugs found in earlier lessons

**Cache key blind to tool descriptions.** `cache_key` covered question/model/steps/
prompt — complete for lesson 7, incomplete the moment tool descriptions became a
variable. A tool experiment would have replayed the baseline's cached runs and reported
"no change". Fixed with a caller-supplied `variant`, empty by default so old entries
stay valid. Same class as lesson 7's cache-the-score bug: **a key that omits a variable
turns a measurement tool into a confident liar.**

**Dataset fingerprint blind to scorer swaps.** v1 hashed `len(case.scorers)`, so
swapping one scorer for another left the fingerprint identical and `compare()` claimed
apples to apples. v2 hashes each scorer's label including its arguments, and carries an
algorithm tag so a mismatch reads "computed differently" rather than "the dataset
changed". Both committed runs were re-saved under v2 for free, from cache.

**`prediction_hit` counted no-prediction as a hit.** Two empty lists compared equal to
two empty outcomes, so an attempt that bet on nothing scored a hit and inflated the one
number this lesson exists to produce. Found by a test.

**Lesson 8's cost commentary was hardcoded prose, and wrong.** `--cost-compare` printed
"cost per success went the wrong way" for every comparison, including one showing an 11%
improvement — and including baseline→strict, where the real figures are −10.2% per case
and **−3.8% per success**. Cost per success improved; it just improved far less. Now
`tracing.cost_verdict()`, computed and tested, and the claim is corrected in lesson 8's
docs. **Cost per success discounts a misleading saving rather than reversing it** — two
thirds of that 10% was illusory. Fourth silent measurement bug in the project, after
lesson 5's label matcher, lesson 7's score cache and lesson 8's trace double-count.
**The prose a measurement tool prints needs tests as much as its arithmetic does.**

## Commands

```powershell
uv run lessons/09-iteration/iterate.py --list        # free
uv run lessons/09-iteration/iterate.py --log         # free
uv run lessons/09-iteration/iterate.py --scorecard   # free
uv run lessons/09-iteration/iterate.py --replay      # free

uv run lessons/09-iteration/iterate.py --try documented_refusal        # free, cached
uv run lessons/09-iteration/iterate.py --noise --repeats 3             # 2 runs
uv run lessons/09-iteration/iterate.py --try verify_conversion         # 1 run
uv run lessons/09-iteration/iterate.py --recheck CASE --experiment EXP --repeats 4
uv run lessons/09-iteration/iterate.py --recheck CASE --experiment baseline  # the control
```

`--recheck ... --experiment baseline` is the one people skip. A flake rate under your
change says nothing about whether your change caused it; only the control does.

## Open

- `verify_first` is unresolved: a real fix carrying a ~20% flake. Softening the prompt
  or handling `PhantomToolCall` as a recoverable observation are both untested.
- `terse_prompt` and `more_steps` are defined, predicted and unrun — the day's tokens
  went on rechecks instead.
- Only one case has a measured flake rate. The other fifteen are assumed stable.
- Every conclusion is bounded by 16 cases. Growing the dataset is the highest-value
  change available and the least interesting to do.
