# Lesson 09 — Iteration: making the agent better on purpose

Lessons 7 and 8 built instruments. This lesson is about using them without fooling
yourself, which turns out to be the harder half.

You have everything you need: a 16-case dataset, deterministic scorers, cached
executions, a calibrated judge, traces, and cost per success. What you do not have is
a *method*. Without one, the loop looks like this: change the prompt, run the suite,
see 15/16 become 16/16, ship it. Every step of that is defensible and the conclusion
can still be wrong — because the gain might be noise, or bought by a regression you
did not look for, or an artifact of a case the eval scores incorrectly.

The code in this lesson is thin. The discipline is the content.

---

## The loop

```
0. READ THE FAILURE           what actually happened? (free, and skipped constantly)
1. HYPOTHESISE                what do you believe, and WHY
2. PREDICT                    which cases will move — written down, before you run
3. CHANGE ONE VARIABLE        exactly one, enforced in code
4. MEASURE                    against a baseline made the same way
5. READ THE RESULT            which cases moved, not just the average
6. DECIDE BY RULE             keep / revert / inconclusive, by a rule set in advance
7. RECORD THE ATTEMPT         including the ones you rejected
```

Step 0 is the one that pays. This project's first experiment was refuted for zero
tokens by reading one answer already sitting in `baseline.json`. Step 2 is the one
nobody does, and it is what makes `--scorecard` possible.

---

## Concepts

### One variable, enforced

`Config` lists everything lesson 9 can change: prompt, model, step cap, tool
descriptions, dataset. `Experiment.validate()` refuses to run anything that differs
from the baseline in more than one field.

This is enforced rather than suggested because it is the rule everyone breaks when
they are in a hurry, and the cost is invisible. Change the prompt and the step cap
together, watch the score rise, and you have learned nothing you can act on: you
cannot ship half of it and you cannot explain the other half. When you genuinely need
a two-variable run, `--force` lets you have it and the changelog records that the rule
was overridden. A rule you can silently opt out of is not a rule.

### A prediction is a bet, recorded before the run

`predicts_fixed` and `predicts_broken` are filled in before running and never edited.
That turns "I had a feeling that would work" into a number.

The grading is deliberately strict: predicting the win but missing a regression counts
as a **miss**, because in production the regression is the part that matters.

### The noise floor: measure the ruler before the thing

Run the identical configuration twice and some cases flip anyway. That flip count is
the floor on what your suite can detect — if two cases move on their own, a two-case
"improvement" is indistinguishable from doing nothing.

Caching and variance measurement are in direct tension, and it is worth sitting with
that. Lesson 7 caches executions so comparisons are *reproducible*: re-run the
baseline and you get identical numbers. That is exactly what you want when comparing
two configurations, and exactly what makes variance invisible. So `--noise` runs with
the cache off and pays full price. One mechanism cannot give you both.

### Two different uncertainties, routinely confused

- **Is this change reproducible?** → run-to-run variance → the noise floor.
- **Does this change generalise?** → sample size → Wilson intervals.

Measuring zero flips answers the first and says nothing about the second. A one-case
gain that reproduces perfectly is still one case out of sixteen, and the intervals for
15/16 and 16/16 overlap heavily. `decide()` answers the first question and reports the
second as a caveat, because gating on overlapping intervals at n=16 would block every
change this suite could ever measure — which is how a team learns to ignore its own
harness.

### Decide by rule, in this order

1. **A regression in a protected category reverts**, whatever the net. The `impossible`
   cases are fabrication and sandbox-escape checks. An agent that starts inventing
   share prices has not got slightly worse; it has acquired a different and worse
   failure mode, and averaging that against two arithmetic wins is how you ship
   something harmful.
2. **A net loss reverts.**
3. **A tie that hangs on an unpredicted break asks for a re-test**, not a revert. At
   one case, a flake and a regression look identical.
4. **No movement reverts.** Nothing gained; do not carry the complexity.
5. **A gain below the noise floor is inconclusive.** This is the rule that costs
   discipline, because it refuses wins you want to bank.
6. **Otherwise keep** — unless the win more than doubled the cost per successful
   answer, which is a product decision rather than an automatic yes.

`inconclusive` is the most useful of the three verdicts and the one teams refuse to
say. It means the suite cannot resolve the change, not that the change is bad. The
follow-up is a bigger dataset, not a bigger opinion.

### The changelog keeps the failures

`attempts.json` is append-only and records rejected attempts alongside kept ones. A log
of only what shipped is worse than no log: six months on, someone proposes an idea that
was already measured and lost, and nothing in the repo contradicts them.

---

## Run it

```powershell
uv run lessons/09-iteration/iterate.py --list         # experiments, predictions, cost
uv run lessons/09-iteration/iterate.py --log          # the changelog, rejections included
uv run lessons/09-iteration/iterate.py --scorecard    # were the predictions right?
uv run lessons/09-iteration/iterate.py --replay        # re-derive decisions, free
```

All four cost nothing. Then the ones that spend tokens:

```powershell
# free: changes scoring only, so cached executions cover it
uv run lessons/09-iteration/iterate.py --try documented_refusal

# ~33k tokens: one full suite run
uv run lessons/09-iteration/iterate.py --noise --repeats 3
uv run lessons/09-iteration/iterate.py --try verify_conversion

# ~5k tokens: one case, repeated. Often the better buy.
uv run lessons/09-iteration/iterate.py --recheck currency_unsupported \
    --experiment verify_first --repeats 4
```

A full run is ~33,000 tokens against 200,000/day, so this is three or four experiments
per day. That constraint is part of the lesson: it forces you to decide which
hypothesis is worth testing, which is a better habit than being able to test them all.

---

## What actually happened

Seven experiments, four recorded attempts, about 165,000 tokens. Everything below is
measured on `openai/gpt-oss-20b` via Groq, against lesson 7's committed `baseline`
(15/16).

### Predictions were right 1 time in 4

| experiment | predicted | actual | |
|---|---|---|---|
| `documented_refusal` | fixes `currency_unsupported` | no change | miss |
| `verify_conversion` | fixes `currency_unsupported` | no change | miss |
| `combined_verify_and_rescore` | fixes `currency_unsupported` | fixed it | **hit** |
| `verify_first` | fixes `currency_unsupported`, breaks `no_tool_definition` | fixed it, broke `files_quote_definition` | miss |

25%. The one hit was a confirmation of a diagnosis already established by
measurement — that is, the only prediction that landed was the one that was barely a
prediction. Every genuine guess about what a prompt would do was wrong, including the
guess about *where* the collateral damage would land.

### Reading one answer refuted the first hypothesis for free

`currency_unsupported` asks the agent to convert 100 USD to Bitcoin. It fails because
the agent refuses *without calling the tool* — right answer, unjustified process.

The obvious diagnosis: `convert_currency`'s description already lists the seven
currencies it supports, so the agent can see BTC is absent and the eval is being
unfair. On that theory the *case* should change, not the agent. So `documented_refusal`
swaps `used_tools` for a scorer that accepts a refusal citing the documented limits.

It changed nothing, and the reason was sitting in `baseline.json` all along:

> "I'm sorry, but I don't have a way to look up the current price of Bitcoin, so I
> can't give you an exact conversion for 100USD."

That is not a claim about the tool's documented currency list. It is a claim about
Bitcoin's price being unknowable, and it is exactly the flawed reasoning the case's
`why` field predicted: the same logic would refuse a currency the tool does handle.
**The case was right and the agent was wrong.** A hypothesis that would have cost a
suite run was refuted by reading one string.

### Removing the currency list changed nothing — the diagnosis was simply wrong

To test the diagnosis directly, `hide_currency_list` strips the supported-currency
list out of the tool description entirely. If the description is what lets the agent
skip the call, hiding it forces the call.

Probed on one case, three repeats, ~5,000 tokens instead of 33,000:

```
1/3  fail  stop=completed  tools=(none)
2/3  fail  stop=completed  tools=(none)
3/3  fail  stop=completed  tools=(none)
```

With no currency list at all, the agent still refuses without calling. The tool
description was never the cause. And `loud_currency_list` — which keeps the list and
adds "call this tool to check rather than assuming" — also scored 0/3, while *the same
sentence in the system prompt* did work. For this model, instruction in the system
prompt carries and instruction in a tool description does not. Worth knowing before
spending an afternoon rewording tool descriptions.

### A real improvement the suite could not see

`verify_conversion` scoped the "attempt the tool before refusing" rule to conversions.
Metric change: none. But the refusal moved from the answer above to:

> "the conversion tool only supports the currencies listed (USD, EUR, GBP, INR, JPY,
> AUD, CAD). Bitcoin isn't included, so I can't perform that conversion."

That is a genuine improvement in grounding — the refusal is now based on something the
agent read rather than something it assumed — and **no scorer in the dataset could
see it**. Both halves of the fix were individually invisible: the prompt improved the
grounding with no scorer to measure it, and the scorer was written for grounding the
agent did not yet have.

Together (forced, two variables, zero tokens because both halves were cached) they
give **16/16**, the only 100% in the project, reproducible 4/4, at *lower* cost per
success (0.89x) because the prompt is shorter and one more case succeeds.

And the rule still records it as `inconclusive`, because +1 case is below the
provisional two-case threshold. That is the rule working, not the rule malfunctioning:
a reproducible 16/16 at n=16 with overlapping confidence intervals is a promising
result the suite cannot resolve. The honest output of this lesson is a *candidate*,
not a shipped change.

This is the blind spot in strict one-variable discipline, and it is worth naming: some
improvements are only visible in combination, and a metric that cannot see an
improvement will report progress as nothing. The one-variable rule still earns its
keep — it tells you which variable moved the *metric* — but "the metric did not move"
and "nothing improved" are different statements.

### The broad prompt worked, and quietly reintroduced a lesson 2 bug

`verify_first` — the same rule, unscoped — did make the agent call the tool:

> "The conversion tool only supports the following currencies: AUD, CAD, EUR, GBP,
> INR, JPY, and USD. The error message was: 'Unsupported currency 'BTC'. Supported:
> AUD, CAD, EUR, GBP, INR, JPY, USD.'"

Tool-choice accuracy rose from 91% to 100% and the run was 1,943 tokens cheaper. But
`files_quote_definition` broke, with `stop_reason=phantom_tool` and an empty answer:
the model requested a tool that was never offered.

That is **lesson 2's finding, re-triggered by a prompt change three lessons later**.
Lesson 2 observed phantom tool calls when a prompt insisted on using tools and none
were available; a prompt that says "attempt the most relevant tool" produces the same
pressure, and when the six real tools do not obviously fit, the model invents one.

I predicted collateral damage and named the wrong case. The over-eagerness I expected —
reaching for a tool on a question that needed none — never happened;
`no_tool_definition` held. The damage landed somewhere I had not considered at all.

### The noise floor was over-confident, and it cost a real fix

Two identical uncached runs of the baseline: **0 of 16 cases flipped**, both 15/16,
every per-case verdict identical. That set `min_detectable` to 1 case, which licensed
believing any single-case movement.

Within the same session, `files_quote_definition` under `verify_first` measured:

```
--recheck files_quote_definition --experiment verify_first  -> 3/4 passed, phantom_tool
--recheck files_quote_definition --experiment baseline      -> 4/4 passed
```

Across ten observations under that prompt, two failed — roughly 20% — against 4/4
clean on the control. So the "regression" that made the rule revert `verify_first` was
largely chance, and the rule threw away a change that fixed its target case and
improved tool-choice accuracy to 100%.

Two fixes followed, both in the code rather than the prose:

- `NoiseFloor.provisional` — fewer than three repeats no longer licenses a threshold of
  1. A case failing one run in seven looks perfectly stable across two runs, so a clean
  R=2 result is weak evidence of stability, not evidence of determinism.
- Rule 3 — a tie that hangs on an *unpredicted* break returns `inconclusive` with a
  `--recheck` instruction instead of reverting. Re-testing one case costs ~5,000
  tokens; discarding a working fix costs the fix.

`--replay` immediately showed 2 of the 4 stored decisions would come out differently
under the corrected rule — `verify_first` from `revert` to `inconclusive`, and
`combined` from `keep` to `inconclusive`. That divergence is the whole reason runs are
kept as artifacts: a decision is a function of evidence and a rule, and both are
allowed to improve. The changelog was then regenerated (free, from cache) so the stored
decisions match the rule as it now stands; the superseded ones are described here.

**Final state: four attempts, nothing kept.** Two reverts, two inconclusive. For a
lesson about improving an agent, that is the honest result — and the useful output is
not a shipped change but a working method, a confirmed 16/16 candidate, a measured
flake, two harness bugs fixed, and a prediction hit rate of 25% that justifies all of
the above.

### The cheapest useful measurement is one case, many times

The headline cost lesson. A full suite run is 33,000 tokens and tells you the average
moved. Four repeats of one case is ~5,000 tokens and tells you whether the case is a
signal at all. When a result turns on one case — and at n=16 it usually does — the
second measurement is worth more than the first.

Three of this lesson's most decisive findings came from single-case probes:
`hide_currency_list` refuted, `loud_currency_list` refuted, and the `verify_first`
regression exposed as a flake. Together those cost less than one suite run.

---

## Three bugs this lesson found in earlier lessons

Both were found by needing something lesson 7 never needed, which is the usual way.

**The cache key was blind to tool descriptions.** `cache_key` hashed question, model,
step cap and system prompt — everything lesson 7 could change, so it was complete *for
lesson 7*. Lesson 9 varies tool descriptions, which changes behaviour and appears
nowhere in that key, so a tool experiment would have been served the baseline's cached
runs and reported "no change" with total confidence. Fixed with a caller-supplied
`variant` component, empty by default so every pre-existing cache entry stays valid.
Note the sharp edge that leaves: forget to pass it and you get silent reuse, not an
error. `Config.residual_key` owns it and a test pins it.

Same class of bug as lesson 7's own cache-the-score mistake: **a cache key that omits a
variable turns a measurement tool into a confident liar.**

**The dataset fingerprint was blind to scorer swaps.** `_dataset_fingerprint` hashed the
*number* of scorers per case, so replacing one scorer with a different one left the
fingerprint unchanged and `compare()` reported apples to apples across two runs graded
by different instruments. Lesson 9 swaps a scorer on purpose, which is how it surfaced.
v2 hashes each scorer's label, arguments included, and the fingerprint now carries an
algorithm tag so a mismatch can be reported as "computed differently" rather than "the
dataset changed" — different claims deserving different words.

Both committed runs were re-saved under v2 for free, because cached executions make
re-scoring a run cost nothing.

**Lesson 8's cost commentary asserted a conclusion instead of reading its data.** The
`--cost-compare` panel was hardcoded prose about the baseline-vs-strict comparison, so
pointing it at a lesson 9 run produced a table showing cost per success *down 11%* under
a paragraph explaining that it had "went the wrong way". Worse, checking the real
figures showed the claim was wrong about its own comparison too: baseline → strict moved
cost per case −10.2% and cost per success −3.8%. Cost per success *improved*, just far
less than the headline figure implied.

The honest lesson is a better one than the original: **cost per success does not reverse
a misleading saving so much as discount it.** Here it revealed that roughly two-thirds
of the apparent 10% saving was illusory, on a change that was a regression on accuracy.
Now computed in `tracing.cost_verdict()` with tests, and corrected in lesson 8's
README, NOTES and the index.

That makes four silent measurement bugs in this project — lesson 5's label matcher,
lesson 7's score cache, lesson 8's trace double-count, and now its commentary. **The
prose a measurement tool prints needs tests as much as its arithmetic does.**

---

## Exercises

1. **Run the noise floor properly.** `--noise --repeats 3` costs two full runs and
   replaces a provisional threshold with a usable one. Does a third repeat find flips
   the second missed?
2. **Recheck a case you believe is stable.** Pick any passing case and run
   `--recheck <id> --experiment baseline --repeats 5`. The interesting outcome is
   finding a second flake, because it changes what the whole suite can claim.
3. **Run `terse_prompt`.** It is defined, predicted and unrun. The hypothesis is that
   accuracy holds and cost drops. Check both, and check cost per *success* rather than
   cost per run.
4. **Make the invisible improvement visible.** `verify_conversion` improved refusal
   grounding and no scorer could see it. Add a case that measures grounding directly,
   then re-run `verify_conversion` — free, from cache — and see whether it becomes a
   one-variable win.
5. **Fix the phantom-tool flake.** `verify_first` is a real improvement carrying a ~20%
   instability. Options: soften the prompt's tool insistence, or handle
   `PhantomToolCall` in the loop by telling the model the tool does not exist and
   continuing rather than aborting. The second is the better product fix. Measure both.
6. **Grow the dataset.** Every caveat in this lesson traces to 16 cases. Scoring is
   free and only new questions cost tokens, so this is the highest-value change
   available and the least glamorous.

---

## What this lesson does not solve

- **16 cases cannot resolve a one-case difference.** Every Wilson interval here
  overlaps. The noise floor tells you a change reproduced; nothing here tells you it
  generalises.
- **`verify_first` is unresolved on purpose.** It fixes the target case, reaches 100%
  tool-choice accuracy, is cheaper, and flakes about 20% of the time on a different
  case. That is a genuine open trade-off, and tuning it until the number looks good
  would be exactly the behaviour this lesson argues against.
- **One flake rate is not a flake profile.** `files_quote_definition` was measured at
  ~20% under one prompt. Nothing has measured the other fifteen cases.
- **The decision rule is a starting point, not a law.** It was already wrong once and
  changed. `--replay` exists because it will be wrong again.

Next: lesson 10, multi-agent patterns — and the eval harness comes along, because
"is a second agent worth it?" is a question this lesson's machinery can answer.
