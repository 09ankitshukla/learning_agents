# Lesson 10 — Multi-agent: delegation, handoff, and whether they earn it

Multi-agent is the most over-applied pattern in this space. This lesson builds it
properly and then measures it, and the measurement is not flattering.

The goal is not a working pipeline — that part is easy. The goal is that you can say
*why* a second agent is not the answer to your problem, because most of the time it
is not, and the argument has to be quantitative to survive contact with someone
enthusiastic.

---

## The idea, in one line

**A sub-agent is a tool whose implementation happens to be another agent loop.**

That is not a teaching simplification. `run_agent` is unchanged from lesson 3. The
wire protocol is unchanged. Lesson 2's dispatcher is still the security boundary. The
parent model sees `ask_researcher` with a description and a `task` parameter, exactly
as it sees `convert_currency`, and cannot tell the difference.

So this lesson adds no new capability. It needed `ToolRegistry.subset` — written in
lesson 2 and unused until now — and nothing else. Every multi-agent framework you will
read about is this plus naming.

Which is why the content is not *how* to delegate. It is the four things that break
once you do, all of them silently.

---

## Concepts

### Delegation vs handoff

**Delegation** (`team.py`) returns control. The parent asks, gets an answer, carries on
deciding. The model chooses whether to delegate, so the route is emergent and the
parent stays responsible for the final answer.

**Handoff** (`pipeline.py`) passes control on. Stage 2 does not report back; it
continues and hands to stage 3. The sequence is fixed in code, so no model decides it.

The rule that falls out: **if you know the sequence in advance, a pipeline beats a
delegating agent**, because otherwise you are paying a model to make a decision you
already made. Delegation earns its cost only when the route depends on what is found
along the way.

### A sub-agent's failure looks exactly like its answer

Both are a string returned from a tool.

Lesson 6 established that `COMPLETED` means "the model stopped asking for tools", not
"the answer is correct". Delegation compounds it: a sub-agent that gives up politely
returns prose, the tool returns that prose, and the parent builds on it as research.

So every result carries its provenance, and **every non-completion is raised as a
`ToolError`** with the partial output explicitly marked UNVERIFIED. A pipeline stage
that fails stops the pipeline rather than passing its half-finished text along — the
alternative is laundering a failure into a confident answer three stages later.

### Least privilege stops being optional

`ToolRegistry.subset` gives each specialist only the tools it needs. The researcher has
no calculator; the calculator cannot read files; the critic has no tools at all, so it
can only judge what it is sent. Give the critic tools and it becomes a second
researcher.

### Recursion is free and unbounded

Give a sub-agent the delegation tools and it can call itself, or a peer that calls
back. Lesson 3's step cap will not catch it: from the parent's view that is still one
tool call that happens to take a while.

`DelegationBudget` caps **depth** (enforced by *withholding* the delegation tools — a
tool the model cannot see is a tool it cannot be talked into using) and **breadth** (a
total call count, because six specialists once each costs the same as one specialist
six times).

### State sharing is the hard part

`relay` passes only the previous stage's output. Cheap, and lossy in a specific way:
every fact a later stage needs has to be plumbed to it by name, and the failure mode is
a stage quietly working from less than you assumed.

`shared` continues the whole message list, so a later stage sees every tool result
directly — the only way a critic can tell a quotation from a paraphrase. It pays lesson
3's growth across the whole pipeline instead of within one agent.

Changing the system prompt mid-conversation is not possible, so `handoff_messages`
replaces message zero. Without that, a "handoff" leaves the previous agent's
instructions in place and you get one agent wearing a second agent's name.

---

## Run it

```powershell
# free, no model involved
uv run lessons/10-multi-agent/multi.py --team
uv run lessons/10-multi-agent/multi.py --recursion

# delegation
uv run lessons/10-multi-agent/multi.py --ask "which lesson covers the agent loop?"
uv run lessons/10-multi-agent/multi.py --router --ask "..."   # force delegation

# handoff
uv run lessons/10-multi-agent/multi.py --pipeline "what is a phantom tool call?"
uv run lessons/10-multi-agent/multi.py --pipeline "..." --mode shared --revise
uv run lessons/10-multi-agent/multi.py --compare "..."        # relay vs shared

# the honest comparison
uv run lessons/10-multi-agent/multi.py --probe arith_precision      # ~5k tokens
uv run lessons/10-multi-agent/multi.py --evaluate --force           # ~82k tokens
```

Start with `--probe`. Lesson 9's clearest finding was that a single case repeated
several times costs a twentieth of a suite run and often answers a sharper question —
and in this lesson it did exactly that, twice.

---

## What actually happened

All figures on Groq. Solo numbers come from lesson 7's dataset and the same tools, so
the only difference is the wiring.

### Given its own tools, the coordinator does not delegate

The first real run was a two-part question — find a lesson folder, then compute a
percentage — with the full team available:

```
step 1: search_files
step 2: search_files
step 3: calculate
step 4: final answer

sub-agents (invisible to Trajectory) : 0 tokens
```

Zero delegations. It had `ask_researcher` and `ask_calculator` and used neither,
because it already had `search_files` and `calculate` and they were obviously
sufficient. That is the correct decision and it is the lesson's central result: **a
capable agent routes around your team.**

Forcing the issue with `--router` (coordinator stripped of its own tools):

| | tokens | delegations |
|---|---|---|
| coordinator with its own tools | 6,034 | 0 |
| pure router | **9,438** | 2 |

**56% more tokens for the same question.** And the parent's own accounting reported
2,265 of those 9,438.

### 76% of a delegating run's tokens are invisible

A sub-agent's model calls happen inside `registry.dispatch()`, which is not a place
lesson 3's `Trajectory` looks — it was written before sub-agents existed and counts
only the steps it can see.

```
parent (what Trajectory.usage sees) :  2,265
sub-agents (invisible to it)        :  7,173
actual total                        :  9,438   -> 76% hidden
```

So lesson 7's per-case token counts and lesson 8's `CostReport` both under-report a
delegating agent by most of its cost. Read off `Trajectory.usage`, the *expensive*
architecture looks **62% cheaper** than the solo one (2,265 vs 6,034) when it is
actually 56% dearer. `DelegationLog` exists for this, and `--ask` prints two numbers
instead of the one you would naturally report.

This is the fifth silent measurement bug in this project and the first that was
predicted rather than discovered.

### The tax for a team you do not use is about 30%

Two cases where the coordinator delegated nothing at all:

| case | solo | team | change | delegated? |
|---|---|---|---|---|
| `no_tool_definition` | 882 | 1,151 | **+30%** | no |
| `files_quote_definition` | 5,735 | 7,620 | **+33%** | no |

Identical tool use, identical answers, a third more tokens. The cost is three extra
tool schemas and a longer coordinator prompt, re-sent on every step of every question.
**You pay for the specialists whether or not you call them**, and nothing in the output
tells you that you are.

### The one case that "broke" was the instrument, not the agent

`arith_precision` — "what is 2 divided by 7, to six decimal places?" — failed under
delegation, reproducibly, on both models:

| | solo | team |
|---|---|---|
| `gpt-oss-20b` | pass | **fail** |
| `gpt-oss-120b`, 2 runs | 2/2 pass | **0/2 pass** |
| tokens | 1,807 | 3,395 (+88%) |

The obvious reading is that precision was lost in the round trip. It was not. Asking
the delegating agent directly:

> 0.285714 (to six decimal places)

The answer is correct. The failing check was `used_tools(["calculate"])`: the
coordinator's `tool_sequence` reads `["ask_calculator"]`, because the calculator ran one
level down where `Trajectory` cannot see it.

**The blind spot that hides a sub-agent's tokens also hides its tool use.** Ten of the
sixteen eval cases assert `used_tools`, and an eleventh asserts
`answered_without_tools` — so 11 of 16 process checks silently measure the wrong thing
the moment work is delegated. A full suite run would have reported a large regression
manufactured by the harness.

`DelegationLog.effective_tool_sequence` expands delegations into the tools actually
used, and `--probe` now scores both ways so the artifact is visible:

```
architecture  run  naive  effective  tokens  tools (effective)
solo            1  yes    yes         1,807  calculate
team            1  no     yes         3,395  calculate

Measurement artifact on: team run 1 — the naive and effective verdicts disagree.
```

Nothing new had to be recorded to fix it. The tool names were in the log all along,
which is the same lesson lesson 8 learned about traces: it was a view problem, not a
collection problem.

### Sharing the whole conversation doubles the cost and bought nothing

Same topic, same stages, the two state-passing modes:

| metric | relay | shared | change |
|---|---|---|---|
| total tokens | 7,491 | 14,896 | **+99%** |
| prompt tokens | 6,732 | 14,313 | +113% |
| duration | 43.7s | 113.8s | +160% |
| critic verdict | approved | approved | same |

Doubling the spend changed no outcome here. `shared` is the only way a critic can
verify a quotation rather than trust a paraphrase, so it is not worthless — but it is a
cost you should be able to point at a reason for, and on this topic there was none.

### A pipeline has no recovery, and that showed up immediately

The first pipeline run died in stage one:

```
ask_researcher   failed  stalled, 2 step(s), 1,239 tokens
(none — pipeline stopped early)
```

Lesson 3's stall detector fired: the researcher issued the same search twice. The
pipeline correctly refused to pass a failed stage's text forward, so it produced no
answer at all.

The cause is lesson 5's motivation reappearing. The topic was "why does this project
cache the agent execution rather than the score?", which has no literal keyword match,
and the researcher only has `search_files` — keyword search. It searched, found
nothing, and searched again identically. A keyword-friendly topic ran cleanly through
all three stages for 7,671 tokens with the critic approving.

The architectural point is the one worth keeping: **a fixed pipeline cannot adapt.** A
delegating coordinator that gets a stall error back can try a different approach; a
pipeline stage that fails ends the run. That is the price of the predictability that
made the pipeline cheaper.

### `--evaluate` is implemented and deliberately unrun

The full 16-case comparison costs ~82,000 tokens. It is not run, for two reasons, and
both are the lesson's own methodology applied rather than described.

First, the daily quota went before it could be: the 20b model hit its 200,000-token
ceiling (`Used 199252`) mid-probe, which is itself the measured fact that **multi-agent
work exhausts a token budget several times faster than single-agent work**.

Second, and more importantly, five probes at ~25,000 tokens total had already answered
the question more sharply than the suite would: delegation fixed nothing, cost 30% more
when unused and ~90% more when used, and its single apparent regression was a
measurement artifact. Lesson 9's finding was that single-case probes beat suite runs
when a result turns on one or two cases. Spending 82,000 tokens to confirm what 25,000
already showed would be ignoring that.

So the prediction stands unscored in the strict sense. Its **direction** — worse and
more expensive — is confirmed. Its **specifics** were wrong: it named
`files_quote_definition` and `no_tool_definition` as breakages and both passed, and it
attributed `arith_precision` to lost precision when the cause was the scorer. That is
the same pattern lesson 9 measured: predicting *that* something will go wrong is much
easier than predicting *what*.

---

## When a second agent is actually worth it

Nothing in this lesson's measurements supports delegation, so it is worth being precise
about what would.

- **The route depends on what you find.** A fixed sequence is a pipeline; only genuine
  branching needs a model to choose.
- **Context isolation is the point.** A sub-agent starting fresh is a feature when the
  parent's conversation is huge and irrelevant to the sub-task — lesson 4's problem
  solved by architecture.
- **The specialist is genuinely different.** A different model, a different tool set
  with real privilege boundaries, a different provider. Three prompts over one model
  and one tool set, as here, is mostly overhead.
- **Verification by a party that cannot see the work.** The critic is the one component
  here that earns its place, and only because it has no tools and no view of how the
  answer was produced.

If none of those apply, you are paying 30% for tool schemas you will not call.

---

## Exercises

1. **Find a question where delegation wins.** It should need branching the coordinator
   cannot plan. Probe it both ways. If you cannot find one, that is the result.
2. **Run `--evaluate --force`** when you have the quota, and check whether the naive and
   effective scorings differ across the suite as they did on `arith_precision`.
3. **Give the researcher lesson 5's semantic retrieval.** The stall came from keyword
   search on a paraphrased question — the exact failure lesson 5 exists to fix. Measure
   whether the pipeline stops stalling.
4. **Measure the critic's actual value.** Run the pipeline with and without it on ten
   topics and count how often it catches something real versus rubber-stamping. Lesson
   8's judge can score the pairs.
5. **Break the depth guard on purpose.** Set `max_depth=3` and give a sub-agent the
   delegation tools. Watch the cost, then decide whether the default of 1 was right.
6. **Fix the cost blind spot properly.** `DelegationLog` patches it for this lesson
   only. Making `Trajectory` aware of nested usage would fix it for lessons 7 and 8 too
   — is that worth changing lesson 3's data model for?

---

## What this lesson does not solve

- **`Trajectory` still under-reports nested cost.** `DelegationLog` works around it from
  outside; anything that reads `Trajectory.usage` directly is still wrong.
- **11 of 16 eval cases have architecture-dependent process checks.** The expansion fix
  handles delegation. It would not handle a differently-shaped architecture, and the
  general problem — process assertions assume a fixed topology — is unsolved.
- **The pipeline's researcher has no semantic retrieval**, so it stalls on paraphrased
  questions. Lesson 5 built the fix and the two were not wired together.
- **One topic measured for relay vs shared.** A topic where the critic needs verbatim
  evidence would likely favour `shared`, and none was tested.
- **The full suite comparison is unrun**, so the strongest claims here rest on five
  probes rather than sixteen cases.

Next: lesson 11, guardrails and failure modes — where the sub-agent boundary becomes a
security boundary, and `ToolRegistry.subset` stops being a tidiness measure.
