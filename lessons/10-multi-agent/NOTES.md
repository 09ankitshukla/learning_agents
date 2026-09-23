# Lesson 10 — revision notes

## The one idea

**A sub-agent is a tool whose implementation is another agent loop.**

`run_agent` unchanged, wire protocol unchanged, lesson 2's dispatcher still the security
boundary. Needed exactly one previously-unused thing: `ToolRegistry.subset`. Every
multi-agent framework is this plus naming.

## Delegation vs handoff

| | delegation | handoff |
|---|---|---|
| control | returns to the parent | passes on |
| who chooses the route | the model | you, in code |
| adapts to what it finds | yes | no |
| cost | higher | lower |
| recovery from a failed step | possible | none |

**If you know the sequence in advance, a pipeline beats a delegating agent** — otherwise
you are paying a model to make a decision you already made.

## Measured results

Groq, `gpt-oss-20b` and `gpt-oss-120b`, against lesson 7's dataset and tools.

**Given its own tools, the coordinator does not delegate.** Full team available, a
two-part question: zero delegations, 0 sub-agent tokens. It used `search_files` and
`calculate` itself. **A capable agent routes around your team.**

| comparison | tokens | note |
|---|---|---|
| coordinator with own tools | 6,034 | 0 delegations |
| pure router (`--router`) | 9,438 | **+56%**, 2 delegations |
| `no_tool_definition` solo → team | 882 → 1,151 | **+30%**, delegated nothing |
| `files_quote_definition` solo → team | 5,735 → 7,620 | **+33%**, delegated nothing |
| `arith_precision` solo → team | 1,807 → 3,395 | **+88%**, delegated |
| pipeline relay → shared | 7,491 → 14,896 | **+99%** tokens, +160% time, same verdict |

**76% of a delegating run's tokens are invisible to `Trajectory`.** Parent saw 2,265 of
9,438. Read off `Trajectory.usage` the expensive architecture looks **62% cheaper**
(2,265 vs 6,034) when it is 56% dearer.

**The tax for a team you never call is ~30%.** Three extra tool schemas plus a longer
coordinator prompt, re-sent every step. Nothing in the output tells you.

## Findings worth keeping

**A sub-agent's failure looks exactly like its answer** — both are a string from a tool.
Lesson 6's "COMPLETED ≠ correct", one level deeper and invisible. So every
non-completion is raised as a `ToolError` with the partial output marked UNVERIFIED, and
a failed pipeline stage stops the pipeline. Otherwise you launder a failure into a
confident answer three stages later.

**Cost disappears into `registry.dispatch()`.** `Trajectory` was written before
sub-agents existed and counts only the steps it can see. Fifth silent measurement bug in
this project, and the first that was predicted rather than discovered — which is why
`DelegationLog` exists before it was needed.

**The same blind spot breaks the scorers.** `arith_precision` failed under delegation on
both models, reproducibly — and the answer was correct (`0.285714`). What failed was
`used_tools(["calculate"])`: the parent's sequence reads `["ask_calculator"]`.
**10 of 16 eval cases assert `used_tools`, and an 11th asserts
`answered_without_tools`**, so most process checks silently measure the wrong thing once
work is delegated. A full suite run would have reported a regression manufactured by the
harness. Fixed by `DelegationLog.effective_tool_sequence`, and `--probe` now prints both
scorings so the artifact stays visible.

Nothing new had to be recorded to fix it — the tool names were already in the log. Same
lesson as lesson 8's traces: **a view problem, not a collection problem.**

**Recursion needs its own guard.** Lesson 3's step cap does not catch it: from the
parent's view a nested delegation is one tool call that takes a while. Depth is enforced
by *withholding* the delegation tools rather than refusing the call — a tool the model
cannot see is a tool it cannot be talked into using. Breadth needs a separate counter,
because six specialists once each costs the same as one specialist six times.

**You cannot change a system prompt mid-conversation.** A naive handoff via
`initial_messages` leaves the previous agent's instructions in place, so you get one
agent wearing a second agent's name. `handoff_messages` replaces message zero; every
other message is something that actually happened and must not be rewritten.

**Relay mode is lossy in a quiet way.** `{previous}` only carries the last stage's
output, so the reviser needed the draft plumbed to it explicitly by name. The failure
mode is a stage working from less than you assumed and nothing reporting it.

**A pipeline cannot recover.** The first run died in stage one: the researcher stalled
(lesson 3's detector) because the topic had no literal keyword match and it only has
`search_files`. Lesson 5's motivation, reappearing. A delegating coordinator could have
tried something else; the pipeline just ended.

**Multi-agent burns quota several times faster.** Hit the 200,000/day ceiling mid-probe
(`Used 199252`) on a lesson that ran fewer live commands than lesson 9.

## Two descriptions, two audiences

`SubAgent.description` is read by the **parent**, to decide whether to delegate — a
capability and its limits. `SubAgent.system_prompt` is read by the **sub-agent**, to
decide how to behave — a job. Writing one and reusing it for both is the most common
multi-agent mistake: the parent cannot tell what the agent is for, or the agent has no
instructions of its own.

## When a second agent is worth it

- the route genuinely depends on what you find (otherwise: pipeline)
- context isolation is the point — a fresh sub-agent is lesson 4's problem solved by architecture
- the specialist is genuinely different: different model, real privilege boundary, different provider
- verification by a party that cannot see the work (the critic, which has no tools)

Three prompts over one model and one tool set — as here — is mostly overhead.

## Commands

```powershell
uv run lessons/10-multi-agent/multi.py --team        # free
uv run lessons/10-multi-agent/multi.py --recursion   # free
uv run lessons/10-multi-agent/multi.py --ask "..."
uv run lessons/10-multi-agent/multi.py --router --ask "..."     # force delegation
uv run lessons/10-multi-agent/multi.py --pipeline "..." --mode shared --revise
uv run lessons/10-multi-agent/multi.py --compare "..."          # relay vs shared
uv run lessons/10-multi-agent/multi.py --probe CASE_ID          # ~5k, do this first
uv run lessons/10-multi-agent/multi.py --evaluate --force       # ~82k
```

## Open

- `--evaluate` is implemented and unrun: the quota went, and five probes at ~25k had
  already answered the question more sharply than 82k would. Lesson 9's method applied.
- The prediction stands unscored. Direction (worse, dearer) confirmed; specifics wrong —
  it named two cases that passed and misattributed the third. Same pattern as lesson 9:
  predicting *that* something breaks is far easier than predicting *what*.
- `Trajectory` still under-reports nested cost; `DelegationLog` patches it from outside.
- Process assertions assume a fixed topology. The expansion fix handles delegation only.
- The researcher has no semantic retrieval, so it stalls on paraphrased questions.
  Lesson 5 built the fix; the two were never wired together.
