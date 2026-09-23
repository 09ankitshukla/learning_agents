# Lesson 6 — Notes

## The one idea

**"Agents are non-deterministic so they cannot be tested" is half true, and believing it leads to testing nothing.**

Most of an agent is ordinary code: the dispatcher, the sandbox, the calculator, trimming, the token estimator, chunking, keyword search. None of it involves a model, all of it is where correctness and security live, and all of it is testable normally.

The model is the only stochastic part, and you substitute it.

## Substitute at the seam, not at the HTTP layer

Patching `httpx` or the OpenAI SDK tests that your code can talk to a fake HTTP server — a test of the SDK, not of your agent. The interesting logic sits above that.

`llmkit.LLMClient` is a Protocol whose only significant method is:

```python
chat(messages, tools=None, ...) -> LLMResponse
```

So anything with that shape is a drop-in model. That Protocol was written in lesson 0 for provider-swapping; it turns out to be the seam that makes the whole project testable. **Designing for a substitutable boundary before you need one is most of what makes code testable later.**

## Two doubles, two jobs

**`ScriptedClient`** — you write the responses. Best for specific behaviour, especially failures you cannot reliably provoke from a real model:

```python
ScriptedClient([
    tool_response("calculate", malformed='{"expression": "2 +'),
    text_response("fixed it"),
])
```

Token starvation, malformed tool arguments, a phantom tool, an infinite tool loop — all trivial to construct, all impossible to summon on demand from a real model.

**`CassetteClient`** — replays responses recorded from a real model. Preserves the quirks you would not have thought to fake: `content: null` on tool turns, reasoning-token counts, exact argument JSON, the `finish_reason` values you did not expect.

**Use both.** Scripted doubles test the paths you care about; cassettes stop your fakes drifting into fiction. When the two disagree, the cassette is right.

## Assert on the trajectory, never on prose

In order of usefulness:

1. `stop_reason` — did it finish, or give up, and why?
2. `tool_sequence` — did it take the right steps?
3. **the requests the double received** — did we send a valid conversation?
4. the final text — almost never.

Point 3 is the underused one. Many agent bugs live in what you *send*: a trimmed conversation that orphans a tool result, tools missing from a later call, a budget that is too small. Those are only visible if your double records requests. So `ScriptedClient` keeps every request — and must **copy** the message list, because the loop keeps mutating its own.

## The doubles need their own tests

Two bugs in my test doubles, both of which would have produced false results:

**The cassette replayed entry 0 twice.** Key lookups and the sequential fallback used separate counters, so a key hit did not advance the position. The agent saw two identical tool calls and the loop reported `STALLED` — a bug in the test double masquerading as a bug in the code under test. Now a single `_used` set governs both paths.

**Running out of responses must be an error.** If a double silently repeats its last response, a runaway 40-iteration loop passes as green. `ScriptedClient(strict=True)` raises instead.

**A lying fake is worse than no test.** Test the harness.

## Live tests are opt-in, and few

```powershell
uv run pytest lessons/06-testing          # 111 tests, 0.9s, free, offline
uv run pytest lessons/06-testing -m live  # 8 tests, hits the API
```

Deselected by default via `pytest_collection_modifyitems`. **A suite you avoid running because it is expensive provides no safety.** The default invocation must be safe to run on a plane and a hundred times an hour.

What belongs in live tests is narrow: assumptions about the provider that a double cannot check. Does the configured model still exist (this project already lost one mid-course)? Does tool calling still work? Does `max_tokens=512` still yield visible text on a reasoning model? Is the token estimator still in the right ballpark? These are closer to monitoring than unit testing.

They should also be tolerant. The estimator test allows −35% to +55% because the point is to catch a *structural* break — a new chat template, a different tokenizer — not to chase a few percent.

## What the suite found on first run

**A real bug, immediately: two classes named `ToolError`.**

Lesson 2 defined its own. Lesson 3 promoted the dispatcher into `llmkit.tools.ToolRegistry`, which caught `llmkit`'s `ToolError` — a different class. Lesson 2's tool functions therefore fell through to the generic `except Exception`.

Effect: every timezone and currency error in lessons 3, 4 and 5 was reported as `failed unexpectedly (ToolError)` instead of `Unknown timezone 'Mumbai'. Use an IANA name such as 'Asia/Kolkata'`. The message that makes a model self-correct was replaced by one that says nothing — silently undoing the behaviour lesson 2 spends a whole scenario demonstrating.

**Completely invisible to inspection.** Both classes have the same name, so the code reads correctly at every site. It took an assertion on the error *text* to surface it.

Fixed by importing `ToolError` from `llmkit` in lesson 2 rather than redeclaring it.

## Then I broke the lesson's own rule

My live end-to-end test asserted `"6319" in final_answer` after stripping commas. It failed: the model wrote the answer as LaTeX, `6{,}319`, so comma-stripping produced `6{}319`. The arithmetic was perfect and my assertion was wrong.

Rewritten to check the `calculate` tool's **result** rather than the prose. Tool results are deterministic; how a model chooses to format them is not.

## Property-shaped assertions beat example-shaped ones

For trimming, the useful test is not "it produces exactly these messages" but:

```python
for budget in range(200, total + 200, 150):
    assert validate(trim_safe(messages, budget).messages) == []
```

*Whatever* it produces is structurally valid, across many budgets. That would have caught the orphan bug directly.

Similarly, `test_every_request_is_a_valid_conversation` checks all recorded requests rather than just the last, because the invalid one is usually in the middle.

## Keep a counterexample test

`test_naive_trim_can_produce_an_invalid_conversation` asserts that the deliberately-broken function *is* broken. If someone "fixes" `trim_naive`, lesson 4's demonstration silently stops demonstrating anything. Pinning intentional badness is legitimate.

## The regression suite is the honest documentation

One test per bug this project actually shipped: the `ToolError` split, the orphaned tool result, the starved summariser, the fallback that silently did nothing, the 91%-low token estimator, teaching prose leaking into a prompt schema, the provider error mis-mapping, the Windows Unicode crash.

Each cost real debugging time. Each test costs milliseconds and runs forever. **Read the docstrings even if you skip the assertions** — the failure modes are more instructive than the fixes.

## Practical wrinkle worth naming

Lesson folders are named `03-agent-loop`, which is not a valid Python identifier, so they cannot be packages. `conftest.py` puts every lesson directory on `sys.path`.

That is a genuine cost of organising code by lesson rather than as a package, and it is worth naming rather than hiding. In a real project you would have `src/myagent/loop.py` and import it normally.

One collision to know: lessons 2–5 all contain an `agent.py`, so the tests never import `agent` — only the uniquely-named library modules. Lesson 5's `store.py` was named that way specifically to avoid clashing with lesson 2's `tools.py`.

## Carry forward

- 111 offline tests in under a second is the property that matters. Speed is what makes a suite get run.
- Cassettes are committed fixtures. A fresh clone runs the whole suite with no API key.
- `queries.py` plus this infrastructure is most of lesson 7. The missing piece is scoring *task success* rather than mechanics — `COMPLETED` still does not mean correct.
