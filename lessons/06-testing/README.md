# Lesson 6 — Testing agents

**Time:** 1.5–2 hours
**You will end with:** 111 tests that run offline in under a second, two kinds of model double, recorded cassettes, and a regression suite covering every bug this project actually shipped
**Depends on:** lessons 0–5

---

## Setup

```powershell
uv sync --extra retrieval --extra dev
uv run pytest lessons/06-testing
```

Expect **111 passed, 8 skipped in ~1 second**. The 8 skipped are live tests, deselected by default.

---

## Learn first

### The excuse to get past

"Agents are non-deterministic, so they can't really be tested."

Half true, and believing it leads to testing nothing. Look at what an agent is actually made of:

| Component | Deterministic? |
|---|---|
| the dispatcher | yes |
| the path sandbox | yes |
| the AST calculator | yes |
| trimming and `validate()` | yes |
| the token estimator | yes |
| chunking | yes |
| keyword search | yes |
| **what the model chooses to say** | **no** |

One row is stochastic. Every other row is where your security and correctness properties live, and all of them test normally. Lesson 6 starts with those, in `test_deterministic.py`, because that's the part people skip.

### Then substitute the model

Not by patching HTTP. If you mock `httpx` or the OpenAI SDK you end up asserting that your code can talk to a fake HTTP server — a test of the SDK, not of your agent.

Substitute at the seam. `llmkit.LLMClient` is a Protocol with one method that matters:

```python
chat(messages, tools=None, ...) -> LLMResponse
```

Anything with that shape is a drop-in model. That Protocol was written back in lesson 0 for provider-swapping, and it turns out to be the seam that makes the whole project testable. **Designing for a substitutable boundary before you need one is most of what makes code testable later.**

### Two doubles, two jobs

**`ScriptedClient`** — you write the responses by hand:

```python
client = ScriptedClient([
    tool_response("calculate", malformed='{"expression": "2 +'),
    text_response("fixed it"),
])
trajectory = run_agent(client, "q", registry, max_steps=5)
assert trajectory.failed_executions[0].failure_kind == "malformed_json"
```

This is how you test failures you cannot summon on demand: token starvation, malformed arguments, a phantom tool, a loop that never terminates.

**`CassetteClient`** — replays responses recorded from a real model, preserving the quirks you wouldn't think to fake: `content: null` on tool turns, reasoning-token counts, the exact argument JSON.

Use both. Scripted doubles test the paths you care about; cassettes stop your fakes drifting into fiction. When they disagree, the cassette is right.

### What to assert on

In order of usefulness:

1. **`stop_reason`** — did it finish, or give up, and why?
2. **`tool_sequence`** — did it take the right steps?
3. **the requests your double received** — did you send a valid conversation?
4. the final text — almost never.

Point 3 is the underused one. Plenty of agent bugs live in what you *send*: a trimmed conversation that orphans a tool result, tools missing from a later call. Those are only visible if the double records requests — which is why `ScriptedClient` keeps every one, and why it **copies** the message list, since the loop keeps mutating its own.

Point 4 deserves emphasis because I got it wrong in this very lesson. My live test asserted `"6319" in final_answer` after stripping commas; the model wrote `6{,}319` in LaTeX, so it became `6{}319` and failed. The arithmetic was perfect and my assertion was wrong. It now checks the tool's result instead.

### Property-shaped beats example-shaped

For trimming, the useful test isn't "it produces exactly these messages":

```python
for budget in range(200, total + 200, 150):
    assert validate(trim_safe(messages, budget).messages) == []
```

*Whatever* it produces is structurally valid, across many budgets. That would have caught lesson 4's orphan bug directly.

---

## Then apply

### Run it

```powershell
uv run pytest lessons/06-testing              # offline, free, ~1s
uv run pytest lessons/06-testing -v           # see every test name
uv run pytest lessons/06-testing -m live      # the 8 live tests
uv run pytest lessons/06-testing -k Sandbox   # just the sandbox tests
```

Live tests are opt-in via `pytest_collection_modifyitems` in `conftest.py`. **A suite you avoid running because it's expensive provides no safety** — the default invocation has to be safe to run on a plane and a hundred times an hour.

### Record your own cassettes

```powershell
uv run lessons/06-testing/record.py --list
uv run lessons/06-testing/record.py --all
```

This is the only script here that costs tokens, run once per scenario and then committed. Cassettes are test fixtures: small, deliberate, and they let a fresh clone run the whole suite with no API key.

### What the suite found on its first run

A real bug, within seconds: **two different classes named `ToolError`.**

Lesson 2 defined its own. Lesson 3 promoted the dispatcher into `llmkit.tools.ToolRegistry`, which caught `llmkit`'s `ToolError` — a different class — so lesson 2's tool functions fell through to the generic `except Exception`.

The effect was that every timezone and currency error in lessons 3, 4 and 5 came back as `failed unexpectedly (ToolError)` instead of `Unknown timezone 'Mumbai'. Use an IANA name such as 'Asia/Kolkata'`. The message that makes a model self-correct had been silently replaced by one that says nothing — quietly undoing the behaviour lesson 2 spends a whole scenario demonstrating.

It was **invisible to inspection**, because both classes have the same name and every call site reads correctly. Only an assertion on the error *text* surfaced it.

That's the honest argument for this lesson. Not "tests are good practice" but "there was a real bug in code you already read twice, and a test found it in under a second."

### The regression suite

`test_regressions.py` has one test per bug this project actually shipped:

| Bug | Why it hid |
|---|---|
| two `ToolError` classes | same name, code reads correctly |
| orphaned tool result → HTTP 400 | naive trimming only *sometimes* splits a group |
| summariser starved at 400 tokens | call succeeded, tokens billed, output empty |
| fallback trimmed to a hardcoded budget | reported success while doing nothing |
| token estimator 91% low | no ground truth until measured |
| teaching prose leaked into a prompt schema | Pydantic serialises docstrings |
| 401 reported as "server down" | wrong diagnosis sends you to the wrong place |
| Windows Unicode crash | failed at print time, far from the cause |

Read the docstrings even if you skip the assertions — the failure modes are more instructive than the fixes.

### Make changes

1. **Break something and watch a test catch it.** Delete the `except ToolError` branch in `llmkit/tools.py`, run the suite. Then restore it. That loop — break, observe the failure message, fix — is how you learn whether a test is actually useful.
2. **Make a test lie.** Set `ScriptedClient(strict=False)` in `test_running_out_of_responses_is_an_error` and see a runaway loop pass as green. This is why the double raises.
3. **Add a regression test for a bug you hit.** As you continue, every bug you debug should end with a test. That's the habit worth building, more than any technique here.
4. **Re-record a cassette after changing the system prompt.** Watch `strict_match=True` reject the stale recording, then re-record. This is the maintenance cost of cassettes, and it's worth feeling once.
5. **Write a property test for chunking.** Assert no chunk exceeds `DEFAULT_MAX_CHARS` for *any* chunk size between 200 and 3000, not just the default.
6. **Measure coverage.** `uv run --with pytest-cov pytest lessons/06-testing --cov=src/llmkit`. Then resist optimising the number — coverage shows what is *unexecuted*, not what is *correct*.

---

## Checkpoint

You should be able to answer these without looking:

- Which parts of an agent are deterministic, and why does that matter?
- Why substitute at the `LLMClient` seam rather than mocking HTTP?
- When would you reach for a scripted double, and when for a cassette?
- Why must a double record the requests it receives, and why copy them?
- Why must running out of scripted responses be an error?
- Why are live tests opt-in?
- Why assert on `tool_sequence` rather than on the answer text?

Then read [NOTES.md](./NOTES.md).

## Files

| File | What it is |
|---|---|
| `fakes.py` | **The lesson.** `ScriptedClient`, `CassetteClient`, `RecordingClient`, response builders. |
| `conftest.py` | Makes earlier lessons importable; registers the `live` marker and skip logic. |
| `test_deterministic.py` | 56 tests for the non-model parts: sandbox, dispatcher, trimming, chunking. |
| `test_loop.py` | The agent loop under scripted doubles: stop reasons, recovery, requests sent. |
| `test_regressions.py` | One test per bug this project shipped. |
| `test_cassettes.py` | Replays recorded runs; asserts on real model quirks. |
| `test_live.py` | 8 opt-in tests: provider contract, estimator drift, end-to-end. |
| `record.py` | Records cassettes. The only script here that costs tokens. |
| `cassettes/` | Committed fixtures. |

**Next:** lesson 07, evaluation. This lesson tests whether the machinery works; lesson 7 asks whether the agent is any *good* — `COMPLETED` still doesn't mean correct.
