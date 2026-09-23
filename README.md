# Learning Agents

Building LLM agents from scratch, one runnable agent per lesson. No frameworks until you understand what they'd be hiding.

Every lesson runs on **open-source models on your own machine** with no API keys. Every lesson also runs unchanged against a hosted API if you'd rather. That's a one-line config change, never a code change.

---

## What this is

A learn-by-doing course where each lesson ends with a working agent you can clone, run, and read. The order is deliberate: mechanics first, then capability, then how to tell whether any of it works, then how to ship it.

Each lesson folder contains:

- `README.md` — the concept explained first, then the exercises
- runnable Python — the agent itself, commented as teaching material
- `NOTES.md` — a revision sheet: the one idea, the rules of thumb, the gotchas

For a fast refresher across everything, read **[docs/00-index.md](docs/00-index.md)** — the condensed version of every lesson's key learnings.

## Quickstart

```powershell
winget install --id=astral-sh.uv -e     # Python toolchain (installs Python too)
uv sync                                 # create .venv, install pinned deps
                                        # (add --extra retrieval for lesson 5)

Copy-Item .env.example .env             # then add a free Groq key
uv run lessons/00-setup/check_env.py    # verifies setup, measures your model
```

Getting the key takes a minute at [console.groq.com](https://console.groq.com) and costs nothing. Prefer to run models on your own machine with no key at all? That's Path B in lesson 0, and every lesson works there unchanged.

Full walkthrough with both paths, macOS/Linux equivalents, and troubleshooting: **[lessons/00-setup/README.md](lessons/00-setup/README.md)**.

## Choose your model

Three tiers, one interface. Edit `.env`, change nothing else.

| Tier | Setup | Speed | When |
|---|---|---|---|
| **Hosted open weights** | free Groq key | ~240 tok/s | recommended start; open models, no download |
| **Local** | Ollama + Qwen2.5 | ~5–8 tok/s on CPU | fully offline, zero cost, no key |
| **Commercial** | OpenAI or Anthropic key | fast | if you'd rather use them |

Seven of the eight supported providers speak the same OpenAI-compatible HTTP API, so one thin client covers Ollama, llama.cpp, vLLM, LM Studio, Groq, OpenRouter, Together and OpenAI. Anthropic gets a small adapter. That's the entire `src/llmkit` layer.

Note that hosted providers retire models regularly — `check_env.py` prints the live list, which you should trust over any documentation including this file.

## The curriculum

**Part 1 — Mechanics.** Build the loop by hand so you know what it is.

| # | Lesson | Concepts | Deliverable |
|---|---|---|---|
| 00 | [Setup](lessons/00-setup) | tokens, context windows, temperature, reasoning models, quantisation | benchmarked environment |
| 01 | [Structured output](lessons/01-structured-output) | model as unreliable function; validate and repair | support-ticket triage extractor |
| 02 | [Tool calling by hand](lessons/02-tool-calling) | tool schemas, the call/result protocol, dispatchers as a security boundary, 6 failure modes | multi-tool assistant, no framework |
| 03 | [The agent loop](lessons/03-agent-loop) | the loop, step caps, stop reasons, stall detection, trajectories, sandboxing | multi-step research agent |

**Part 2 — Capability.** Make it useful.

| # | Lesson | Concepts | Deliverable |
|---|---|---|---|
| 04 | [Memory and context](lessons/04-memory-context) | token accounting, atomic message groups, trimming, summarisation, persistence | agent that stays inside a budget |
| 05 | [Retrieval](lessons/05-retrieval) | chunking, embeddings, vector search, keyword vs semantic vs hybrid, measured | agent that answers over your own notes |
| 06 | [Testing agents](lessons/06-testing) | deterministic vs stochastic parts, scripted doubles, cassettes, trajectory assertions | 111 offline tests in under a second |

**Part 3 — Knowing whether it works.** The part most tutorials skip.

| # | Lesson | Concepts | Deliverable |
|---|---|---|---|
| 07 | [Evaluation](lessons/07-evaluation) | eval datasets, deterministic scorers, fabrication guards, cached runs, regression detection | eval harness + scorecard |
| 08 | Judging and tracing | LLM-as-judge and its biases, spans, token/cost/latency accounting | trace viewer + cost report |
| 09 | Iteration | A/B testing prompts, models and tool designs against your evals | a measured improvement |

**Part 4 — Real systems.**

| # | Lesson | Concepts | Deliverable |
|---|---|---|---|
| 10 | Multi-agent | delegation, handoffs, shared state, when one agent is better | research/write/critique pipeline |
| 11 | Guardrails and failure modes | prompt injection, tool sandboxing, allowlists, approval gates | hardened agent + injection demo |
| 12 | Deployment | HTTP API, streaming, auth, containers, config and secrets | Dockerised service, deployed |
| 13 | Framework comparison | rebuild lesson 3 in a framework; see exactly what's abstracted | side-by-side implementations |

Testing sits at lesson 6, not lesson 12, on purpose: build five agents with no tests and adding tests later means rewriting all five.

Frameworks come last, also on purpose. LangGraph and friends are good tools, but they hide the agent loop — which is the thing you're here to learn. By lesson 13 you'll be able to judge them instead of just adopting them.

## Repo layout

```
src/llmkit/          shared model layer (config, providers, types) - read it once, then ignore it
lessons/NN-name/     one lesson: README, code, NOTES
docs/00-index.md     condensed key learnings across all lessons
docs/glossary.md     terminology
```

Everything runs from the repo root:

```powershell
uv run lessons/01-structured-output/extract.py
```

## Status

Lessons 00 through 07 are complete and verified against a live model. Lesson 08 is next.

There's a test suite from lesson 6, and an eval harness from lesson 7:

```powershell
uv sync --all-extras
uv run pytest lessons                    # 153 tests, offline, ~2 seconds
uv run pytest lessons -m live            # 8 more, hits the API

uv run lessons/07-evaluation/evaluate.py --show baseline      # 15/16, from a saved run
uv run lessons/07-evaluation/evaluate.py --compare baseline strict
```

The two committed eval runs let you see a measured regression without spending any tokens.

By the end of lesson 3 you have a working agent: it plans across multiple steps, recovers from its own mistakes, refuses to escape its sandbox, and reports honestly when it gives up. Lesson 4 keeps it inside a token budget and lets a session survive a restart.

Resuming after a break, or picking this up cold? Start with **[docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)** — current status, environment, design decisions and the findings that cost real debugging time.

## Licence

MIT. Use it, fork it, teach from it.
