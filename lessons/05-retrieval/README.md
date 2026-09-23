# Lesson 5 — Retrieval

**Time:** 1.5–2 hours
**You will end with:** an agent that searches your own notes by meaning, a labelled query set, and measured evidence of how well it actually works
**Depends on:** lessons 0–4

---

## Setup

This is the first lesson with an extra dependency:

```powershell
uv sync --extra retrieval
```

Two things worth knowing before you run it. **Groq serves no embedding models** — I checked the live list, and it's chat, safety classifiers and speech only. So embeddings run locally here whatever provider you use for chat.

The library is `fastembed` rather than the more common `sentence-transformers`, because it runs the model through ONNX and needs no PyTorch: a ~30 MB quantised download instead of a multi-hundred-MB torch wheel. Measured on this CPU-only machine, embedding 193 chunks takes a couple of seconds.

This also forced `requires-python` from 3.11 up to 3.12, since fastembed needs `numpy>=2.3` which needs 3.12. In fairness 3.11 was only ever *claimed* — every lesson was written and tested on 3.12.

---

## Learn first

### The problem lesson 3 left behind

Lesson 3 gave the agent `search_files`, which matches literal words. Its own tool description admits the limitation:

> Matching is case-insensitive and literal, not semantic — you must guess the actual wording used.

So asking "why not use eval?" works, because the note says "eval". Asking **"how do we stop the model running dangerous code?"** finds nothing, because the note says `eval`, `AST` and `allowlist` and never says "dangerous".

That's not a tuning problem. It's structural: keyword search can only match words that are literally present.

### Embeddings, and the one line that matters

An embedding turns text into a vector positioned so that *similar meanings land near each other*. Once documents and query are both vectors, "most relevant" becomes "nearest", and nearness is a dot product.

```python
query_vector = embed(query)                  # one vector
scores = document_vectors @ query_vector     # similarity search
order = np.argsort(-scores)[:top_k]          # ranking
```

That's the whole mechanism. There's no vector database in this lesson and none is needed for a few hundred chunks — a numpy array and one matrix multiply beat any database at this scale, and writing it out keeps it visible.

Normalising every vector to unit length up front is what lets cosine similarity reduce to a plain dot product: `cos(a,b) = a·b / (|a||b|)`, so if both are length 1 it's just `a·b`.

### Retrieval is the answer to lesson 4's problem

Lesson 4 was about salvaging a context window that was already full. Retrieval is about not filling it.

Measured on this repo: the corpus is 132,590 characters (~33,000 tokens). Retrieving the top 3 chunks for one question sent 2,942 characters — **2.2% of the corpus**, 883 input tokens.

It's a trade, though, not a free win. Stuffing cannot miss; the answer is definitely in there somewhere. Retrieval can miss, and then the model answers confidently from the wrong three chunks. That's why the accuracy numbers below matter, and why the system prompt explicitly permits "the notes don't cover this."

### Similarity scores are nearly meaningless in absolute terms

Cosine similarity sits in a narrow band — typically 0.6 to 0.8 even for a poor match. Only the **ranking** and the **gap between first and second** carry information.

This has a practical consequence that catches people: a threshold like "only use hits above 0.75" looks principled and is almost impossible to tune, because the band shifts with the model and the corpus. Prefer top-k plus a gap check.

---

## Then apply

### Build the index

```powershell
uv run lessons/05-retrieval/agent.py --build
```

193 chunks from 14 files, 384-dimensional vectors, cached to disk. The cache is keyed on a fingerprint of the content plus the model name, so editing a note invalidates it automatically — a cache keyed by filename alone would silently serve vectors for text that no longer exists.

### Compare the three methods on one query

```powershell
uv run lessons/05-retrieval/agent.py --search "how do I stop a tool reading files outside the project"
```

Keyword returns four chunks from `PROJECT_STATE.md`, because the word "project" appears in its heading. Semantic finds the actual sandboxing material in lesson 3. That contrast is the lesson in one command.

### Measure it properly

```powershell
uv run lessons/05-retrieval/agent.py --measure
```

Ten labelled queries, each deliberately phrased to avoid the target note's own vocabulary. If a question reuses the document's words, keyword search already works and embeddings add nothing.

| method | top-1 | recall@4 |
|---|---|---|
| keyword | 0/10 | 7/10 |
| semantic | 7/10 | 8/10 |
| hybrid | 7/10 | 9/10 |

Two things to read carefully here.

**recall@k is the metric that matters for an agent.** It reads all k results, so a correct chunk ranked third is still useful. top-1 matters when you show a single answer to a user. Keyword's 0/10 top-1 looks catastrophic until you see its 7/10 recall@4.

**A one-query margin is noise.** Ten queries cannot resolve "hybrid beats semantic by 1". Fixing that is lesson 7.

### The experiment that refuted itself

```powershell
uv run lessons/05-retrieval/agent.py --chunking
```

The theory says you should embed a chunk's heading alongside its body, since a chunk lifted from mid-document has lost its context. Tested on a hand-written sample of 8 chunks it worked well: top-1 went 4/7 → 6/7, beating what a model with twice the dimensions achieved.

On the real 193-chunk corpus it **did not replicate**: 8/10 body-only against 7/10 with the heading, recall identical.

Why the small sample misled: it had no competing documents, and every heading was informative. At scale many headings are generic ("The one idea", "Carry forward") and add noise. **A result from a tiny hand-picked sample is a hypothesis, not a finding.**

The heading is still embedded — for citation quality, not ranking. Saying so plainly is the point.

### Retrieval versus stuffing

```powershell
uv run lessons/05-retrieval/agent.py --stuffing
```

The token comparison above, plus a real answer generated from only the retrieved chunks, with citations.

### The deliverable

```powershell
uv run lessons/05-retrieval/agent.py --ask "Why did the summariser return an empty result and how was it fixed?"
```

An agent with seven tools, including both `search_files` (literal) and `search_notes` (semantic). Observed trajectory:

```
search_notes -> read_file -> read_file -> answer
```

It searched semantically to find *where* the answer lived, then read the files for full detail, then answered with line-level citations. That's the intended shape, and it's why `search_notes` returns truncated excerpts rather than whole documents — retrieval points, `read_file` fetches.

It chose `search_notes` over `search_files` unprompted. That's lesson 2's point at work: two genuinely similar tools, disambiguated entirely by their descriptions.

### Make changes

1. **Ask something the notes don't cover.** Try `--ask "what is the airspeed velocity of an unladen swallow"`. A good answer says the notes don't cover it. Watch whether it does.
2. **Implement rank-based fusion.** Replace min-max normalisation in `search_hybrid` with `1/(60 + rank)` per method and rerun `--measure`. This is the standard production choice and min-max is the weak link — see whether it beats 9/10 recall.
3. **Change chunk size.** Set `DEFAULT_MAX_CHARS` to 400, then 3000, rerunning `--measure --rebuild` each time. Too small loses context; too large dilutes the embedding so one vector represents several ideas and sits near none.
4. **Try the bigger model.** `--measure --model BAAI/bge-base-en-v1.5 --rebuild`. On a small probe it gained nothing; check whether that holds on the full corpus.
5. **Add your own queries.** Extend `QUERY_SET` in `queries.py` with five questions you'd genuinely ask. Then read the failures — that's where the information is.
6. **Index code as well as markdown.** Add `lessons/**/*.py` to `build_corpus`. Code chunks embed differently from prose; see whether it helps or just adds noise.

---

## Checkpoint

You should be able to answer these without looking:

- What is similarity search, mechanically?
- Why does normalising vectors let you use a dot product?
- Why can't keyword search answer "how do we stop dangerous code running?"
- Why is recall@k the more relevant metric for an agent than top-1?
- Why is an absolute similarity threshold hard to tune?
- What's the difference between HTTP 429 and 413, and which one does waiting fix?
- Why does `search_notes` return excerpts rather than whole documents?

Then read [NOTES.md](./NOTES.md) — it has three failure modes that only appear once retrieval meets an agent, and two measurement bugs I made while writing this.

## Files

| File | What it is |
|---|---|
| `chunking.py` | Heading-aware markdown splitting, and the finding that didn't replicate. |
| `store.py` | **The lesson.** Embeddings, caching, and three search methods to compare. |
| `queries.py` | 10 labelled queries. An eval dataset in embryo — lesson 7 grows it. |
| `agent.py` | The deliverable: six experiment modes and the `search_notes` tool. |
| `NOTES.md` | Revision notes. |

This lesson uses lesson 4's `ContextManager` unchanged, and not decoratively: without it the agent exceeded Groq's 8,000 tokens-per-minute ceiling and got HTTP 413.

**Next:** lesson 06, testing. `queries.py` and lesson 3's `Trajectory` are already most of what a test suite needs.
