# Lesson 5 — Notes

## The one idea

**Similarity search is a dot product followed by a sort.** Embed your documents once, embed the query, multiply, take the top k. There is no vector database here and none is needed for a few hundred chunks.

```python
query_vector = embed(query)          # one vector
scores = document_vectors @ query_vector   # similarity search
order = np.argsort(-scores)[:top_k]        # ranking
```

Normalising every vector to unit length up front is what lets cosine similarity be a plain dot product: `cos(a,b) = a·b / (|a||b|)`, so if both are length 1 it reduces to `a·b`.

## Why retrieval, after lesson 4

Lesson 4 salvaged a context window that was already full. Retrieval avoids filling it.

Measured: the corpus is 132,590 characters (~33,000 tokens). Retrieving the top 3 chunks for a question sent 2,942 characters — **2.2% of the corpus**, 883 input tokens.

The trade is real, not free. Stuffing cannot miss; the answer is definitely in there. Retrieval can miss, and then the model answers confidently from the wrong three chunks. That is why measuring accuracy matters and why the system prompt must permit "the notes do not cover this".

## Practical setup

**Groq serves no embedding models** — verified against the live list: chat, safety classifiers and speech only. Embeddings run locally whatever you choose.

**fastembed, not sentence-transformers.** fastembed runs the model through ONNX, so no PyTorch: a ~30 MB quantised download instead of a multi-hundred-MB torch wheel. Measured 0.02s to embed three texts on CPU with no GPU. 193 chunks embed in a couple of seconds.

This forced `requires-python` from 3.11 to 3.12, because fastembed needs `numpy>=2.3` which needs 3.12. Worth noting 3.11 was only ever *claimed* — every lesson was written and tested on 3.12.

**Cache embeddings, and key the cache on a fingerprint of the content plus model name.** A cache keyed by filename alone happily returns vectors for text that no longer exists, and the symptom is "retrieval got worse for no reason" rather than "the cache is stale".

## Measured results

193 chunks from 14 files, 384 dimensions, 10 labelled queries phrased to avoid the target note's vocabulary:

| method | top-1 | recall@4 |
|---|---|---|
| keyword | 0/10 | 7/10 |
| semantic | 7/10 | 8/10 |
| hybrid | 7/10 | 9/10 |

**Keyword search never ranked the right chunk first** but usually had it in the top four. Semantic ranks well. Hybrid ties on top-1 and edges recall.

**recall@k is the metric that matters for an agent**, because it reads all k results — a correct chunk ranked third is still useful. top-1 matters when you show one answer to a user. Reporting only one of these hides half the picture.

**Treat a one-query margin as noise.** Ten queries cannot resolve "hybrid beats semantic by 1". That limitation is lesson 7's problem.

## The measurement bugs, which are the real lesson

**Bug 1: the labels were wrong, not the retrieval.** The first version matched the source *path*, so a query expecting `02-tool-calling` scored a MISS when retrieval returned:

```
docs/00-index.md > Key learnings index > Lesson 02 - Tool calling
```

That is the condensed summary of exactly the right material. Semantic search was right and the label said otherwise. Fixed by matching the chunk's full label (path **and** heading path).

**Your ground truth encodes assumptions. An apparent miss is sometimes the system finding a better answer than the one you wrote down.** Read your failures before trusting a score.

**Bug 2: the same mistake, made twice.** `--chunking` had the identical path-only bug after `--measure` was fixed, and produced 3/10 where `--measure` produced 7/10 for the same configuration. It was caught *only* because two experiments measuring the same thing disagreed. Two measurements that contradict each other are a gift; one measurement quietly wrong is not.

**An eval that is subtly wrong is more dangerous than no eval, because it looks authoritative.**

## The finding that did not replicate

Theory: embed the heading alongside the chunk body, because a chunk lifted from mid-document has lost its context.

On a hand-written probe of 8 chunks it worked well — top-1 from 4/7 to 6/7, beating what doubling the model's dimensions achieved.

On the real 193-chunk corpus it **did not replicate**: 8/10 body-only against 7/10 with the heading, recall@4 identical at 8/10.

Why the small sample misled: it had no competing documents, and every heading was informative. At scale many headings are generic ("The one idea", "Carry forward") and add noise.

**A result from a tiny hand-picked sample is a hypothesis, not a finding.**

The heading is still embedded — for citation quality, not ranking. Being explicit about why a choice survives measurement matters more than the choice.

## Similarity scores are nearly meaningless in absolute terms

Cosine similarity sits in a narrow band, typically 0.6–0.8 even for a bad match. Only the **ranking** and the **gap between first and second** carry information.

Practical consequence: a threshold like "only use hits above 0.75" looks principled and is almost impossible to tune, because the band shifts with model and corpus. Prefer top-k plus a gap check.

## Why hybrid needs care

The two scores live on incompatible scales — cosine in a narrow band, summed IDF unbounded — so they must be normalised before blending. Min-max is the weak link: it is outlier-sensitive, so one strong keyword match stretches the scale and can drag an unrelated chunk above a correct semantic hit.

Rank-based fusion (`1/(k + rank)` instead of normalised value) avoids this and is the standard production choice. **"Combine both signals" is not automatically better — combining badly is worse than not combining.**

## A summary document is an attractor

`docs/00-index.md` condenses every lesson's key learnings, so it matches almost any query about the project reasonably well and crowds out the detailed source note. Real-world analogue: an FAQ page outranking the actual documentation.

Not necessarily wrong — the condensed answer is often what you want — but know it is happening, and expect it to skew an eval whose labels point at detailed sources.

## Three failures that only appear once retrieval meets an agent

**1. HTTP 413, request too large.** Not the same as 429. Retrieval returns large results, seven tool schemas are re-sent every call, and history accumulates: the request hit 10,020 tokens against Groq's 8,000 tokens-per-minute ceiling. **429 means wait; 413 means this request will never fit.** Waiting does not help — reduce what you send.

The fix was to use lesson 4's `ContextManager` unchanged. Lessons compose.

**2. Over-compression destroys retrieval.** Lesson 4's 600-character default crushed a 6,195-token search result to 765 tokens, deleting the excerpts the search had just found. The agent could not answer and re-searched five times until the step cap.

Two corrections: bound the result **at source** (return 450-character excerpts, not whole chunks) and raise `tool_result_max_chars` to 1,800 for this agent. **Compression thresholds are not universal — they depend on how much of a tool's output is signal.** Retrieval output is dense; a file listing is not.

**3. An agent with a search tool will search forever.** When the corpus genuinely lacks an answer, the model rephrases the query and searches again, indefinitely. Stall detection does not catch it because each query differs slightly.

Fixed with an explicit instruction: "use at most two searches, then answer with what you have; if the notes do not cover it, say so and stop." **Telling the agent when to give up is your job, not the model's.**

## The working shape

```
search_notes -> read_file -> read_file -> answer
```

Retrieval finds *where* the answer lives; `read_file` gets the full text. That is why `search_notes` returns truncated excerpts with citations rather than whole documents, and why it sits alongside `read_file` instead of replacing it.

The agent chose `search_notes` over `search_files` unprompted, which is lesson 2's point at work: two genuinely similar tools disambiguated entirely by their descriptions.

## Keyword search is not obsolete

It scored 0/10 on top-1 here, but that is because every query was deliberately paraphrased. Ask for a literal string — a function name, an error message, an identifier — and keyword search is exact where semantic search is approximate. Keep both. The tool descriptions tell the model which to reach for.

## Carry forward

- `queries.py` is an eval dataset in embryo. Lesson 7 grows it and adds proper metrics.
- The 413/TPM ceiling constrains lessons 7–9 further: eval suites need small per-request footprints, not just a daily token budget.
- Chunk boundaries remain unsolved: an answer straddling two chunks is retrievable by neither half. Overlap mitigates, never fixes.
