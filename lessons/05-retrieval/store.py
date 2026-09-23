"""Embeddings and a vector store, built from parts rather than imported.

There is no vector database here. For a few hundred chunks, a numpy array and one
matrix multiply beat any database, and writing it out makes the mechanism
completely visible: **similarity search is a dot product followed by a sort.**

Three search methods are implemented so they can be compared instead of assumed:

  * `search_keyword`  -- literal word overlap, the lesson-3 approach
  * `search_semantic` -- cosine similarity between embeddings
  * `search_hybrid`   -- a blend of both

Measured on this repo's own notes, 193 chunks against 10 labelled queries:

    method     top-1    recall@4
    keyword     0/10      7/10
    semantic    7/10      8/10
    hybrid      7/10      9/10

Keyword search never ranks the right chunk first but usually has it somewhere in
the top four. Semantic ranks well. Hybrid ties on top-1 and has the best recall.

Be suspicious of those numbers anyway, including mine: 10 queries cannot resolve a
one-query difference, so "hybrid beats semantic by 1" is not a finding. The reason
all three are implemented is so you can re-measure on your own corpus rather than
trust anybody's claim. Lesson 7 is about doing that properly.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chunking import Chunk, Corpus

#: bge-small rather than bge-base. On a small probe the two scored identically,
#: so the small one wins on size (~30 MB vs ~130 MB) and CPU speed. That probe was
#: 8 chunks though, and the heading experiment shows how badly small samples
#: mislead -- so treat this as "no evidence the bigger model helps here" rather
#: than proof it does not. Re-run --measure with
#: --model BAAI/bge-base-en-v1.5 to check on your own corpus.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

#: Words too common to carry meaning. Only used by keyword search, where they
#: would otherwise match everything.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "get", "give", "gives", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "out", "should",
    "so", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "use", "using", "was", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


@dataclass
class Hit:
    """One retrieved chunk, with the score that put it there."""

    chunk: Chunk
    score: float
    method: str
    #: Present for hybrid, so you can see which component did the work.
    semantic_score: float | None = None
    keyword_score: float | None = None

    def cite(self) -> str:
        return f"{self.chunk.label} (line {self.chunk.start_line})"


def tokenize(text: str) -> set[str]:
    """Words worth matching on: alphanumeric, 3+ characters, not a stopword."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


class NoteStore:
    """A searchable corpus. Embeddings are computed once and cached to disk."""

    def __init__(self, corpus: Corpus, model_name: str = DEFAULT_MODEL) -> None:
        self.corpus = corpus
        self.model_name = model_name
        self._vectors: np.ndarray | None = None
        self._model = None
        # Precomputed for keyword search so it is not re-tokenised per query.
        self._word_sets = [tokenize(c.embedding_text) for c in corpus.chunks]
        # Inverse document frequency: a word appearing in every chunk tells you
        # nothing, so rare words should score higher. This is the core idea of
        # TF-IDF, and keyword search is noticeably worse without it.
        total = max(1, len(corpus.chunks))
        counts: dict[str, int] = {}
        for words in self._word_sets:
            for word in words:
                counts[word] = counts.get(word, 0) + 1
        self._idf = {
            word: math.log(total / count) for word, count in counts.items()
        }

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is None:
            # Imported lazily: it pulls in onnxruntime and downloads a model on
            # first use, and lessons 0-4 must not pay that cost.
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed and L2-normalise.

        Normalising to unit length is what lets cosine similarity be a plain dot
        product later: cos(a,b) = a.b / (|a||b|), so if |a| = |b| = 1 then
        cos(a,b) = a.b. One normalisation up front turns every later comparison
        into a single multiply-add.
        """
        model = self._load_model()
        vectors = np.array(list(model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def build(self, cache_path: Path | None = None, verbose: bool = False) -> None:
        """Compute embeddings for the whole corpus, using a cache if valid."""
        if cache_path and self._load_cache(cache_path, verbose):
            return

        texts = [c.embedding_text for c in self.corpus.chunks]
        if verbose:
            print(f"  embedding {len(texts)} chunks with {self.model_name}...")
        self._vectors = self.embed_texts(texts)

        if cache_path:
            self._save_cache(cache_path)
            if verbose:
                print(f"  cached to {cache_path}")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _fingerprint(self) -> str:
        """Identity of this corpus + model, so a stale cache is never used.

        Embeddings silently become wrong when the source text or the model
        changes. A cache keyed only by filename would happily return vectors for
        text that no longer exists, and the failure looks like "retrieval got
        worse for no reason" rather than "the cache is stale".
        """
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.model_name.encode())
        for chunk in self.corpus.chunks:
            digest.update(chunk.embedding_text.encode("utf-8", errors="replace"))
        return digest.hexdigest()[:16]

    def _save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self._vectors)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "fingerprint": self._fingerprint(),
                    "model": self.model_name,
                    "chunks": len(self.corpus.chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_cache(self, path: Path, verbose: bool = False) -> bool:
        meta_path = path.with_suffix(".json")
        vec_path = path.with_suffix(".npy")
        if not meta_path.exists() or not vec_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if meta.get("fingerprint") != self._fingerprint():
            if verbose:
                print("  cache is stale (notes or model changed); re-embedding")
            return False
        vectors = np.load(vec_path)
        if vectors.shape[0] != len(self.corpus.chunks):
            return False
        self._vectors = vectors
        if verbose:
            print(f"  loaded {vectors.shape[0]} cached embeddings ({vectors.shape[1]}d)")
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_semantic(self, query: str, top_k: int = 4) -> list[Hit]:
        """Rank by cosine similarity. The entire mechanism is two lines."""
        if self._vectors is None:
            raise RuntimeError("Call build() before searching.")

        query_vector = self.embed_texts([query])[0]
        scores = self._vectors @ query_vector          # <- similarity search
        order = np.argsort(-scores)[:top_k]            # <- ranking

        return [
            Hit(self.corpus.chunks[i], float(scores[i]), "semantic")
            for i in order
        ]

    def search_keyword(self, query: str, top_k: int = 4) -> list[Hit]:
        """Rank by IDF-weighted word overlap. What lesson 3's search_files did.

        Its weakness is structural, not a matter of tuning: it can only match
        words that are literally present. Ask "how do we stop dangerous code
        running?" of a chunk that says "eval", "AST" and "allowlist" and there is
        no overlap to find.
        """
        query_words = tokenize(query)
        scores = np.zeros(len(self.corpus.chunks), dtype=np.float32)
        for position, words in enumerate(self._word_sets):
            shared = query_words & words
            if shared:
                scores[position] = sum(self._idf.get(w, 0.0) for w in shared)

        order = np.argsort(-scores)[:top_k]
        return [
            Hit(self.corpus.chunks[i], float(scores[i]), "keyword")
            for i in order
            if scores[i] > 0
        ]

    def search_hybrid(
        self, query: str, top_k: int = 4, semantic_weight: float = 0.7
    ) -> list[Hit]:
        """Blend both rankings.

        The thing to understand is that the two scores live on incompatible
        scales. Cosine similarity sits in a narrow band (typically 0.6-0.8 even
        for poor matches), while summed IDF is unbounded and depends on how many
        query words happen to appear. Adding them directly would let keyword
        scores dominate entirely, so each is min-max normalised first.

        Min-max is the weak link: it is sensitive to outliers, so one chunk with a
        strong keyword match stretches the scale and can drag an unrelated result
        above a correct semantic hit. Rank-based fusion -- scoring by
        1/(k + rank) instead of by normalised value -- avoids that and is the
        standard production choice. It is left as a README exercise so you can
        measure the difference yourself.

        Measured here: hybrid ties semantic on top-1 (7/10) and edges it on
        recall@4 (9/10 vs 8/10). One query is not a real margin on a 10-query set.
        """
        if self._vectors is None:
            raise RuntimeError("Call build() before searching.")

        query_vector = self.embed_texts([query])[0]
        semantic = self._vectors @ query_vector

        query_words = tokenize(query)
        keyword = np.zeros(len(self.corpus.chunks), dtype=np.float32)
        for position, words in enumerate(self._word_sets):
            shared = query_words & words
            if shared:
                keyword[position] = sum(self._idf.get(w, 0.0) for w in shared)

        def normalise(values: np.ndarray) -> np.ndarray:
            spread = values.max() - values.min()
            if spread < 1e-9:
                return np.zeros_like(values)
            return (values - values.min()) / spread

        blended = (
            semantic_weight * normalise(semantic)
            + (1 - semantic_weight) * normalise(keyword)
        )
        order = np.argsort(-blended)[:top_k]

        return [
            Hit(
                self.corpus.chunks[i],
                float(blended[i]),
                "hybrid",
                semantic_score=float(semantic[i]),
                keyword_score=float(keyword[i]),
            )
            for i in order
        ]

    def search(self, query: str, top_k: int = 4, method: str = "semantic") -> list[Hit]:
        if method == "keyword":
            return self.search_keyword(query, top_k)
        if method == "hybrid":
            return self.search_hybrid(query, top_k)
        return self.search_semantic(query, top_k)

    @property
    def dimensions(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors is not None else 0
