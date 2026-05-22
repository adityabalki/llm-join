import numpy as np
import pytest
from llm_join.retriever import EmbeddingRetriever

CORPUS = ["Apple Inc", "Microsoft Corporation", "Google LLC", "Amazon", "Meta Platforms"]


def make_embed():
    """Deterministic embed: each unique text gets a stable random-ish vector."""
    cache = {}
    def embed(texts):
        vecs = []
        for t in texts:
            if t not in cache:
                import hashlib
                seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % (2**31)
                rng = np.random.RandomState(seed)
                cache[t] = rng.rand(32).astype("float32")
            vecs.append(cache[t])
        return np.array(vecs)
    return embed


def test_returns_top_k_candidates_per_query():
    retriever = EmbeddingRetriever(embed_fn=make_embed())
    results = retriever.retrieve(["apple", "google"], CORPUS, top_k=3)
    assert len(results) == 2
    for candidates in results:
        assert len(candidates) == 3


def test_candidates_are_strings_from_corpus():
    retriever = EmbeddingRetriever(embed_fn=make_embed())
    results = retriever.retrieve(["apple"], CORPUS, top_k=2)
    for c in results[0]:
        assert c in CORPUS


def test_top_k_capped_at_corpus_size():
    retriever = EmbeddingRetriever(embed_fn=make_embed())
    results = retriever.retrieve(["apple"], ["Apple Inc", "Microsoft"], top_k=10)
    assert len(results[0]) == 2


def test_empty_corpus_returns_empty():
    retriever = EmbeddingRetriever(embed_fn=make_embed())
    results = retriever.retrieve(["apple"], [], top_k=5)
    assert results == [[]]
