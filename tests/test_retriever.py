import numpy as np
import pytest
from llm_join.retriever import EmbeddingRetriever

CORPUS = ["Apple Inc", "Microsoft Corporation", "Google LLC", "Amazon", "Meta Platforms"]
QUERIES = ["apple", "google", "amazon web services"]

def test_returns_top_k_candidates_per_query():
    retriever = EmbeddingRetriever(embed_model="all-MiniLM-L6-v2")
    results = retriever.retrieve(QUERIES, CORPUS, top_k=3)
    assert len(results) == len(QUERIES)
    for candidates in results:
        assert len(candidates) == 3

def test_candidates_are_strings_from_corpus():
    retriever = EmbeddingRetriever(embed_model="all-MiniLM-L6-v2")
    results = retriever.retrieve(QUERIES, CORPUS, top_k=2)
    for candidates in results:
        for c in candidates:
            assert c in CORPUS

def test_apple_retrieves_apple_inc():
    retriever = EmbeddingRetriever(embed_model="all-MiniLM-L6-v2")
    results = retriever.retrieve(["apple"], CORPUS, top_k=3)
    assert "Apple Inc" in results[0]

def test_custom_embed_fn():
    # Custom embed fn: random vectors (smoke test only — not checking quality)
    def my_embed(texts):
        return np.random.rand(len(texts), 32).astype("float32")

    retriever = EmbeddingRetriever(embed_fn=my_embed)
    results = retriever.retrieve(["apple"], CORPUS, top_k=2)
    assert len(results) == 1
    assert len(results[0]) == 2

def test_top_k_capped_at_corpus_size():
    retriever = EmbeddingRetriever(embed_model="all-MiniLM-L6-v2")
    results = retriever.retrieve(["apple"], ["Apple Inc", "Microsoft"], top_k=10)
    assert len(results[0]) == 2  # capped at corpus size
