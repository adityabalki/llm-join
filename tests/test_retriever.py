"""Tests for EmbeddingRetriever: retrieval correctness, batching, sync/async concurrency."""
import asyncio
import hashlib
import threading
import time

import numpy as np
import pytest

from llm_join.retriever import EmbeddingRetriever


CORPUS = ["Apple Inc", "Microsoft Corporation", "Google LLC", "Amazon", "Meta Platforms"]


# ---------------------------------------------------------------------------
# Embed helpers
# ---------------------------------------------------------------------------

def make_embed():
    """Deterministic sync embed: each unique text gets a stable random-ish vector."""
    cache = {}

    def embed(texts):
        vecs = []
        for t in texts:
            if t not in cache:
                seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % (2**31)
                rng = np.random.RandomState(seed)
                cache[t] = rng.rand(32).astype("float32")
            vecs.append(cache[t])
        return np.array(vecs)

    return embed


def _make_logged_sync_embed(call_log=None, sleep_per_batch=0.0, dim=8):
    """Sync embed_fn that records each call's batch size, thread id, timestamp."""
    def embed_fn(texts):
        if call_log is not None:
            call_log.append({
                "batch_size": len(texts),
                "thread": threading.get_ident(),
                "ts": time.time(),
            })
        if sleep_per_batch:
            time.sleep(sleep_per_batch)
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**31))
        return rng.randn(len(texts), dim).astype("float32")
    return embed_fn


def _make_logged_async_embed(call_log=None, sleep_per_batch=0.0, dim=8):
    """Async embed_fn that records each call's batch size, timestamp."""
    async def embed_fn(texts):
        if call_log is not None:
            call_log.append({
                "batch_size": len(texts),
                "ts": time.time(),
            })
        if sleep_per_batch:
            await asyncio.sleep(sleep_per_batch)
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**31))
        return rng.randn(len(texts), dim).astype("float32")
    return embed_fn


# ---------------------------------------------------------------------------
# Retrieval correctness (kept from original test_retriever.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_batch_size_actually_batches_texts(self):
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_sync_embed(log),
            batch_size=4,
            embed_concurrency=1,
        )
        # 5 corpus + 3 query = 8 → 2 batches of 4
        retriever.retrieve_with_scores(
            query_vals=["q1", "q2", "q3"],
            corpus_vals=["c1", "c2", "c3", "c4", "c5"],
            top_k=2,
        )
        sizes = [c["batch_size"] for c in log]
        assert sizes == [4, 4]

    def test_batch_size_uneven_last_batch(self):
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_sync_embed(log),
            batch_size=3,
            embed_concurrency=1,
        )
        # 4 corpus + 3 query = 7 → batches of 3,3,1
        retriever.retrieve_with_scores(
            query_vals=["q1", "q2", "q3"],
            corpus_vals=["c1", "c2", "c3", "c4"],
            top_k=2,
        )
        sizes = sorted([c["batch_size"] for c in log])
        assert sizes == [1, 3, 3]

    def test_batch_size_larger_than_total_one_call(self):
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_sync_embed(log),
            batch_size=1000,
            embed_concurrency=1,
        )
        retriever.retrieve_with_scores(
            query_vals=["q1"],
            corpus_vals=["c1", "c2"],
            top_k=1,
        )
        assert len(log) == 1
        assert log[0]["batch_size"] == 3

    def test_batch_size_one_per_text(self):
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_sync_embed(log),
            batch_size=1,
            embed_concurrency=1,
        )
        retriever.retrieve_with_scores(
            query_vals=["q1", "q2"],
            corpus_vals=["c1", "c2", "c3"],
            top_k=2,
        )
        assert len(log) == 5
        assert all(c["batch_size"] == 1 for c in log)


# ---------------------------------------------------------------------------
# Sync concurrency (ThreadPoolExecutor)
# ---------------------------------------------------------------------------

class TestSyncConcurrency:
    def test_sync_concurrency_uses_multiple_threads(self):
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_sync_embed(log, sleep_per_batch=0.1),
            batch_size=2,
            embed_concurrency=4,
        )
        retriever.retrieve_with_scores(
            query_vals=[f"q{i}" for i in range(4)],
            corpus_vals=[f"c{i}" for i in range(8)],
            top_k=2,
        )
        unique_threads = {c["thread"] for c in log}
        assert len(unique_threads) > 1, "Expected multiple threads"

    def test_sync_concurrency_one_runs_sequential(self):
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_sync_embed(log, sleep_per_batch=0.05),
            batch_size=2,
            embed_concurrency=1,
        )
        retriever.retrieve_with_scores(
            query_vals=["q1", "q2"],
            corpus_vals=["c1", "c2", "c3", "c4"],
            top_k=2,
        )
        unique_threads = {c["thread"] for c in log}
        assert len(unique_threads) == 1

    def test_sync_concurrency_speeds_up_runtime(self):
        # 6 batches × 0.1s sequential = 0.6s; concurrency=4 → ~0.2s
        retriever_par = EmbeddingRetriever(
            embed_fn=_make_logged_sync_embed(sleep_per_batch=0.1),
            batch_size=2,
            embed_concurrency=4,
        )
        t0 = time.time()
        retriever_par.retrieve_with_scores(
            query_vals=[f"q{i}" for i in range(4)],
            corpus_vals=[f"c{i}" for i in range(8)],
            top_k=2,
        )
        elapsed = time.time() - t0
        assert elapsed < 0.45, f"Parallel run took {elapsed:.2f}s, concurrency not working"


# ---------------------------------------------------------------------------
# Async detection + concurrency (asyncio.Semaphore)
# ---------------------------------------------------------------------------

class TestAsyncEmbed:
    def test_async_embed_detected_and_called(self):
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_async_embed(log),
            batch_size=4,
            embed_concurrency=2,
        )
        assert retriever._is_async is True

        results = retriever.retrieve_with_scores(
            query_vals=["q1", "q2"],
            corpus_vals=["c1", "c2", "c3", "c4", "c5"],
            top_k=2,
        )
        assert len(results) == 2
        # 7 texts → batches of 4, 3
        sizes = sorted([c["batch_size"] for c in log])
        assert sizes == [3, 4]

    def test_async_concurrency_limited_by_semaphore(self):
        # 4 batches × 0.1s with semaphore=2 → ~0.2s (2 rounds of 2)
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_async_embed(sleep_per_batch=0.1),
            batch_size=2,
            embed_concurrency=2,
        )
        t0 = time.time()
        retriever.retrieve_with_scores(
            query_vals=[f"q{i}" for i in range(4)],
            corpus_vals=[f"c{i}" for i in range(4)],
            top_k=2,
        )
        elapsed = time.time() - t0
        assert 0.15 < elapsed < 0.4, f"Expected ~0.2s with concurrency=2, got {elapsed:.2f}s"

    def test_async_high_concurrency_runs_in_one_round(self):
        # 4 batches with concurrency=10 → all fire at once → ~0.1s
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_async_embed(sleep_per_batch=0.1),
            batch_size=2,
            embed_concurrency=10,
        )
        t0 = time.time()
        retriever.retrieve_with_scores(
            query_vals=[f"q{i}" for i in range(4)],
            corpus_vals=[f"c{i}" for i in range(4)],
            top_k=2,
        )
        elapsed = time.time() - t0
        assert elapsed < 0.25, f"High concurrency should be ~0.1s, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_async_embed_works_inside_running_loop(self):
        """Jupyter/Databricks: event loop already running. Library must dispatch to side thread."""
        log = []
        retriever = EmbeddingRetriever(
            embed_fn=_make_logged_async_embed(log),
            batch_size=3,
            embed_concurrency=2,
        )
        results = retriever.retrieve_with_scores(
            query_vals=["q1", "q2"],
            corpus_vals=["c1", "c2", "c3", "c4"],
            top_k=2,
        )
        assert len(results) == 2
        assert len(log) >= 1


# ---------------------------------------------------------------------------
# Output correctness — concurrency must NOT break ordering
# ---------------------------------------------------------------------------

class TestOutputCorrectness:
    def test_sync_concurrent_matches_sequential_ranking(self):
        # Same RNG seed → same embeddings → same FAISS ranking
        corpus = [f"corpus_{i}" for i in range(20)]
        queries = [f"query_{i}" for i in range(5)]

        r_seq = EmbeddingRetriever(_make_logged_sync_embed(), batch_size=4, embed_concurrency=1)
        r_par = EmbeddingRetriever(_make_logged_sync_embed(), batch_size=4, embed_concurrency=8)

        out_seq = r_seq.retrieve_with_scores(queries, corpus, top_k=3)
        out_par = r_par.retrieve_with_scores(queries, corpus, top_k=3)

        for s, p in zip(out_seq, out_par):
            assert [c for c, _ in s] == [c for c, _ in p]

    def test_async_concurrent_matches_sync(self):
        corpus = [f"corpus_{i}" for i in range(15)]
        queries = [f"query_{i}" for i in range(4)]

        r_sync = EmbeddingRetriever(_make_logged_sync_embed(), batch_size=4, embed_concurrency=2)
        r_async = EmbeddingRetriever(_make_logged_async_embed(), batch_size=4, embed_concurrency=2)

        out_sync = r_sync.retrieve_with_scores(queries, corpus, top_k=3)
        out_async = r_async.retrieve_with_scores(queries, corpus, top_k=3)

        for s, a in zip(out_sync, out_async):
            assert [c for c, _ in s] == [c for c, _ in a]

    def test_vector_order_preserved_across_concurrent_batches(self):
        """If batch results re-assembled out of order, FAISS index gets garbage."""
        def positional_embed(texts):
            vecs = np.zeros((len(texts), 4), dtype="float32")
            for i, t in enumerate(texts):
                vecs[i, 0] = abs(hash(t)) % 1000
            return vecs

        retriever = EmbeddingRetriever(positional_embed, batch_size=2, embed_concurrency=4)
        corpus = [f"c_{i}" for i in range(10)]
        queries = [f"q_{i}" for i in range(3)]

        result = retriever.retrieve_with_scores(queries, corpus, top_k=2)
        assert len(result) == 3
        for r in result:
            assert all(c.startswith("c_") for c, _ in r)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_corpus_returns_empty_lists_for_each_query(self):
        retriever = EmbeddingRetriever(_make_logged_sync_embed(), batch_size=4, embed_concurrency=2)
        result = retriever.retrieve_with_scores(["q1", "q2"], [], top_k=3)
        assert result == [[], []]

    def test_empty_query_returns_empty_list(self):
        retriever = EmbeddingRetriever(_make_logged_sync_embed(), batch_size=4, embed_concurrency=2)
        result = retriever.retrieve_with_scores([], ["c1", "c2"], top_k=2)
        assert result == []

    def test_top_k_larger_than_corpus_caps(self):
        retriever = EmbeddingRetriever(_make_logged_sync_embed(), batch_size=4, embed_concurrency=2)
        result = retriever.retrieve_with_scores(["q1"], ["c1", "c2"], top_k=10)
        assert len(result[0]) == 2

    def test_single_corpus_single_query(self):
        retriever = EmbeddingRetriever(_make_logged_sync_embed(), batch_size=4, embed_concurrency=2)
        result = retriever.retrieve_with_scores(["q1"], ["c1"], top_k=1)
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_retriever_defaults(self):
        retriever = EmbeddingRetriever(_make_logged_sync_embed())
        assert retriever._batch_size == 32
        assert retriever._embed_concurrency == 10
        assert retriever._is_async is False


# ---------------------------------------------------------------------------
# Integration with fuzzy_join — async embed_fn end-to-end
# ---------------------------------------------------------------------------

class TestFuzzyJoinIntegration:
    def test_fuzzy_join_with_async_embed_fn(self):
        import json
        import pandas as pd
        from llm_join import fuzzy_join

        async def async_embed(texts):
            rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**31))
            return rng.randn(len(texts), 8).astype("float32")

        def sync_llm(prompt):
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "match"}])

        df1 = pd.DataFrame({"a": ["alpha", "beta", "gamma"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2", "gamma_v2"]})

        result = fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=sync_llm,
            embed_fn=async_embed,
            context="test",
            llm_concurrency=2, embed_concurrency=2,
            batch_size=2,
        )
        assert isinstance(result, pd.DataFrame)

    def test_fuzzy_join_with_sync_embed_concurrent(self):
        import json
        import pandas as pd
        from llm_join import fuzzy_join

        call_count = [0]
        def sync_embed(texts):
            call_count[0] += 1
            rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**31))
            return rng.randn(len(texts), 8).astype("float32")

        def sync_llm(prompt):
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "match"}])

        df1 = pd.DataFrame({"a": [f"item_{i}" for i in range(10)]})
        df2 = pd.DataFrame({"b": [f"thing_{i}" for i in range(10)]})

        result = fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=sync_llm,
            embed_fn=sync_embed,
            context="test",
            llm_concurrency=2, embed_concurrency=2,
            batch_size=4,
        )
        # 10 corpus + 10 query = 20 texts → 5 batches at batch_size=4
        assert call_count[0] >= 5
        assert isinstance(result, pd.DataFrame)
