"""
Tests for llm_concurrency — parallel LLM calls.
Covers: threaded (sync LLM), async (async LLM), fallback, order preservation.
"""
import asyncio
import json
import re
import time
import pandas as pd
import numpy as np
import pytest
from llm_join import fuzzy_join


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

LEFT = pd.DataFrame({"drug": ["aspirin", "ibuprofen", "paracetamol", "metformin"]})
RIGHT = pd.DataFrame({"brand": ["Bayer Aspirin", "Advil", "Tylenol", "Glucophage"]})
CTX = "pharmaceutical drug names — match generics to brand names"


def mock_embed(texts):
    """Deterministic embeddings based on text content.
    Different texts always get different vectors → cosine never reaches 1.0 between
    distinct texts, so embed_skip_threshold=1.0 (default) doesn't fire in tests.
    """
    dim = 32
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        seed = abs(hash(text)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        vecs[i] = rng.randn(dim).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)


def single_match_llm(prompt: str) -> str:
    """Returns exactly one match: index 0 with score 0.9."""
    return json.dumps([{"index": 0, "score": 0.9, "reasoning": "mock match"}])


def all_match_llm(prompt: str) -> str:
    """Returns ALL candidates above threshold (one per numbered pair in prompt)."""
    count = len(re.findall(r"\d+\. LEFT:", prompt))
    return json.dumps([
        {"index": i, "score": 0.9, "reasoning": "mock match"}
        for i in range(count)
    ])


def _left_val_from_prompt(prompt: str) -> str:
    """Extract the LEFT value from a prompt (e.g. LEFT: \"aspirin\" → 'aspirin')."""
    m = re.search(r'LEFT: "([^"]+)"', prompt)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Sync LLM — threaded path
# ---------------------------------------------------------------------------

def test_threaded_returns_dataframe():
    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=single_match_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=3, embed_concurrency=3,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(LEFT)


def test_threaded_same_result_as_sequential():
    """Parallel and sequential must produce same rows (order-insensitive)."""
    common = dict(
        left_on="drug", right_on="brand",
        llm_fn=single_match_llm, embed_fn=mock_embed,
        context=CTX, top_k=4, how="inner",
    )
    seq = fuzzy_join(LEFT, RIGHT, **common, llm_concurrency=1, embed_concurrency=1)
    par = fuzzy_join(LEFT, RIGHT, **common, llm_concurrency=4, embed_concurrency=4)

    assert sorted(seq["drug"].tolist()) == sorted(par["drug"].tolist())


def test_threaded_row_order_preserved():
    """Drug groups must appear in original LEFT order in output."""
    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=single_match_llm, embed_fn=mock_embed,
        context=CTX, top_k=4, how="inner",
        llm_concurrency=4, embed_concurrency=4,
    )
    # Each drug appears exactly once — order must match LEFT
    assert result["drug"].tolist() == LEFT["drug"].tolist()


def test_threaded_all_llm_calls_made():
    """Verify all rows go through LLM when concurrency > 1."""
    call_count = []

    def counting_llm(prompt):
        call_count.append(1)
        return single_match_llm(prompt)

    fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=counting_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=2, embed_concurrency=2,
    )
    assert len(call_count) == len(LEFT)


def test_threaded_embed_fallback_on_llm_failure():
    """When LLM raises for all rows, embed rank-0 fallback used for every row."""
    def failing_llm(prompt):
        raise RuntimeError("API down")

    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=failing_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=3, embed_concurrency=3,
        return_reasoning=True,
        max_retries=0,
    )
    assert len(result) == len(LEFT)
    assert (result["_match_method"] == "embed_fallback").all()


def test_threaded_partial_failure_fallback():
    """Only aspirin row falls back; other rows keep LLM match."""

    def partial_llm(prompt):
        left_val = _left_val_from_prompt(prompt)
        if left_val == "aspirin":
            raise RuntimeError("rate limit")
        return single_match_llm(prompt)

    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=partial_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=2, embed_concurrency=2,
        return_reasoning=True,
        max_retries=0,
    )
    assert len(result) == len(LEFT)
    aspirin_rows = result[result["drug"] == "aspirin"]
    other_rows = result[result["drug"] != "aspirin"]
    assert (aspirin_rows["_match_method"] == "embed_fallback").all()
    assert (other_rows["_match_method"] == "llm").all()


def test_threaded_is_faster_than_sequential():
    """Parallel should be meaningfully faster than sequential for slow LLMs."""
    def slow_llm(prompt):
        time.sleep(0.15)
        return single_match_llm(prompt)

    t0 = time.time()
    fuzzy_join(LEFT, RIGHT, left_on="drug", right_on="brand",
               llm_fn=slow_llm, embed_fn=mock_embed, context=CTX, top_k=4,
               llm_concurrency=1, embed_concurrency=1)
    seq_time = time.time() - t0

    t0 = time.time()
    fuzzy_join(LEFT, RIGHT, left_on="drug", right_on="brand",
               llm_fn=slow_llm, embed_fn=mock_embed, context=CTX, top_k=4,
               llm_concurrency=4, embed_concurrency=4)
    par_time = time.time() - t0

    assert par_time < seq_time * 0.7, (
        f"Parallel ({par_time:.2f}s) not faster than sequential ({seq_time:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Async LLM — asyncio path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_llm_returns_dataframe():
    async def async_llm(prompt):
        await asyncio.sleep(0.01)
        return single_match_llm(prompt)

    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=async_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=3, embed_concurrency=3,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(LEFT)


@pytest.mark.asyncio
async def test_async_llm_embed_fallback_on_failure():
    async def failing_async_llm(prompt):
        raise RuntimeError("async API down")

    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=failing_async_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=2, embed_concurrency=2,
        return_reasoning=True,
        max_retries=0,
    )
    assert len(result) == len(LEFT)
    assert (result["_match_method"] == "embed_fallback").all()


@pytest.mark.asyncio
async def test_async_llm_row_order_preserved():
    async def async_llm(prompt):
        await asyncio.sleep(0.01)
        return single_match_llm(prompt)

    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=async_llm, embed_fn=mock_embed,
        context=CTX, top_k=4, how="inner",
        llm_concurrency=4, embed_concurrency=4,
    )
    assert result["drug"].tolist() == LEFT["drug"].tolist()


@pytest.mark.asyncio
async def test_async_llm_partial_failure_fallback():
    """Only aspirin row falls back; others keep LLM match — async path."""
    async def partial_async_llm(prompt):
        left_val = _left_val_from_prompt(prompt)
        if left_val == "aspirin":
            raise RuntimeError("rate limit")
        return single_match_llm(prompt)

    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=partial_async_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=2, embed_concurrency=2,
        return_reasoning=True,
        max_retries=0,
    )
    assert len(result) == len(LEFT)
    assert (result[result["drug"] == "aspirin"]["_match_method"] == "embed_fallback").all()
    assert (result[result["drug"] != "aspirin"]["_match_method"] == "llm").all()


# ---------------------------------------------------------------------------
# Concurrency=1 regression guard
# ---------------------------------------------------------------------------

def test_concurrency_1_unchanged_behaviour():
    result = fuzzy_join(
        LEFT, RIGHT,
        left_on="drug", right_on="brand",
        llm_fn=single_match_llm, embed_fn=mock_embed,
        context=CTX, top_k=4,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(LEFT)
