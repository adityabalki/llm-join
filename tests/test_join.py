import json
import pandas as pd
import numpy as np
import pytest
from llm_join import fuzzy_join

# Mock LLM: always picks candidate at index 0 with score 0.95
def mock_llm(prompt: str) -> str:
    import re
    pairs = re.findall(r'(\d+)\. LEFT:', prompt)
    count = len(pairs)
    return json.dumps([
        {"index": i, "score": 0.95, "reasoning": "mock match"}
        for i in range(count)
    ])

# Mock embed_fn: deterministic vectors so retriever always returns all corpus items
def mock_embed(texts):
    # Each text gets a unique one-hot-ish vector so faiss can retrieve all
    n = len(texts)
    vecs = np.eye(max(n, 10), dtype="float32")[:n]
    return vecs

DF1 = pd.DataFrame({"drug": ["aspirin", "ibuprofen"], "dose": [100, 200]})
DF2 = pd.DataFrame({"brand": ["Bayer Aspirin", "Advil"], "price": [5.0, 8.0]})
CTX = "pharmaceutical drug names — match generic INN names to US brand names"

def test_basic_join_returns_dataframe():
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
    )
    assert isinstance(result, pd.DataFrame)

def test_inner_join_has_both_columns():
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        how="inner",
    )
    assert "dose" in result.columns
    assert "price" in result.columns

def test_return_reasoning_adds_score_column():
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        return_reasoning=True,
    )
    assert "_llm_score" in result.columns
    assert "_llm_reasoning" in result.columns

def test_threshold_low_matches_everything():
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        threshold=0.01,
        how="inner",
    )
    assert len(result) == 2

def test_threshold_one_matches_nothing():
    # mock llm returns 0.95, threshold=1.0 should filter all
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        threshold=1.0,
        how="inner",
    )
    assert len(result) == 0

def test_max_llm_calls_warns(recwarn):
    fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        max_llm_calls=1,  # only 1 allowed, 2 rows -> warn
    )
    assert any("max_llm_calls" in str(w.message) for w in recwarn)


def test_multi_col_left_on():
    df1 = pd.DataFrame({"drug": ["aspirin"], "form": ["tablet"], "dose": [100]})
    df2 = pd.DataFrame({"brand": ["Bayer Aspirin Tablet"], "price": [5.0]})
    result = fuzzy_join(
        df1, df2,
        left_on=["drug", "form"],
        right_on="brand",
        llm=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        how="inner",
    )
    assert isinstance(result, pd.DataFrame)


def test_missing_context_raises():
    with pytest.raises(TypeError):
        fuzzy_join(
            DF1, DF2,
            left_on="drug", right_on="brand",
            llm=mock_llm,
            embed_fn=mock_embed,
            # context omitted — should raise TypeError (missing required arg)
        )


def test_empty_context_raises():
    with pytest.raises(ValueError, match="context must not be empty"):
        fuzzy_join(
            DF1, DF2,
            left_on="drug", right_on="brand",
            llm=mock_llm,
            embed_fn=mock_embed,
            context="   ",  # whitespace-only
        )
