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

# Mock embed_fn: deterministic vectors based on text content.
# Different texts always get different vectors → cosine never reaches 1.0 between
# distinct texts, so embed_skip_threshold=1.0 (default) doesn't fire in tests.
def mock_embed(texts):
    dim = 32
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        seed = abs(hash(text)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        vecs[i] = rng.randn(dim).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)

DF1 = pd.DataFrame({"drug": ["aspirin", "ibuprofen"], "dose": [100, 200]})
DF2 = pd.DataFrame({"brand": ["Bayer Aspirin", "Advil"], "price": [5.0, 8.0]})
CTX = "pharmaceutical drug names — match generic INN names to US brand names"

def test_basic_join_returns_dataframe():
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert isinstance(result, pd.DataFrame)

def test_inner_join_has_both_columns():
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    assert "dose" in result.columns
    assert "price" in result.columns

def test_return_reasoning_adds_score_column():
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        return_reasoning=True,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert "_llm_score" in result.columns
    assert "_llm_reasoning" in result.columns

def test_threshold_low_matches_everything():
    # llm_threshold=0.01 → candidates eligible, but match_all=False (default) → ONE match per left row
    # 2 left rows → 2 output rows (no tie expansion)
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        llm_threshold=0.01,
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(result) == 2

def test_threshold_one_matches_nothing():
    # mock llm returns 0.95, threshold=1.0 should filter all
    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        llm_threshold=1.0,
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(result) == 0

def test_max_llm_calls_warns(recwarn):
    fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        max_llm_calls=1,  # only 1 allowed, 2 rows -> warn
        llm_concurrency=1, embed_concurrency=1,
    )
    assert any("max_llm_calls" in str(w.message) for w in recwarn)


def test_multi_col_left_on():
    df1 = pd.DataFrame({"drug": ["aspirin"], "form": ["tablet"], "dose": [100]})
    df2 = pd.DataFrame({"brand": ["Bayer Aspirin Tablet"], "price": [5.0]})
    result = fuzzy_join(
        df1, df2,
        left_on=["drug", "form"],
        right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    assert isinstance(result, pd.DataFrame)


def test_missing_context_raises():
    with pytest.raises(TypeError):
        fuzzy_join(
            DF1, DF2,
            left_on="drug", right_on="brand",
            llm_fn=mock_llm,
            embed_fn=mock_embed,
            # context omitted — should raise TypeError (missing required arg)
            llm_concurrency=1, embed_concurrency=1,
        )


def test_empty_context_raises():
    with pytest.raises(ValueError, match="context must not be empty"):
        fuzzy_join(
            DF1, DF2,
            left_on="drug", right_on="brand",
            llm_fn=mock_llm,
            embed_fn=mock_embed,
            context="   ",  # whitespace-only
            llm_concurrency=1, embed_concurrency=1,
        )


def test_embed_fallback_on_llm_failure():
    # LLM that always raises — should trigger embed rank-0 fallback for every row
    def failing_llm(prompt: str) -> str:
        raise RuntimeError("LLM API down")

    result = fuzzy_join(
        DF1, DF2,
        left_on="drug", right_on="brand",
        llm_fn=failing_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=2,
        max_retries=0,  # no retries — fail immediately
        return_reasoning=True,
        llm_concurrency=1, embed_concurrency=1,
    )
    # Both rows should still be in result via embed fallback
    assert len(result) == 2
    assert (_result := result["_match_method"] == "embed_fallback").all(), \
        f"Expected all embed_fallback, got: {result['_match_method'].tolist()}"
    assert result["_llm_reasoning"].str.contains("embed rank-0 fallback").all()


# ---------------------------------------------------------------------------
# Multi-key join tests
# ---------------------------------------------------------------------------

def test_multi_col_left_on_result_has_correct_rows():
    """Multi-col left_on: result must have one row per matched left row (top_k=1 → single match)."""
    df1 = pd.DataFrame({"drug": ["aspirin", "ibuprofen"], "form": ["tablet", "capsule"], "dose": [100, 200]})
    df2 = pd.DataFrame({"brand": ["Bayer Aspirin Tablet", "Advil Capsule"], "price": [5.0, 8.0]})
    result = fuzzy_join(
        df1, df2,
        left_on=["drug", "form"],
        right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,   # 1 candidate → mock returns exactly 1 match per left row
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(result) == 2
    assert "dose" in result.columns
    assert "price" in result.columns


def test_multi_col_left_on_no_temp_key_leaked():
    """__left_key__ temp column must not appear in output."""
    df1 = pd.DataFrame({"drug": ["aspirin"], "form": ["tablet"]})
    df2 = pd.DataFrame({"brand": ["Bayer Aspirin Tablet"]})
    result = fuzzy_join(
        df1, df2,
        left_on=["drug", "form"],
        right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert "__left_key__" not in result.columns


def test_multi_col_right_on_result_has_correct_rows():
    """Multi-col right_on: result must have one row per matched left row (top_k=1 → single match)."""
    df1 = pd.DataFrame({"title": ["Software Engineer", "Data Analyst"], "dept": [1, 2]})
    df2 = pd.DataFrame({"role": ["Software", "Data"], "level": ["Engineer", "Analyst"], "salary": [100, 90]})
    result = fuzzy_join(
        df1, df2,
        left_on="title",
        right_on=["role", "level"],
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context="job title matching",
        top_k=1,   # 1 candidate → mock returns exactly 1 match per left row
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(result) == 2
    assert "salary" in result.columns


def test_multi_col_right_on_no_temp_key_leaked():
    """__right_key__ temp column must not appear in output."""
    df1 = pd.DataFrame({"title": ["Software Engineer"]})
    df2 = pd.DataFrame({"role": ["Software"], "level": ["Engineer"]})
    result = fuzzy_join(
        df1, df2,
        left_on="title",
        right_on=["role", "level"],
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context="job title matching",
        top_k=1,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert "__right_key__" not in result.columns


def test_multi_col_both_sides():
    """Both left_on and right_on as lists."""
    df1 = pd.DataFrame({"drug": ["aspirin"], "form": ["tablet"]})
    df2 = pd.DataFrame({"brand": ["Bayer"], "type": ["Aspirin Tablet"]})
    result = fuzzy_join(
        df1, df2,
        left_on=["drug", "form"],
        right_on=["brand", "type"],
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert isinstance(result, pd.DataFrame)
    assert "__left_key__" not in result.columns
    assert "__right_key__" not in result.columns


def test_multi_col_three_left_keys():
    """3-column left_on must concatenate all three and not leak temp key."""
    df1 = pd.DataFrame({
        "drug": ["aspirin"],
        "form": ["tablet"],
        "strength": ["100mg"],
        "dose": [1],
    })
    df2 = pd.DataFrame({"brand": ["aspirin tablet 100mg"], "price": [5.0]})
    result = fuzzy_join(
        df1, df2,
        left_on=["drug", "form", "strength"],
        right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(result) == 1
    assert "__left_key__" not in result.columns
    assert "dose" in result.columns
    assert "price" in result.columns


def test_multi_col_four_left_keys():
    """4-column left_on must concatenate all four and not leak temp key."""
    df1 = pd.DataFrame({
        "drug": ["aspirin"],
        "form": ["tablet"],
        "strength": ["100mg"],
        "route": ["oral"],
        "dose": [1],
    })
    df2 = pd.DataFrame({"brand": ["aspirin tablet 100mg oral"], "price": [5.0]})
    result = fuzzy_join(
        df1, df2,
        left_on=["drug", "form", "strength", "route"],
        right_on="brand",
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(result) == 1
    assert "__left_key__" not in result.columns
    assert "dose" in result.columns


def test_multi_col_three_right_keys():
    """3-column right_on must concatenate all three and not leak temp key."""
    df1 = pd.DataFrame({"title": ["aspirin tablet 100mg"]})
    df2 = pd.DataFrame({
        "drug": ["aspirin"],
        "form": ["tablet"],
        "strength": ["100mg"],
        "price": [5.0],
    })
    result = fuzzy_join(
        df1, df2,
        left_on="title",
        right_on=["drug", "form", "strength"],
        llm_fn=mock_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(result) == 1
    assert "__right_key__" not in result.columns
    assert "price" in result.columns


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

def test_duplicate_left_rows_fan_out():
    """80k-style: df1 with duplicate left values — LLM called once per unique value, rows fan out."""
    # "aspirin" appears 3 times in df1 → should match 3 output rows (one per original row)
    df1 = pd.DataFrame({
        "drug": ["aspirin", "aspirin", "aspirin", "ibuprofen"],
        "patient": [1, 2, 3, 4],
    })
    df2 = pd.DataFrame({"brand": ["Bayer Aspirin", "Advil"], "price": [5.0, 8.0]})

    llm_call_count = []

    def counting_llm(prompt: str) -> str:
        import re
        llm_call_count.append(1)
        pairs = re.findall(r'(\d+)\. LEFT:', prompt)
        count = len(pairs)
        import json
        return json.dumps([{"index": i, "score": 0.95, "reasoning": "mock"} for i in range(count)])

    result = fuzzy_join(
        df1, df2,
        left_on="drug", right_on="brand",
        llm_fn=counting_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    # 2 unique left values → 2 LLM calls (not 4)
    assert len(llm_call_count) == 2, f"Expected 2 LLM calls, got {len(llm_call_count)}"
    # 3 "aspirin" rows + 1 "ibuprofen" row all matched → 4 output rows
    assert len(result) == 4
    assert "patient" in result.columns
    assert "price" in result.columns


def test_all_duplicate_left_rows():
    """All left rows identical → exactly 1 LLM call, result fans out to all rows."""
    df1 = pd.DataFrame({"drug": ["aspirin"] * 5, "patient": range(5)})
    df2 = pd.DataFrame({"brand": ["Bayer Aspirin"], "price": [5.0]})

    llm_call_count = []

    def counting_llm(prompt: str) -> str:
        import re, json
        llm_call_count.append(1)
        pairs = re.findall(r'(\d+)\. LEFT:', prompt)
        return json.dumps([{"index": i, "score": 0.95, "reasoning": "mock"} for i in range(len(pairs))])

    result = fuzzy_join(
        df1, df2,
        left_on="drug", right_on="brand",
        llm_fn=counting_llm,
        embed_fn=mock_embed,
        context=CTX,
        top_k=1,
        how="inner",
        llm_concurrency=1, embed_concurrency=1,
    )
    assert len(llm_call_count) == 1, f"Expected 1 LLM call, got {len(llm_call_count)}"
    assert len(result) == 5  # all 5 rows matched
