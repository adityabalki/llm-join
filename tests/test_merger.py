import pandas as pd
import pytest
from llm_join.merger import Merger, MatchResult


def make_dfs():
    df1 = pd.DataFrame({"drug": ["aspirin", "ibuprofen"], "dose": [100, 200]})
    df2 = pd.DataFrame({"brand": ["Bayer Aspirin", "Advil"], "price": [5.0, 8.0]})
    return df1, df2


def test_inner_join_matches_rows():
    df1, df2 = make_dfs()
    matches = [
        MatchResult(left_val="aspirin", right_val="Bayer Aspirin", score=0.95, reasoning="same drug"),
        MatchResult(left_val="ibuprofen", right_val="Advil", score=0.88, reasoning="brand name"),
    ]
    result = Merger().merge(df1, df2, "drug", "brand", matches, how="inner")
    assert len(result) == 2
    assert "dose" in result.columns
    assert "price" in result.columns


def test_threshold_filters_low_scores():
    df1, df2 = make_dfs()
    matches = [
        MatchResult(left_val="aspirin", right_val="Bayer Aspirin", score=0.95, reasoning=""),
        # ibuprofen has no match — not in matches list
    ]
    result = Merger().merge(df1, df2, "drug", "brand", matches, how="inner")
    assert len(result) == 1
    assert result.iloc[0]["drug"] == "aspirin"


def test_left_join_keeps_unmatched():
    df1, df2 = make_dfs()
    matches = [
        MatchResult(left_val="aspirin", right_val="Bayer Aspirin", score=0.95, reasoning=""),
    ]
    result = Merger().merge(df1, df2, "drug", "brand", matches, how="left")
    assert len(result) == 2


def test_return_reasoning_adds_columns():
    df1, df2 = make_dfs()
    matches = [
        MatchResult(left_val="aspirin", right_val="Bayer Aspirin", score=0.95, reasoning="same drug"),
    ]
    result = Merger().merge(df1, df2, "drug", "brand", matches, how="inner", return_reasoning=True)
    assert "_llm_score" in result.columns
    assert "_llm_reasoning" in result.columns
    assert "_embed_rank" in result.columns
    assert "_match_method" in result.columns
    assert result.iloc[0]["_llm_score"] == 0.95
    assert result.iloc[0]["_match_method"] == "llm"


def test_no_matches_returns_empty_inner():
    df1, df2 = make_dfs()
    result = Merger().merge(df1, df2, "drug", "brand", [], how="inner")
    assert len(result) == 0
