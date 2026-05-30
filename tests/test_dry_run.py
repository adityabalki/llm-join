"""Adversarial tests for dry_run=True mode in fuzzy_join."""
import os
import pytest
import pandas as pd
import numpy as np

from llm_join.dry_run import DryRunResult


def _make_dfs():
    df1 = pd.DataFrame({"vendor": ["apple", "banana", "apple", "cherry"]})
    df2 = pd.DataFrame({"supplier": ["apple inc", "banana corp", "cherry ltd"]})
    return df1, df2


def _dummy_llm(prompt: str) -> str:
    raise AssertionError("llm_fn must not be called in dry_run mode")


def _dummy_embed(texts):
    raise AssertionError("embed_fn must not be called in dry_run mode")


def _valid_embed(texts):
    rng = np.random.default_rng(0)
    return rng.random((len(texts), 4)).astype("float32")


def _run_dry(**kwargs):
    from llm_join.join import fuzzy_join
    df1, df2 = _make_dfs()
    defaults = dict(
        left_on="vendor", right_on="supplier",
        llm_fn=_dummy_llm, embed_fn=_dummy_embed,
        context="company names",
        llm_concurrency=1, embed_concurrency=1,
        dry_run=True,
    )
    defaults.update(kwargs)
    return fuzzy_join(df1, df2, **defaults)


def test_returns_dryrunresult_not_dataframe():
    result = _run_dry()
    assert isinstance(result, DryRunResult), f"Expected DryRunResult, got {type(result)}"


def test_never_calls_llm_or_embed_fn():
    # _dummy_llm and _dummy_embed raise AssertionError if called
    # If this test passes without raising, the fns were never called
    result = _run_dry()
    assert isinstance(result, DryRunResult)


def test_catches_missing_left_column():
    result = _run_dry(left_on="nonexistent_col")
    assert not result.is_valid
    assert any("nonexistent_col" in e for e in result.validation_errors)


def test_catches_missing_right_column():
    result = _run_dry(right_on="nonexistent_col")
    assert not result.is_valid
    assert any("nonexistent_col" in e for e in result.validation_errors)


def test_catches_empty_context():
    result = _run_dry(context="")
    assert not result.is_valid
    assert any("context" in e.lower() for e in result.validation_errors)


def test_catches_invalid_threshold():
    result = _run_dry(llm_threshold=1.5)
    assert not result.is_valid
    assert any("llm_threshold" in e for e in result.validation_errors)


def test_catches_invalid_retrieval():
    result = _run_dry(retrieval="typo_retrieval")
    assert not result.is_valid
    assert any("retrieval" in e for e in result.validation_errors)


def test_multiple_errors_all_collected():
    """Non-fail-fast: all errors collected, not just first."""
    result = _run_dry(
        left_on="bad_col",
        context="",
        llm_threshold=1.5,
    )
    assert not result.is_valid
    # Must report all 3 errors, not just the first
    assert len(result.validation_errors) == 3, f"Expected exactly 3 errors but got {len(result.validation_errors)}: {result.validation_errors}"
    error_text = " ".join(result.validation_errors)
    assert "bad_col" in error_text
    assert "context" in error_text.lower()
    assert "llm_threshold" in error_text


def test_max_llm_calls_caps_estimate():
    """max_llm_calls=5 with 100 unique left vals -> n_llm_calls_max == 5."""
    from llm_join.join import fuzzy_join
    df1 = pd.DataFrame({"vendor": [f"vendor_{i}" for i in range(100)]})
    df2 = pd.DataFrame({"supplier": ["s1", "s2"]})
    result = fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_dummy_llm, embed_fn=_dummy_embed,
        context="company names",
        llm_concurrency=1, embed_concurrency=1,
        max_llm_calls=5,
        dry_run=True,
    )
    assert result.n_llm_calls_max == 5, f"Expected 5 but got {result.n_llm_calls_max}"
    assert result.n_left_unique == 100


def test_embed_skip_threshold_affects_min():
    """embed_skip_threshold < 1.0 -> n_llm_calls_min == 0; >= 1.0 -> min == max."""
    from llm_join.join import fuzzy_join
    df1, df2 = _make_dfs()

    result_skip = fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_dummy_llm, embed_fn=_dummy_embed,
        context="company names",
        llm_concurrency=1, embed_concurrency=1,
        embed_skip_threshold=0.9,
        dry_run=True,
    )
    assert result_skip.n_llm_calls_min == 0, f"Expected 0 but got {result_skip.n_llm_calls_min}"

    result_no_skip = fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_dummy_llm, embed_fn=_dummy_embed,
        context="company names",
        llm_concurrency=1, embed_concurrency=1,
        embed_skip_threshold=1.0,
        dry_run=True,
    )
    assert result_no_skip.n_llm_calls_min == result_no_skip.n_llm_calls_max
