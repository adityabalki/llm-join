"""
Thorough parameter-level tests for fuzzy_join.

Each test targets ONE parameter in isolation so failures point to the exact knob.
Parameters covered:
    context, llm_threshold, embed_skip_threshold, top_k, how,
    max_llm_calls, max_retries, return_reasoning, match_all,
    llm_concurrency, column_context, left_on / right_on (multi-col)
"""
import json
import re
import warnings
import numpy as np
import pandas as pd
import pytest
from llm_join import fuzzy_join


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CTX = "pharmaceutical drug names — match generic INN names to US brand names"

DF1 = pd.DataFrame({"drug": ["aspirin", "ibuprofen"], "dose": [100, 200]})
DF2 = pd.DataFrame({"brand": ["Bayer Aspirin", "Advil"], "price": [5.0, 8.0]})


def mock_embed(texts):
    """Text-hash-based unit vectors. Different texts → cosine < 1.0."""
    dim = 32
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        seed = abs(hash(text)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        vecs[i] = rng.randn(dim).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)


def all_same_embed(texts):
    """Every text gets the same unit vector → cosine=1.0 between any pair.
    Use this to reliably trigger embed_skip_threshold."""
    n = len(texts)
    return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), (n, 1))


def mock_llm(prompt: str) -> str:
    """Returns index 0 with score 0.95 for each LEFT value in prompt."""
    pairs = re.findall(r"(\d+)\. LEFT:", prompt)
    count = len(pairs)
    return json.dumps([
        {"index": i, "score": 0.95, "reasoning": "mock match"}
        for i in range(count)
    ])


def failing_llm(prompt: str) -> str:
    raise RuntimeError("LLM must not be called")


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

class TestContext:
    def test_missing_context_raises_type_error(self):
        with pytest.raises(TypeError):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                llm_concurrency=1,
                # context omitted
            )

    def test_whitespace_only_context_raises(self):
        with pytest.raises(ValueError, match="context must not be empty"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context="   ",
                llm_concurrency=1,
            )

    def test_valid_context_runs(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, llm_concurrency=1,
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# llm_threshold
# ---------------------------------------------------------------------------

class TestLlmThreshold:
    def test_low_threshold_matches_all(self):
        # mock_llm returns 0.95; 0.95 >= 0.01 → eligible, but match_all=False → ONE match per left row
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=2, llm_threshold=0.01,
            how="inner", llm_concurrency=1,
        )
        assert len(result) == 2

    def test_high_threshold_matches_nothing(self):
        # mock_llm returns 0.95; 0.95 < 1.0 → no matches
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=2, llm_threshold=1.0,
            how="inner", llm_concurrency=1,
        )
        assert len(result) == 0

    def test_exact_threshold_matches(self):
        # mock_llm returns 0.95; 0.95 >= 0.95 → matches
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_threshold=0.95,
            how="inner", llm_concurrency=1,
        )
        assert len(result) == 2

    def test_above_score_no_match(self):
        # mock_llm returns 0.95; threshold 0.96 → no match
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_threshold=0.96,
            how="inner", llm_concurrency=1,
        )
        assert len(result) == 0

    def test_invalid_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="llm_threshold"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, llm_threshold=1.5, llm_concurrency=1,
            )

    def test_invalid_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="llm_threshold"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, llm_threshold=0.0, llm_concurrency=1,
            )


# ---------------------------------------------------------------------------
# embed_skip_threshold
# ---------------------------------------------------------------------------

class TestEmbedSkipThreshold:
    def test_default_1_0_skips_exact_matches(self):
        """embed_skip_threshold=1.0 (default): identical vectors skip LLM."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm,        # proves LLM never called
            embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        assert len(result) == 2
        assert (result["_match_method"] == "embed_skip").all()

    def test_embed_skip_reasoning_contains_threshold(self):
        """Reasoning explains why LLM was skipped."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm, embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        assert result["_llm_reasoning"].str.contains("embed_skip_threshold").all()

    def test_low_embed_skip_threshold_skips_llm(self):
        """embed_skip_threshold=0.99: cosine=1.0 >= 0.99 → skips LLM."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm, embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            embed_skip_threshold=0.99, return_reasoning=True, llm_concurrency=1,
        )
        assert (result["_match_method"] == "embed_skip").all()

    def test_embed_skip_does_not_fire_when_cosine_below_threshold(self):
        """mock_embed gives cosine < 1.0 → default embed_skip_threshold=1.0 doesn't fire → LLM called."""
        llm_called = []

        def counting_llm(prompt):
            llm_called.append(1)
            return mock_llm(prompt)

        fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=counting_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
        )
        assert len(llm_called) > 0, "LLM should have been called (embed cosine < 1.0)"

    def test_embed_skip_match_method_column(self):
        """match_method='embed_skip' when embed score triggers the shortcut."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm, embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        assert "_match_method" in result.columns
        assert set(result["_match_method"]) == {"embed_skip"}

    def test_embed_skip_score_is_cosine_similarity(self):
        """_llm_score for embed_skip rows is the cosine similarity (1.0 for identical vectors)."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm, embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        assert ((result["_llm_score"] - 1.0).abs() < 1e-5).all()

    def test_invalid_embed_skip_threshold_raises(self):
        with pytest.raises(ValueError, match="embed_skip_threshold"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, embed_skip_threshold=1.5, llm_concurrency=1,
            )

    def test_invalid_embed_skip_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="embed_skip_threshold"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, embed_skip_threshold=0.0, llm_concurrency=1,
            )


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------

class TestTopK:
    def test_top_k_one_sends_one_candidate(self):
        """top_k=1 → each LLM prompt has exactly 1 candidate."""
        prompts = []

        def capturing_llm(prompt):
            prompts.append(prompt)
            return mock_llm(prompt)

        fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=capturing_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
        )
        for p in prompts:
            candidates_in_prompt = re.findall(r"^\d+\.", p, re.MULTILINE)
            assert len(candidates_in_prompt) == 1

    def test_top_k_two_sends_two_candidates(self):
        """top_k=2 → each prompt has 2 candidates."""
        prompts = []

        def capturing_llm(prompt):
            prompts.append(prompt)
            return mock_llm(prompt)

        fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=capturing_llm, embed_fn=mock_embed,
            context=CTX, top_k=2, llm_concurrency=1,
        )
        for p in prompts:
            candidates_in_prompt = re.findall(r"^\d+\.", p, re.MULTILINE)
            assert len(candidates_in_prompt) == 2

    def test_top_k_larger_than_corpus_capped(self):
        """top_k=10 with only 2 right rows → retrieves 2, no crash."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=10, llm_concurrency=1,
        )
        assert isinstance(result, pd.DataFrame)

    def test_top_k_invalid_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, top_k=0, llm_concurrency=1,
            )


# ---------------------------------------------------------------------------
# how (join type)
# ---------------------------------------------------------------------------

class TestHow:
    def _df1_partial(self):
        return pd.DataFrame({"drug": ["aspirin", "unknown_drug"], "dose": [100, 999]})

    def test_inner_keeps_only_matched(self):
        result = fuzzy_join(
            self._df1_partial(), DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1, how="inner",
        )
        assert "dose" in result.columns
        assert "price" in result.columns
        # inner: only rows where match was found above threshold
        assert result["price"].notna().all()

    def test_left_keeps_all_left_rows(self):
        result = fuzzy_join(
            self._df1_partial(), DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1, how="left",
        )
        # All left rows must appear
        assert len(result) >= 2
        assert set(result["drug"]).issuperset({"aspirin", "unknown_drug"})

    def test_inner_does_not_have_nan_prices(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1, how="inner",
        )
        assert result["price"].notna().all()

    def test_full_outer_join(self):
        """how='full' returns all rows from both sides (full outer join)."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, how="full", llm_concurrency=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert "dose" in result.columns
        assert "price" in result.columns

    def test_outer_string_now_invalid(self):
        """how='outer' is no longer valid — must use 'full'."""
        with pytest.raises(ValueError, match="how"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, how="outer", llm_concurrency=1,
            )

    def test_invalid_how_raises(self):
        with pytest.raises(ValueError, match="how"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, how="cross", llm_concurrency=1,
            )


# ---------------------------------------------------------------------------
# max_llm_calls
# ---------------------------------------------------------------------------

class TestMaxLlmCalls:
    def test_max_llm_calls_warns_when_exceeded(self, recwarn):
        fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, max_llm_calls=1,
            llm_concurrency=1,
        )
        assert any("max_llm_calls" in str(w.message) for w in recwarn)

    def test_max_llm_calls_returns_partial(self, recwarn):
        """With max_llm_calls=1 and 2 left rows, result is partial (≤ 2 rows)."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, max_llm_calls=1, how="inner",
            llm_concurrency=1,
        )
        assert len(result) <= 2

    def test_max_llm_calls_not_exceeded_no_warn(self, recwarn):
        fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, max_llm_calls=100,
            llm_concurrency=1,
        )
        assert not any("max_llm_calls" in str(w.message) for w in recwarn)

    def test_invalid_max_llm_calls_raises(self):
        with pytest.raises(ValueError, match="max_llm_calls"):
            fuzzy_join(
                DF1, DF2, left_on="drug", right_on="brand",
                llm=mock_llm, embed_fn=mock_embed,
                context=CTX, max_llm_calls=0, llm_concurrency=1,
            )


# ---------------------------------------------------------------------------
# max_retries
# ---------------------------------------------------------------------------

class TestMaxRetries:
    def test_max_retries_zero_no_retry_on_failure(self, recwarn):
        """max_retries=0 → fail immediately, no retry warnings."""
        def fail_once_llm(prompt):
            raise RuntimeError("API down")

        fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=fail_once_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, max_retries=0,
            return_reasoning=True, llm_concurrency=1,
        )
        retry_warns = [w for w in recwarn if "Retrying" in str(w.message)]
        assert len(retry_warns) == 0

    def test_max_retries_zero_uses_embed_fallback(self):
        """max_retries=0 + always-failing LLM → embed_fallback for all rows."""
        def fail_llm(prompt):
            raise RuntimeError("always fails")

        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=fail_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, max_retries=0,
            return_reasoning=True, llm_concurrency=1,
        )
        assert (result["_match_method"] == "embed_fallback").all()

    def test_embed_fallback_reasoning_text(self):
        """Embed fallback sets human-readable reasoning."""
        def fail_llm(prompt):
            raise RuntimeError("always fails")

        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=fail_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, max_retries=0,
            return_reasoning=True, llm_concurrency=1,
        )
        assert result["_llm_reasoning"].str.contains("embed rank-0 fallback").all()


# ---------------------------------------------------------------------------
# return_reasoning
# ---------------------------------------------------------------------------

class TestReturnReasoning:
    def test_false_no_debug_columns(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, return_reasoning=False, llm_concurrency=1,
        )
        for col in ["_llm_score", "_llm_reasoning", "_embed_rank", "_match_method", "_llm_candidates"]:
            assert col not in result.columns

    def test_true_adds_all_debug_columns(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, return_reasoning=True, llm_concurrency=1,
        )
        for col in ["_llm_score", "_llm_reasoning", "_embed_rank", "_match_method", "_llm_candidates"]:
            assert col in result.columns

    def test_llm_score_is_float(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, return_reasoning=True, llm_concurrency=1,
        )
        assert result["_llm_score"].dtype in [float, "float64"]

    def test_llm_candidates_is_list_of_dicts(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, return_reasoning=True, llm_concurrency=1,
        )
        for candidates in result["_llm_candidates"]:
            assert isinstance(candidates, list)
            for c in candidates:
                assert "candidate" in c
                assert "embed_score" in c

    def test_match_method_llm_for_normal_matches(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, return_reasoning=True, llm_concurrency=1,
        )
        assert (result["_match_method"] == "llm").all()


# ---------------------------------------------------------------------------
# match_all
# ---------------------------------------------------------------------------

class TestMatchAll:
    def test_match_all_false_returns_one_per_row(self):
        """Default: only best match per left row."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_threshold=0.01,
            match_all=False, how="inner", llm_concurrency=1,
        )
        # 2 left rows × 1 candidate each = 2 rows max
        assert len(result) == 2

    def test_match_all_true_returns_all_above_threshold(self):
        """match_all=True: all candidates above threshold returned."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=2, llm_threshold=0.01,
            match_all=True, how="inner", llm_concurrency=1,
        )
        # 2 left × 2 candidates each = 4 rows
        assert len(result) == 4

    def test_match_all_reasoning_contains_count(self):
        """match_all annotates reasoning with count of candidates returned."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=2, llm_threshold=0.01,
            match_all=True, return_reasoning=True, llm_concurrency=1,
        )
        assert result["_llm_reasoning"].str.contains("match_all").all()


# ---------------------------------------------------------------------------
# llm_concurrency
# ---------------------------------------------------------------------------

class TestLlmConcurrency:
    def test_concurrency_1_sequential(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
        )
        assert isinstance(result, pd.DataFrame)

    def test_concurrency_2_same_result_as_sequential(self):
        r1 = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1, how="inner",
        )
        r2 = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=2, how="inner",
        )
        assert len(r1) == len(r2)
        assert set(r1.columns) == set(r2.columns)

    def test_concurrency_preserves_row_count(self):
        df1_large = pd.DataFrame({
            "drug": ["aspirin", "ibuprofen", "paracetamol", "metformin"],
        })
        df2_large = pd.DataFrame({
            "brand": ["Bayer Aspirin", "Advil", "Tylenol", "Glucophage"],
        })
        result = fuzzy_join(
            df1_large, df2_large, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=3, how="inner",
        )
        assert len(result) == 4


# ---------------------------------------------------------------------------
# column_context
# ---------------------------------------------------------------------------

class TestColumnContext:
    def test_column_context_included_in_prompt(self):
        prompts = []

        def capturing_llm(prompt):
            prompts.append(prompt)
            return mock_llm(prompt)

        fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=capturing_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
            column_context={
                "drug": "generic INN drug name",
                "brand": "US commercial brand name",
            },
        )
        assert any("generic INN drug name" in p for p in prompts)
        assert any("US commercial brand name" in p for p in prompts)

    def test_column_context_empty_dict_runs(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
            column_context={},
        )
        assert isinstance(result, pd.DataFrame)

    def test_column_context_none_runs(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
            column_context=None,
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# left_on / right_on (multi-column)
# ---------------------------------------------------------------------------

class TestMultiColumnKeys:
    def test_multi_left_on_no_temp_col_leaked(self):
        df1 = pd.DataFrame({"drug": ["aspirin"], "form": ["tablet"], "dose": [100]})
        df2 = pd.DataFrame({"brand": ["Bayer Aspirin Tablet"], "price": [5.0]})
        result = fuzzy_join(
            df1, df2, left_on=["drug", "form"], right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
        )
        assert "__left_key__" not in result.columns

    def test_multi_right_on_no_temp_col_leaked(self):
        df1 = pd.DataFrame({"title": ["Software Engineer"]})
        df2 = pd.DataFrame({"role": ["Software"], "level": ["Engineer"]})
        result = fuzzy_join(
            df1, df2, left_on="title", right_on=["role", "level"],
            llm=mock_llm, embed_fn=mock_embed,
            context="job title matching", top_k=1, llm_concurrency=1,
        )
        assert "__right_key__" not in result.columns

    def test_multi_left_on_result_has_payload_columns(self):
        df1 = pd.DataFrame({"drug": ["aspirin", "ibuprofen"], "form": ["tablet", "capsule"], "dose": [100, 200]})
        df2 = pd.DataFrame({"brand": ["Bayer Aspirin Tablet", "Advil Capsule"], "price": [5.0, 8.0]})
        result = fuzzy_join(
            df1, df2, left_on=["drug", "form"], right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, how="inner", llm_concurrency=1,
        )
        assert "dose" in result.columns
        assert "price" in result.columns
        assert len(result) == 2

    def test_multi_right_on_result_has_payload_columns(self):
        df1 = pd.DataFrame({"title": ["Software Engineer", "Data Analyst"], "dept": [1, 2]})
        df2 = pd.DataFrame({"role": ["Software", "Data"], "level": ["Engineer", "Analyst"], "salary": [100, 90]})
        result = fuzzy_join(
            df1, df2, left_on="title", right_on=["role", "level"],
            llm=mock_llm, embed_fn=mock_embed,
            context="job title matching", top_k=1, how="inner", llm_concurrency=1,
        )
        assert "salary" in result.columns
        assert len(result) == 2

    def test_multi_both_sides_no_temp_cols_leaked(self):
        df1 = pd.DataFrame({"drug": ["aspirin"], "form": ["tablet"]})
        df2 = pd.DataFrame({"brand": ["Bayer"], "type": ["Aspirin Tablet"]})
        result = fuzzy_join(
            df1, df2, left_on=["drug", "form"], right_on=["brand", "type"],
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
        )
        assert "__left_key__" not in result.columns
        assert "__right_key__" not in result.columns

    def test_three_left_keys_concatenated(self):
        df1 = pd.DataFrame({
            "drug": ["aspirin"], "form": ["tablet"], "strength": ["100mg"], "dose": [1],
        })
        df2 = pd.DataFrame({"brand": ["aspirin tablet 100mg"], "price": [5.0]})
        result = fuzzy_join(
            df1, df2, left_on=["drug", "form", "strength"], right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, llm_concurrency=1,
        )
        assert len(result) == 1
        assert "dose" in result.columns
        assert "__left_key__" not in result.columns


# ---------------------------------------------------------------------------
# embed_skip + return_reasoning interaction
# ---------------------------------------------------------------------------

class TestEmbedSkipWithReasoning:
    def test_embed_skip_does_not_populate_llm_candidates(self):
        """embed_skip rows bypassed LLM → _llm_candidates should be None (no prompt was built)."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm, embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        # _llm_candidates is None for embed_skip rows (no LLM call was made)
        assert result["_llm_candidates"].isna().all()

    def test_embed_skip_score_in_valid_range(self):
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm, embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        assert (result["_llm_score"] >= 0.0).all()
        assert (result["_llm_score"] <= 1.0).all()


# ---------------------------------------------------------------------------
# drop_duplicates safety net
# ---------------------------------------------------------------------------

class TestNoDuplicates:
    def test_identical_df1_rows_deduplicated(self):
        """df1 has two completely identical rows → output must not have duplicate rows."""
        df1 = pd.DataFrame({"drug": ["aspirin", "aspirin"], "dose": [100, 100]})  # truly identical
        df2 = pd.DataFrame({"brand": ["Bayer Aspirin"], "price": [5.0]})
        result = fuzzy_join(
            df1, df2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, how="inner", llm_concurrency=1,
        )
        # Both df1 rows are identical in all columns; after join both match same right row
        # drop_duplicates should collapse to 1 row
        assert len(result) == 1
        assert not result.duplicated().any()

    def test_distinct_df1_rows_not_dropped(self):
        """df1 rows differ in non-key column → both preserved after join."""
        df1 = pd.DataFrame({"drug": ["aspirin", "aspirin"], "dose": [100, 200]})  # different dose
        df2 = pd.DataFrame({"brand": ["Bayer Aspirin"], "price": [5.0]})
        result = fuzzy_join(
            df1, df2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, how="inner", llm_concurrency=1,
        )
        # Rows differ in dose → NOT duplicates → both preserved
        assert len(result) == 2
        assert set(result["dose"]) == {100, 200}

    def test_multiple_left_values_same_right_no_cartesian(self):
        """Two different left values both match the same right value → 2 output rows, not 4."""
        # RC2: multiple left vals → same right val should not cartesian-explode
        df1 = pd.DataFrame({"drug": ["aspirin 100mg", "aspirin 200mg"], "patient": [1, 2]})
        df2 = pd.DataFrame({"brand": ["Bayer Aspirin"], "price": [5.0]})
        result = fuzzy_join(
            df1, df2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, how="inner", llm_concurrency=1,
        )
        assert len(result) == 2
        assert set(result["patient"]) == {1, 2}

    def test_no_duplicates_with_return_reasoning(self):
        """drop_duplicates must not crash when _llm_candidates (list column) is present."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        assert not result.drop(columns=["_llm_candidates"]).duplicated().any()

    def test_no_duplicates_with_embed_skip_and_reasoning(self):
        """embed_skip path + return_reasoning=True must not crash or produce duplicates."""
        result = fuzzy_join(
            DF1, DF2, left_on="drug", right_on="brand",
            llm=failing_llm, embed_fn=all_same_embed,
            context=CTX, top_k=1, how="inner",
            return_reasoning=True, llm_concurrency=1,
        )
        assert not result.drop(columns=["_llm_candidates"]).duplicated().any()

    def test_output_never_has_more_rows_than_df1_inner(self):
        """inner join, match_all=False: output rows ≤ df1 rows (each left row gets at most 1 match)."""
        df1 = pd.DataFrame({"drug": ["aspirin", "ibuprofen", "paracetamol"], "dose": [100, 200, 500]})
        df2 = pd.DataFrame({"brand": ["Bayer Aspirin", "Advil", "Tylenol"], "price": [5.0, 8.0, 3.0]})
        result = fuzzy_join(
            df1, df2, left_on="drug", right_on="brand",
            llm=mock_llm, embed_fn=mock_embed,
            context=CTX, top_k=1, how="inner", llm_concurrency=1,
        )
        assert len(result) <= len(df1)
