"""Tests for JoinStats tracking + verbose logging output."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from llm_join import fuzzy_join
from llm_join.stats import JoinStats, is_rate_limit_error


def mock_embed(texts):
    """Content-based embed: same text always gets same vector regardless of batch position."""
    dim = 8
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        seed = abs(hash(t)) % (2**31)
        rng = np.random.RandomState(seed)
        vecs[i] = rng.randn(dim).astype("float32")
    # Normalize so same text gives unit cosine = 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)


def make_llm(score=0.9, reasoning="match"):
    def llm(prompt):
        return json.dumps([{"index": 0, "score": score, "reasoning": reasoning}])
    return llm


# ---------------------------------------------------------------------------
# JoinStats unit tests
# ---------------------------------------------------------------------------

class TestJoinStatsUnit:
    def test_defaults_all_zero(self):
        s = JoinStats()
        assert s.n_left_total == 0
        assert s.n_llm_called == 0
        assert s.elapsed_seconds == 0.0

    def test_inc_increments(self):
        s = JoinStats()
        s.inc("n_llm_called")
        s.inc("n_llm_called", 4)
        assert s.n_llm_called == 5

    def test_summary_line_includes_all_fields(self):
        s = JoinStats(
            n_left_total=100, n_left_unique=80,
            n_right_total=50, n_right_unique=45,
            n_embed_skipped=20, n_llm_called=60,
            n_rate_limited=2, n_failed=1, elapsed_seconds=12.34,
        )
        line = s.summary_line()
        assert "llm-join:" in line
        assert "100 left" in line
        assert "deduped to 80" in line
        assert "20 embed_skipped" in line
        assert "60 LLM" in line
        assert "2 rate-limited" in line
        assert "1 failed" in line
        assert "12.3s" in line


class TestRateLimitDetection:
    def test_class_name_ratelimit(self):
        class RateLimitError(Exception): pass
        assert is_rate_limit_error(RateLimitError("oops")) is True

    def test_class_name_too_many_requests(self):
        class TooManyRequests(Exception): pass
        assert is_rate_limit_error(TooManyRequests()) is True

    def test_message_429(self):
        assert is_rate_limit_error(Exception("HTTP 429 throttled")) is True

    def test_message_rate_limit(self):
        assert is_rate_limit_error(Exception("rate limit exceeded")) is True

    def test_unrelated_error_false(self):
        assert is_rate_limit_error(ValueError("bad input")) is False
        assert is_rate_limit_error(KeyError("missing")) is False


# ---------------------------------------------------------------------------
# Summary always prints at end
# ---------------------------------------------------------------------------

class TestSummaryAlwaysPrints:
    def test_summary_appears_in_stderr(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha", "beta"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
        )
        captured = capsys.readouterr()
        assert "llm-join:" in captured.err
        assert "left" in captured.err
        assert "LLM" in captured.err

    def test_summary_records_left_total_and_unique(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "alpha", "beta"]})  # 1 dup
        df2 = pd.DataFrame({"b": ["alpha", "beta"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
        )
        captured = capsys.readouterr()
        assert "3 left" in captured.err
        assert "deduped to 2" in captured.err


# ---------------------------------------------------------------------------
# verbose=0 (default) — no per-record, no progress
# ---------------------------------------------------------------------------

class TestVerboseZero:
    def test_no_per_record_lines_at_verbose_zero(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "beta", "gamma"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2", "gamma_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            verbose=0,
        )
        captured = capsys.readouterr()
        # No per-row log lines
        assert "Row:" not in captured.err
        # But summary still present
        assert "llm-join:" in captured.err


# ---------------------------------------------------------------------------
# verbose=2 — per-record log line
# ---------------------------------------------------------------------------

class TestVerboseTwo:
    def test_per_record_log_for_llm_matches(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(score=0.95), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            verbose=2,
        )
        captured = capsys.readouterr()
        # Each left val should produce a Row: log line
        row_lines = [line for line in captured.err.split("\n") if "Row:" in line]
        assert len(row_lines) >= 2
        # Lines should contain score, embed_rank, method
        for line in row_lines:
            assert "score=" in line
            assert "embed_rank=" in line

    def test_per_record_log_for_embed_skip(self, capsys):
        # Force embed_skip by using same texts on both sides (cosine = 1.0)
        df1 = pd.DataFrame({"a": ["identical"]})
        df2 = pd.DataFrame({"b": ["identical"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            verbose=2,
            embed_skip_threshold=0.99,
        )
        captured = capsys.readouterr()
        assert "embed_skip" in captured.err


# ---------------------------------------------------------------------------
# Stats counters via end-to-end fuzzy_join
# ---------------------------------------------------------------------------

class TestStatsCounters:
    def test_llm_called_counted(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "beta", "gamma"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2", "gamma_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
        )
        captured = capsys.readouterr()
        # 3 unique left vals = 3 LLM calls
        m = re.search(r"(\d+) LLM", captured.err)
        assert m is not None
        assert int(m.group(1)) == 3

    def test_embed_skipped_counted(self, capsys):
        # Identical texts → embed_skip
        df1 = pd.DataFrame({"a": ["identical1", "identical2"]})
        df2 = pd.DataFrame({"b": ["identical1", "identical2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            embed_skip_threshold=0.99,
        )
        captured = capsys.readouterr()
        m = re.search(r"(\d+) embed_skipped", captured.err)
        assert m is not None
        assert int(m.group(1)) >= 1

    def test_failed_counted_on_persistent_llm_failure(self, capsys):
        def failing_llm(prompt):
            raise RuntimeError("always fails")
        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["beta"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=failing_llm, embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            max_retries=0,
        )
        captured = capsys.readouterr()
        m = re.search(r"(\d+) failed", captured.err)
        assert m is not None
        assert int(m.group(1)) == 1

    def test_rate_limited_counted(self, capsys):
        attempt = [0]
        def rate_limited_llm(prompt):
            attempt[0] += 1
            if attempt[0] == 1:
                raise RuntimeError("HTTP 429 rate limit exceeded")
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "ok"}])

        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["alpha_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=rate_limited_llm, embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            max_retries=2,
        )
        captured = capsys.readouterr()
        m = re.search(r"(\d+) rate-limited", captured.err)
        assert m is not None
        assert int(m.group(1)) >= 1


# ---------------------------------------------------------------------------
# Backward compat — scorer works without stats/verbose passed
# ---------------------------------------------------------------------------

class TestScorerBackwardCompat:
    def test_scorer_no_stats_no_crash(self):
        from llm_join.scorer import LLMScorer
        scorer = LLMScorer(make_llm())
        result = scorer.score("alpha", ["alpha_v2"], "test")
        assert len(result) == 1

    def test_scorer_with_stats(self):
        from llm_join.scorer import LLMScorer
        stats = JoinStats()
        scorer = LLMScorer(make_llm(), stats=stats)
        scorer.score("alpha", ["alpha_v2"], "test")
        assert stats.n_llm_called == 1


# ---------------------------------------------------------------------------
# Stats with concurrent paths (threaded + async LLM)
# ---------------------------------------------------------------------------

class TestStatsConcurrentPaths:
    def test_stats_correct_under_threaded_concurrency(self, capsys):
        df1 = pd.DataFrame({"a": [f"item_{i}" for i in range(20)]})
        df2 = pd.DataFrame({"b": [f"thing_{i}" for i in range(20)]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=8,  # threaded path
        )
        captured = capsys.readouterr()
        m = re.search(r"(\d+) LLM", captured.err)
        assert int(m.group(1)) == 20  # all 20 unique scored

    def test_stats_correct_under_async_concurrency(self, capsys):
        async def async_llm(prompt):
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "ok"}])

        df1 = pd.DataFrame({"a": [f"item_{i}" for i in range(15)]})
        df2 = pd.DataFrame({"b": [f"thing_{i}" for i in range(15)]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=async_llm, embed_fn=mock_embed,
            context="test",
            llm_concurrency=5,  # async path
        )
        captured = capsys.readouterr()
        m = re.search(r"(\d+) LLM", captured.err)
        assert int(m.group(1)) == 15

    def test_stats_with_async_embed_and_async_llm(self, capsys):
        async def async_embed(texts):
            return mock_embed(texts)

        async def async_llm(prompt):
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "ok"}])

        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=async_llm, embed_fn=async_embed,
            context="test",
            llm_concurrency=2,
            embed_concurrency=2,
        )
        captured = capsys.readouterr()
        assert "2 LLM" in captured.err

    def test_stats_with_sync_embed_async_llm(self, capsys):
        async def async_llm(prompt):
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "ok"}])

        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=async_llm, embed_fn=mock_embed,  # sync embed
            context="test",
            llm_concurrency=2,
        )
        captured = capsys.readouterr()
        assert "llm-join:" in captured.err


# ---------------------------------------------------------------------------
# JoinStats thread-safety (no lost counts under concurrent inc)
# ---------------------------------------------------------------------------

class TestStatsThreadSafety:
    def test_concurrent_inc_no_lost_updates(self):
        from concurrent.futures import ThreadPoolExecutor
        stats = JoinStats()
        N = 1000
        with ThreadPoolExecutor(max_workers=20) as ex:
            list(ex.map(lambda _: stats.inc("n_llm_called"), range(N)))
        assert stats.n_llm_called == N

    def test_concurrent_inc_different_fields(self):
        from concurrent.futures import ThreadPoolExecutor
        stats = JoinStats()
        N = 500
        def work(i):
            if i % 2 == 0:
                stats.inc("n_llm_called")
            else:
                stats.inc("n_rate_limited")
        with ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(work, range(N)))
        assert stats.n_llm_called == N // 2
        assert stats.n_rate_limited == N // 2


# ---------------------------------------------------------------------------
# Verbose=1 progress bars don't crash
# ---------------------------------------------------------------------------

class TestVerboseOneProgress:
    def test_verbose_one_runs_without_error(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "beta", "gamma"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2", "gamma_v2"]})
        # Should not crash whether tqdm installed or not
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=2,
            verbose=1,
        )
        captured = capsys.readouterr()
        # Summary still prints
        assert "llm-join:" in captured.err

    def test_verbose_one_with_sequential_llm(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,  # sequential path
            verbose=1,
        )
        captured = capsys.readouterr()
        assert "llm-join:" in captured.err

    def test_verbose_one_with_async_llm(self, capsys):
        async def async_llm(prompt):
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "ok"}])

        df1 = pd.DataFrame({"a": ["alpha", "beta", "gamma"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2", "gamma_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=async_llm, embed_fn=mock_embed,
            context="test",
            llm_concurrency=2,
            verbose=1,
        )
        captured = capsys.readouterr()
        assert "llm-join:" in captured.err


# ---------------------------------------------------------------------------
# Verbose=2 detail: per-record exact format
# ---------------------------------------------------------------------------

class TestVerboseTwoFormat:
    def test_per_record_format_contains_required_fields(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["alpha_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(score=0.95, reasoning="strong match"),
            embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            verbose=2,
        )
        captured = capsys.readouterr()
        # tqdm uses \r — Row lines may share a line with progress bars; check substring
        assert "Row:" in captured.err
        # Find the section after "Row:"
        idx = captured.err.find("Row:")
        chunk = captured.err[idx:idx + 200]
        assert '"alpha"' in chunk
        assert '->' in chunk
        assert '"alpha_v2"' in chunk
        assert "score=0.95" in chunk  # "0.950" satisfies this substring
        assert "embed_rank=" in chunk
        assert "llm" in chunk  # match_method

    def test_per_record_with_match_all(self, capsys):
        # match_all returns multiple results per left val
        def multi_match_llm(prompt):
            return json.dumps([
                {"index": 0, "score": 0.95, "reasoning": "first"},
                {"index": 1, "score": 0.85, "reasoning": "second"},
            ])
        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["alpha_v1", "alpha_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=multi_match_llm, embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            verbose=2,
            match_all=True,
            top_k=2,
        )
        captured = capsys.readouterr()
        row_lines = [line for line in captured.err.split("\n") if "Row:" in line]
        # match_all returns 2 results → 2 row lines
        assert len(row_lines) >= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_summary_prints_even_when_no_matches(self, capsys):
        # Threshold so high nothing matches
        def low_score_llm(prompt):
            return json.dumps([{"index": 0, "score": 0.1, "reasoning": "weak"}])

        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["beta"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=low_score_llm, embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
            llm_threshold=0.9,
        )
        captured = capsys.readouterr()
        assert "llm-join:" in captured.err

    def test_summary_includes_elapsed_seconds(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["alpha_v2"]})
        fuzzy_join(
            df1, df2,
            left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test",
            llm_concurrency=1,
        )
        captured = capsys.readouterr()
        # Should match "X.Xs total" pattern
        assert re.search(r"\d+\.\d+s total", captured.err)

    def test_stats_dont_persist_across_calls(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2"]})

        fuzzy_join(
            df1, df2, left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test", llm_concurrency=1,
        )
        first = capsys.readouterr().err

        fuzzy_join(
            df1, df2, left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test", llm_concurrency=1,
        )
        second = capsys.readouterr().err

        # Both runs should report 2 LLM calls, not accumulating
        m1 = re.search(r"(\d+) LLM", first)
        m2 = re.search(r"(\d+) LLM", second)
        assert int(m1.group(1)) == 2
        assert int(m2.group(1)) == 2

    def test_summary_goes_to_stderr_not_stdout(self, capsys):
        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["alpha_v2"]})
        fuzzy_join(
            df1, df2, left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test", llm_concurrency=1,
        )
        captured = capsys.readouterr()
        assert "llm-join:" in captured.err
        assert "llm-join:" not in captured.out

    def test_multiple_retries_count_each_rate_limit(self, capsys):
        # 3 rate-limited attempts before success
        attempts = [0]
        def flaky_llm(prompt):
            attempts[0] += 1
            if attempts[0] <= 3:
                raise RuntimeError("HTTP 429 too many requests")
            return json.dumps([{"index": 0, "score": 0.9, "reasoning": "ok"}])

        df1 = pd.DataFrame({"a": ["alpha"]})
        df2 = pd.DataFrame({"b": ["alpha_v2"]})
        fuzzy_join(
            df1, df2, left_on="a", right_on="b",
            llm=flaky_llm, embed_fn=mock_embed,
            context="test", llm_concurrency=1,
            max_retries=5,
        )
        captured = capsys.readouterr()
        m = re.search(r"(\d+) rate-limited", captured.err)
        # Each 429 throw should count
        assert int(m.group(1)) >= 3

    def test_failed_increments_only_after_exhausted_retries(self, capsys):
        def always_fail(prompt):
            raise RuntimeError("always fails")

        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2"]})
        fuzzy_join(
            df1, df2, left_on="a", right_on="b",
            llm=always_fail, embed_fn=mock_embed,
            context="test", llm_concurrency=1,
            max_retries=2,
        )
        captured = capsys.readouterr()
        m = re.search(r"(\d+) failed", captured.err)
        # Both left vals fail
        assert int(m.group(1)) == 2


# ---------------------------------------------------------------------------
# tqdm graceful fallback (simulate not installed)
# ---------------------------------------------------------------------------

class TestTqdmFallback:
    def test_works_without_tqdm_installed(self, capsys, monkeypatch):
        # Block tqdm import
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "tqdm", None)

        df1 = pd.DataFrame({"a": ["alpha", "beta"]})
        df2 = pd.DataFrame({"b": ["alpha_v2", "beta_v2"]})
        # Should not crash even with verbose=1
        fuzzy_join(
            df1, df2, left_on="a", right_on="b",
            llm=make_llm(), embed_fn=mock_embed,
            context="test", llm_concurrency=1,
            verbose=1,
        )
        captured = capsys.readouterr()
        assert "llm-join:" in captured.err
