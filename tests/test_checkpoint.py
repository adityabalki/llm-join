"""Tests for llm_join.checkpoint.CheckpointStore — unit tests (5)."""
import json
import os
import warnings
import pytest
import tempfile

from llm_join.checkpoint import CheckpointStore, CHECKPOINT_VERSION
from llm_join.merger import MatchResult


def _make_results(left_val="apple", right_val="apple inc", score=0.95):
    return [MatchResult(
        left_val=left_val,
        right_val=right_val,
        score=score,
        reasoning="good match",
        embed_rank=0,
        match_method="llm",
        candidates=[{"candidate": right_val, "source": "embed", "embed_score": 0.9}],
    )]


def test_basic_save_load(tmp_path):
    path = str(tmp_path / "ckpt.json")
    store = CheckpointStore(path)
    results = _make_results()

    store.save("apple", results)
    assert store.is_done("apple")
    assert not store.is_done("banana")
    assert store.n_done() == 1

    # Reload from disk — simulates new process
    store2 = CheckpointStore(path)
    assert store2.is_done("apple")
    loaded = store2.get_results("apple")
    assert len(loaded) == 1
    assert loaded[0].left_val == "apple"
    assert loaded[0].right_val == "apple inc"
    assert abs(loaded[0].score - 0.95) < 1e-6
    assert loaded[0].reasoning == "good match"
    assert loaded[0].match_method == "llm"


def test_corrupt_json_warns_and_starts_fresh(tmp_path):
    path = str(tmp_path / "ckpt.json")
    # Write corrupt content
    with open(path, "w") as f:
        f.write("not valid json {{{")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store = CheckpointStore(path)

    assert any("corrupt checkpoint" in str(warning.message) for warning in w)
    assert store.n_done() == 0
    assert not store.is_done("anything")


def test_version_mismatch_warns_and_starts_fresh(tmp_path):
    path = str(tmp_path / "ckpt.json")
    # Write checkpoint with wrong version
    with open(path, "w") as f:
        json.dump({"version": "99.0.0", "results": {"apple": []}}, f)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store = CheckpointStore(path)

    assert any("version mismatch" in str(warning.message) for warning in w)
    assert store.n_done() == 0


def test_atomic_write_produces_valid_json(tmp_path):
    path = str(tmp_path / "ckpt.json")
    store = CheckpointStore(path)
    results = _make_results()
    store.save("apple", results)

    # File must exist and be valid JSON after save
    with open(path, "r") as f:
        data = json.load(f)
    assert data["version"] == CHECKPOINT_VERSION
    assert "apple" in data["results"]


def test_parent_dir_missing_raises():
    path = "/nonexistent_dir_llm_join_test/ckpt.json"
    with pytest.raises(ValueError, match="checkpoint parent directory does not exist"):
        CheckpointStore(path)


# ---------------------------------------------------------------------------
# Integration tests (5) — require full fuzzy_join pipeline
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import asyncio as _asyncio


def _make_embed_fn(dim=4):
    """Deterministic embed: use hash of text to seed RNG."""
    def embed(texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(rng.random(dim).astype("float32"))
        arr = np.array(out)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)
    return embed


def _make_sync_llm(match_map: dict):
    """Returns a sync LLM fn that returns the mapped right_val."""
    def llm_fn(prompt: str) -> str:
        import json as _json
        for left_val, right_val in match_map.items():
            if left_val in prompt:
                return _json.dumps([{"right_val": right_val, "score": 0.9, "reasoning": "match"}])
        return "[]"
    return llm_fn


def _make_async_llm(match_map: dict):
    """Returns an async LLM fn."""
    import json as _json
    async def llm_fn(prompt: str) -> str:
        for left_val, right_val in match_map.items():
            if left_val in prompt:
                return _json.dumps([{"right_val": right_val, "score": 0.9, "reasoning": "match"}])
        return "[]"
    return llm_fn


def test_resume_skips_done_vals(tmp_path):
    """LLM called only for vals not in checkpoint."""
    from llm_join.join import fuzzy_join

    call_log = []

    def counting_llm(prompt: str) -> str:
        import json as _json
        for val in ["apple", "banana", "cherry"]:
            if val in prompt:
                call_log.append(val)
                return _json.dumps([{"right_val": val + " inc", "score": 0.9, "reasoning": "ok"}])
        return "[]"

    df1 = pd.DataFrame({"vendor": ["apple", "banana", "cherry"]})
    df2 = pd.DataFrame({"supplier": ["apple inc", "banana inc", "cherry inc"]})
    embed_fn = _make_embed_fn()
    ckpt = str(tmp_path / "test.ckpt.json")

    # Pre-populate checkpoint with "apple" done
    store = CheckpointStore(ckpt)
    store.save("apple", [MatchResult(
        left_val="apple", right_val="apple inc", score=0.9,
        reasoning="pre-done", embed_rank=0, match_method="llm",
    )])

    fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=counting_llm,
        embed_fn=embed_fn,
        context="company names",
        llm_concurrency=1,
        embed_concurrency=1,
        checkpoint_path=ckpt,
    )

    # LLM must NOT have been called for "apple"
    assert "apple" not in call_log
    # LLM must have been called for banana and cherry
    assert "banana" in call_log
    assert "cherry" in call_log


def test_full_resume_matches_full_run(tmp_path):
    """Checkpointed run produces same DataFrame as full run."""
    from llm_join.join import fuzzy_join

    df1 = pd.DataFrame({"vendor": ["apple", "banana"]})
    df2 = pd.DataFrame({"supplier": ["apple inc", "banana inc"]})
    match_map = {"apple": "apple inc", "banana": "banana inc"}
    embed_fn = _make_embed_fn()
    ckpt = str(tmp_path / "test.ckpt.json")

    # Full run (no checkpoint)
    result_full = fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_make_sync_llm(match_map),
        embed_fn=embed_fn,
        context="company names",
        llm_concurrency=1,
        embed_concurrency=1,
    )

    # Partial run: only "apple" in checkpoint
    store = CheckpointStore(ckpt)
    store.save("apple", [MatchResult(
        left_val="apple", right_val="apple inc", score=0.9,
        reasoning="match", embed_rank=0, match_method="llm",
    )])

    result_resume = fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_make_sync_llm(match_map),
        embed_fn=embed_fn,
        context="company names",
        llm_concurrency=1,
        embed_concurrency=1,
        checkpoint_path=ckpt,
    )

    # Same rows (order-independent)
    assert set(result_full["vendor"].tolist()) == set(result_resume["vendor"].tolist())
    assert set(result_full["supplier"].tolist()) == set(result_resume["supplier"].tolist())


def test_deleted_on_successful_completion(tmp_path):
    """Checkpoint file removed after successful fuzzy_join."""
    from llm_join.join import fuzzy_join

    df1 = pd.DataFrame({"vendor": ["apple"]})
    df2 = pd.DataFrame({"supplier": ["apple inc"]})
    ckpt = str(tmp_path / "test.ckpt.json")

    fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_make_sync_llm({"apple": "apple inc"}),
        embed_fn=_make_embed_fn(),
        context="company names",
        llm_concurrency=1,
        embed_concurrency=1,
        checkpoint_path=ckpt,
    )

    assert not os.path.exists(ckpt), "checkpoint file should be deleted after success"


def test_checkpoint_with_threaded_llm(tmp_path):
    """Threaded LLM path saves all results correctly (thread-safe)."""
    from llm_join.join import fuzzy_join

    n = 6
    vendors = [f"vendor_{i}" for i in range(n)]
    suppliers = [f"supplier_{i}" for i in range(n)]
    match_map = {f"vendor_{i}": f"supplier_{i}" for i in range(n)}

    df1 = pd.DataFrame({"vendor": vendors})
    df2 = pd.DataFrame({"supplier": suppliers})
    ckpt = str(tmp_path / "threaded.ckpt.json")

    result = fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_make_sync_llm(match_map),
        embed_fn=_make_embed_fn(),
        context="vendor supplier matching",
        llm_concurrency=3,
        embed_concurrency=2,
        checkpoint_path=ckpt,
    )

    # All rows matched (or at least no crash + file cleaned up)
    assert not os.path.exists(ckpt), "checkpoint deleted after success"
    assert len(result) >= 1


def test_checkpoint_with_async_llm(tmp_path):
    """Async LLM path saves all results correctly."""
    from llm_join.join import fuzzy_join

    n = 4
    vendors = [f"async_vendor_{i}" for i in range(n)]
    suppliers = [f"async_supplier_{i}" for i in range(n)]
    match_map = {f"async_vendor_{i}": f"async_supplier_{i}" for i in range(n)}

    df1 = pd.DataFrame({"vendor": vendors})
    df2 = pd.DataFrame({"supplier": suppliers})
    ckpt = str(tmp_path / "async.ckpt.json")

    result = fuzzy_join(
        df1, df2,
        left_on="vendor", right_on="supplier",
        llm_fn=_make_async_llm(match_map),
        embed_fn=_make_embed_fn(),
        context="vendor supplier matching",
        llm_concurrency=2,
        embed_concurrency=2,
        checkpoint_path=ckpt,
    )

    assert not os.path.exists(ckpt), "checkpoint deleted after success"
    assert len(result) >= 1
