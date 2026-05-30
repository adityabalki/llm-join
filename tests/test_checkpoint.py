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
