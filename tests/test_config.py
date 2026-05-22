import pytest
from llm_join.config import ColumnConfig

def test_defaults():
    cfg = ColumnConfig(left_col="a", right_col="b")
    assert cfg.top_k == 5
    assert cfg.threshold == 0.7
    assert cfg.context_str == ""
    assert cfg.embed_model == "all-MiniLM-L6-v2"
    assert cfg.batch_size == 32
    assert cfg.embed_threshold is None
    assert cfg.max_llm_calls is None

def test_context_str_global_only():
    cfg = ColumnConfig(left_col="a", right_col="b", context="drug names")
    assert cfg.context_str == "drug names"

def test_context_str_column_context_only():
    cfg = ColumnConfig(
        left_col="drug", right_col="brand",
        column_context={"drug": "generic name", "brand": "US brand"},
    )
    assert "generic name" in cfg.context_str
    assert "US brand" in cfg.context_str

def test_context_str_both():
    cfg = ColumnConfig(
        left_col="drug", right_col="brand",
        context="pharma",
        column_context={"drug": "generic"},
    )
    assert "pharma" in cfg.context_str
    assert "generic" in cfg.context_str

def test_invalid_threshold():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="a", right_col="b", threshold=1.5)

def test_invalid_top_k():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="a", right_col="b", top_k=0)
