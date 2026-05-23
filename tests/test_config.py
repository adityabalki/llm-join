import numpy as np
import pytest
from llm_join.config import ColumnConfig


def dummy_embed(texts):
    return np.random.rand(len(texts), 32).astype("float32")


CTX = "test match context"


def test_defaults():
    cfg = ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context=CTX)
    assert cfg.top_k == 5
    assert cfg.threshold == 0.7
    assert cfg.context_str == CTX
    assert cfg.batch_size == 32
    assert cfg.embed_threshold is None
    assert cfg.max_llm_calls is None


def test_context_str_global_only():
    cfg = ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context="drug names")
    assert cfg.context_str == "drug names"


def test_context_str_column_context_only():
    cfg = ColumnConfig(
        left_col="drug", right_col="brand",
        embed_fn=dummy_embed,
        context="pharma matching",
        column_context={"drug": "generic name", "brand": "US brand"},
    )
    assert "generic name" in cfg.context_str
    assert "US brand" in cfg.context_str


def test_context_str_both():
    cfg = ColumnConfig(
        left_col="drug", right_col="brand",
        embed_fn=dummy_embed,
        context="pharma",
        column_context={"drug": "generic"},
    )
    assert "pharma" in cfg.context_str
    assert "generic" in cfg.context_str


def test_invalid_threshold():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context=CTX, threshold=1.5)


def test_invalid_top_k():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context=CTX, top_k=0)


def test_invalid_embed_threshold():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context=CTX, embed_threshold=1.5)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context=CTX, batch_size=0)


def test_invalid_max_llm_calls():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context=CTX, max_llm_calls=0)


def test_empty_left_col():
    with pytest.raises(ValueError):
        ColumnConfig(left_col="", right_col="b", embed_fn=dummy_embed, context=CTX)


def test_empty_context_raises():
    with pytest.raises(ValueError, match="context must not be empty"):
        ColumnConfig(left_col="a", right_col="b", embed_fn=dummy_embed, context="   ")
