"""Tests: retrieval default is 'embedding', not 'hybrid'."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from llm_join.config import ColumnConfig
from llm_join.join import fuzzy_join


def _dummy_embed(texts):
    rng = np.random.default_rng(0)
    return rng.random((len(texts), 4)).astype("float32")


def _dummy_llm(prompt):
    return '[{"right_val": "b", "score": 0.9, "reasoning": "match"}]'


def test_config_default_retrieval_is_embedding():
    cfg = ColumnConfig(
        left_col="l",
        right_col="r",
        embed_fn=_dummy_embed,
        context="test context",
    )
    assert cfg.retrieval == "embedding"


def test_fuzzy_join_default_uses_embedding_retriever():
    df1 = pd.DataFrame({"l": ["a"]})
    df2 = pd.DataFrame({"r": ["b"]})

    with patch("llm_join.join.EmbeddingRetriever", wraps=None) as mock_embed:
        # Configure the mock so fuzzy_join can actually run
        mock_instance = MagicMock()
        mock_instance.retrieve_with_scores.return_value = [[("b", 0.9)]]
        mock_embed.return_value = mock_instance
        fuzzy_join(
            df1, df2,
            left_on="l", right_on="r",
            llm_fn=_dummy_llm,
            embed_fn=_dummy_embed,
            context="test context",
            llm_concurrency=1,
            embed_concurrency=1,
        )
        mock_embed.assert_called_once()
