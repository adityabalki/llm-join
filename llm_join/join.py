import warnings
from typing import Callable, Optional, Union
import pandas as pd

from llm_join.config import ColumnConfig
from llm_join.retriever import EmbeddingRetriever
from llm_join.scorer import LLMScorer
from llm_join.merger import Merger, MatchResult


def fuzzy_join(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    left_on: Union[str, list[str]],
    right_on: Union[str, list[str]],
    llm: Callable,
    context: str = "",
    column_context: Optional[dict] = None,
    top_k: int = 5,
    threshold: float = 0.7,
    how: str = "inner",
    embed_model: str = "all-MiniLM-L6-v2",
    embed_fn: Optional[Callable] = None,
    batch_size: int = 32,
    embed_threshold: Optional[float] = None,
    max_llm_calls: Optional[int] = None,
    return_reasoning: bool = False,
) -> pd.DataFrame:
    # Normalise column names to single string (multi-col join concatenates values)
    left_col, right_col, df1, df2 = _normalise_cols(df1, df2, left_on, right_on)

    cfg = ColumnConfig(
        left_col=left_col,
        right_col=right_col,
        context=context,
        column_context=column_context or {},
        top_k=top_k,
        threshold=threshold,
        embed_model=embed_model,
        embed_fn=embed_fn,
        batch_size=batch_size,
        embed_threshold=embed_threshold,
        max_llm_calls=max_llm_calls,
    )

    retriever = EmbeddingRetriever(embed_model=cfg.embed_model, embed_fn=cfg.embed_fn)
    scorer = LLMScorer(llm)
    merger = Merger()

    left_vals = df1[left_col].astype(str).tolist()
    right_vals = df2[right_col].astype(str).tolist()

    candidates_per_row = retriever.retrieve(left_vals, right_vals, top_k=cfg.top_k)

    matches: list[MatchResult] = []
    llm_call_count = 0

    for left_val, candidates in zip(left_vals, candidates_per_row):
        if not candidates:
            continue

        # embed_threshold short-circuit: skip LLM for obvious cases
        if cfg.embed_threshold is not None:
            top_embed_score = _top_embed_score(retriever, left_val, candidates)
            if top_embed_score >= cfg.embed_threshold:
                matches.append(MatchResult(
                    left_val=left_val,
                    right_val=candidates[0],
                    score=top_embed_score,
                    reasoning="embed_threshold match",
                    embed_rank=0,
                ))
                continue
            if top_embed_score < (1.0 - cfg.embed_threshold):
                continue  # obvious non-match, skip LLM

        # max_llm_calls cap
        if cfg.max_llm_calls is not None and llm_call_count >= cfg.max_llm_calls:
            warnings.warn(
                f"max_llm_calls={cfg.max_llm_calls} reached. "
                "Remaining rows skipped. Result is partial.",
                UserWarning,
                stacklevel=2,
            )
            break

        result = scorer.score(left_val, candidates, cfg.context_str, threshold=cfg.threshold)
        llm_call_count += 1
        if result is not None:
            matches.append(result)

    return merger.merge(df1, df2, left_col, right_col, matches, how=how, return_reasoning=return_reasoning)


def _normalise_cols(df1, df2, left_on, right_on):
    if isinstance(left_on, list):
        col = "__left_key__"
        df1 = df1.copy()
        df1[col] = df1[left_on].astype(str).agg(" ".join, axis=1)
        left_on = col
    if isinstance(right_on, list):
        col = "__right_key__"
        df2 = df2.copy()
        df2[col] = df2[right_on].astype(str).agg(" ".join, axis=1)
        right_on = col
    return left_on, right_on, df1, df2


def _top_embed_score(retriever: EmbeddingRetriever, query: str, candidates: list[str]) -> float:
    import faiss
    import numpy as np
    q_vec = retriever._embed([query])
    c_vecs = retriever._embed(candidates)
    faiss.normalize_L2(q_vec)
    faiss.normalize_L2(c_vecs)
    scores = (c_vecs @ q_vec.T).flatten()
    return float(scores.max())
