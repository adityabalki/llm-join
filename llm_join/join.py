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
    embed_fn: Callable,
    context: str,
    column_context: Optional[dict] = None,
    top_k: int = 5,
    threshold: float = 0.7,
    how: str = "inner",
    batch_size: int = 32,
    embed_threshold: Optional[float] = None,
    max_llm_calls: Optional[int] = None,
    max_retries: int = 3,
    return_reasoning: bool = False,
) -> pd.DataFrame:
    # Normalise column names to single string (multi-col join concatenates values)
    left_col, right_col, df1, df2 = _normalise_cols(df1, df2, left_on, right_on)

    cfg = ColumnConfig(
        left_col=left_col,
        right_col=right_col,
        embed_fn=embed_fn,
        context=context,
        column_context=column_context or {},
        top_k=top_k,
        threshold=threshold,
        batch_size=batch_size,
        embed_threshold=embed_threshold,
        max_llm_calls=max_llm_calls,
        max_retries=max_retries,
    )

    retriever = EmbeddingRetriever(embed_fn=cfg.embed_fn)
    scorer = LLMScorer(llm, max_retries=cfg.max_retries)
    merger = Merger()

    left_vals = df1[left_col].astype(str).tolist()
    right_vals = df2[right_col].astype(str).tolist()

    # Use retrieve_with_scores for embed_threshold path
    candidates_per_row = retriever.retrieve_with_scores(left_vals, right_vals, top_k=cfg.top_k)

    matches: list[MatchResult] = []
    llm_call_count = 0

    for left_val, candidates_with_scores in zip(left_vals, candidates_per_row):
        if not candidates_with_scores:
            continue

        # embed_threshold short-circuit
        if cfg.embed_threshold is not None:
            best_candidate, best_score = candidates_with_scores[0]  # already sorted by score desc
            if best_score >= cfg.embed_threshold:
                matches.append(MatchResult(
                    left_val=left_val,
                    right_val=best_candidate,
                    score=best_score,
                    reasoning="skipped — embed score above threshold",
                    embed_rank=0,
                    match_method="embed_threshold",
                ))
                continue
            if best_score < (1.0 - cfg.embed_threshold):
                warnings.warn(
                    f"Row '{left_val}' skipped: top embed score {best_score:.3f} "
                    f"below non-match threshold {1.0 - cfg.embed_threshold:.3f}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

        # max_llm_calls cap
        if cfg.max_llm_calls is not None and llm_call_count >= cfg.max_llm_calls:
            warnings.warn(
                f"max_llm_calls={cfg.max_llm_calls} reached. "
                "Remaining rows skipped. Result is partial.",
                UserWarning,
                stacklevel=2,
            )
            break

        candidates = [c for c, _ in candidates_with_scores]
        results = scorer.score(left_val, candidates, cfg.context_str, threshold=cfg.threshold)
        llm_call_count += 1
        if results is None:
            # LLM failed all retries — fall back to highest-scoring embed candidate
            best_candidate, best_embed_score = candidates_with_scores[0]
            matches.append(MatchResult(
                left_val=left_val,
                right_val=best_candidate,
                score=best_embed_score,
                reasoning="LLM failed — embed rank-0 fallback used",
                embed_rank=0,
                match_method="embed_fallback",
                candidates=candidates,
            ))
        elif results:
            matches.extend(results)

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


