import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    match_all: bool = False,
    llm_concurrency: int = 1,
) -> pd.DataFrame:
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
        match_all=match_all,
    )

    retriever = EmbeddingRetriever(embed_fn=cfg.embed_fn)
    scorer = LLMScorer(llm, max_retries=cfg.max_retries)
    merger = Merger()

    left_vals = df1[left_col].astype(str).tolist()
    right_vals = df2[right_col].astype(str).tolist()

    candidates_per_row = retriever.retrieve_with_scores(left_vals, right_vals, top_k=cfg.top_k)

    # --- Pass 1: embed_threshold short-circuit, collect rows needing LLM ---
    embed_matches: list[MatchResult] = []
    llm_queue: list[tuple] = []  # (left_val, candidates_with_scores)

    for left_val, candidates_with_scores in zip(left_vals, candidates_per_row):
        if not candidates_with_scores:
            continue

        if cfg.embed_threshold is not None:
            best_candidate, best_score = candidates_with_scores[0]
            if best_score >= cfg.embed_threshold:
                embed_matches.append(MatchResult(
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

        llm_queue.append((left_val, candidates_with_scores))

    # --- Apply max_llm_calls cap ---
    if cfg.max_llm_calls is not None and len(llm_queue) > cfg.max_llm_calls:
        warnings.warn(
            f"max_llm_calls={cfg.max_llm_calls} reached. "
            f"{len(llm_queue) - cfg.max_llm_calls} rows skipped. Result is partial.",
            UserWarning,
            stacklevel=2,
        )
        llm_queue = llm_queue[:cfg.max_llm_calls]

    # --- Pass 2: LLM scoring (sequential or concurrent) ---
    llm_matches = _score_rows(scorer, llm_queue, cfg, llm_concurrency)

    matches = embed_matches + llm_matches
    result = merger.merge(df1, df2, left_col, right_col, matches, how=how, return_reasoning=return_reasoning)
    # Drop temp composite-key columns injected by _normalise_cols
    return result.drop(columns=["__left_key__", "__right_key__"], errors="ignore")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _make_debug(candidates_with_scores: list) -> list:
    return [{"candidate": c, "embed_score": round(s, 4)} for c, s in candidates_with_scores]


def _process_results(
    results,
    left_val: str,
    candidates_with_scores: list,
    candidates_debug: list,
) -> list[MatchResult]:
    """Convert scorer output to MatchResult list, applying embed fallback on None."""
    if results is None:
        # LLM failed all retries — fall back to highest-scoring embed candidate
        best_candidate, best_embed_score = candidates_with_scores[0]
        return [MatchResult(
            left_val=left_val,
            right_val=best_candidate,
            score=best_embed_score,
            reasoning="LLM failed — embed rank-0 fallback used",
            embed_rank=0,
            match_method="embed_fallback",
            candidates=candidates_debug,
        )]
    if results:
        for r in results:
            r.candidates = candidates_debug
        return list(results)
    return []


def _score_rows(
    scorer: LLMScorer,
    llm_queue: list,
    cfg: ColumnConfig,
    llm_concurrency: int,
) -> list[MatchResult]:
    if not llm_queue:
        return []
    if llm_concurrency <= 1:
        return _score_sequential(scorer, llm_queue, cfg)
    if scorer._is_async:
        return _score_async(scorer, llm_queue, cfg, llm_concurrency)
    return _score_threaded(scorer, llm_queue, cfg, llm_concurrency)


def _score_sequential(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig) -> list[MatchResult]:
    matches = []
    for left_val, candidates_with_scores in llm_queue:
        candidates = [c for c, _ in candidates_with_scores]
        candidates_debug = _make_debug(candidates_with_scores)
        results = scorer.score(
            left_val, candidates, cfg.context_str,
            threshold=cfg.threshold, match_all=cfg.match_all,
        )
        matches.extend(_process_results(results, left_val, candidates_with_scores, candidates_debug))
    return matches


def _score_threaded(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, llm_concurrency: int) -> list[MatchResult]:
    """Parallel scoring for sync LLM functions using ThreadPoolExecutor."""
    result_map: dict[int, list[MatchResult]] = {}

    def _score_one(idx: int, left_val: str, candidates_with_scores: list):
        candidates = [c for c, _ in candidates_with_scores]
        results = scorer.score(
            left_val, candidates, cfg.context_str,
            threshold=cfg.threshold, match_all=cfg.match_all,
        )
        return idx, results, left_val, candidates_with_scores

    with ThreadPoolExecutor(max_workers=llm_concurrency) as executor:
        futures = {
            executor.submit(_score_one, i, lv, cws): i
            for i, (lv, cws) in enumerate(llm_queue)
        }
        for future in as_completed(futures):
            idx, results, left_val, candidates_with_scores = future.result()
            candidates_debug = _make_debug(candidates_with_scores)
            result_map[idx] = _process_results(results, left_val, candidates_with_scores, candidates_debug)

    # Preserve original row order
    matches = []
    for i in range(len(llm_queue)):
        matches.extend(result_map.get(i, []))
    return matches


def _score_async(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, llm_concurrency: int) -> list[MatchResult]:
    """Parallel scoring for async LLM functions using asyncio + semaphore."""

    async def _run_all():
        sem = asyncio.Semaphore(llm_concurrency)

        async def _score_one(left_val: str, candidates_with_scores: list):
            async with sem:
                candidates = [c for c, _ in candidates_with_scores]
                results = await scorer.score_async(
                    left_val, candidates, cfg.context_str,
                    threshold=cfg.threshold, match_all=cfg.match_all,
                )
                return results, left_val, candidates_with_scores

        tasks = [_score_one(lv, cws) for lv, cws in llm_queue]
        return await asyncio.gather(*tasks)

    # Handle Jupyter/Databricks where an event loop is already running
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Run coroutine in a separate thread that owns its own event loop
        with ThreadPoolExecutor(max_workers=1) as pool:
            raw = pool.submit(asyncio.run, _run_all()).result()
    else:
        raw = asyncio.run(_run_all())

    matches = []
    for results, left_val, candidates_with_scores in raw:
        candidates_debug = _make_debug(candidates_with_scores)
        matches.extend(_process_results(results, left_val, candidates_with_scores, candidates_debug))
    return matches


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

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
