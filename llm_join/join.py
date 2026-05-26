import asyncio
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Union
import pandas as pd

from llm_join.config import ColumnConfig
from llm_join.retriever import EmbeddingRetriever
from llm_join.scorer import LLMScorer
from llm_join.merger import Merger, MatchResult
from llm_join.stats import JoinStats


def _maybe_tqdm(iterable, total: int, desc: str, verbose: int):
    """Wrap iterable in tqdm if verbose >= 1 and tqdm installed; else passthrough."""
    if verbose < 1:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, file=sys.stderr)
    except ImportError:
        print(f"  [{desc}] starting ({total} items)", file=sys.stderr, flush=True)
        return iterable


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
    llm_threshold: float = 0.7,
    how: str = "inner",
    batch_size: int = 32,
    embed_concurrency: int = 10,
    embed_skip_threshold: float = 1.0,
    max_llm_calls: Optional[int] = None,
    max_retries: int = 3,
    return_reasoning: bool = False,
    match_all: bool = False,
    verbose: int = 0,
    llm_concurrency: int,
) -> pd.DataFrame:
    """Fuzzy join two DataFrames using embeddings + LLM scoring.

    verbose:
      0 = silent (default; one-line summary at end always prints)
      1 = progress bars (tqdm if installed) + per-batch failure detail
      2 = adds per-record log line: '"left" -> "right" [score=..., embed_rank=..., method]'
    """
    t0 = time.time()
    stats = JoinStats()

    left_col, right_col, df1, df2 = _normalise_cols(df1, df2, left_on, right_on)

    cfg = ColumnConfig(
        left_col=left_col,
        right_col=right_col,
        embed_fn=embed_fn,
        context=context,
        column_context=column_context or {},
        top_k=top_k,
        llm_threshold=llm_threshold,
        batch_size=batch_size,
        embed_concurrency=embed_concurrency,
        embed_skip_threshold=embed_skip_threshold,
        max_llm_calls=max_llm_calls,
        max_retries=max_retries,
        match_all=match_all,
    )

    retriever = EmbeddingRetriever(
        embed_fn=cfg.embed_fn,
        batch_size=cfg.batch_size,
        embed_concurrency=cfg.embed_concurrency,
        verbose=verbose,
    )
    scorer = LLMScorer(llm, max_retries=cfg.max_retries, stats=stats, verbose=verbose)
    merger = Merger()

    # Deduplicate left values — LLM called once per unique value, pandas merge fans out.
    _seen: set = set()
    left_vals: list[str] = []
    for v in df1[left_col].astype(str):
        if v not in _seen:
            _seen.add(v)
            left_vals.append(v)
    stats.n_left_total = len(df1)
    stats.n_left_unique = len(left_vals)
    if stats.n_left_unique < stats.n_left_total:
        warnings.warn(
            f"llm-join: deduplicated {stats.n_left_total} left rows → {stats.n_left_unique} unique values. "
            f"LLM called {stats.n_left_unique} times (not {stats.n_left_total}). Results fanned out via merge.",
            UserWarning,
            stacklevel=2,
        )

    # Deduplicate right values — silent compute optimization (FAISS sees unique candidates).
    # Pandas merge fans results back to all matching df2 rows.
    _r_seen: set = set()
    right_vals: list[str] = []
    for v in df2[right_col].astype(str):
        if v not in _r_seen:
            _r_seen.add(v)
            right_vals.append(v)
    stats.n_right_total = len(df2)
    stats.n_right_unique = len(right_vals)

    # Warn on large scale
    if stats.n_left_unique > 5_000 and cfg.embed_skip_threshold >= 1.0 and cfg.max_llm_calls is None:
        warnings.warn(
            f"llm-join: {stats.n_left_unique} unique left values → up to {stats.n_left_unique} LLM calls. "
            f"For large datasets, consider embed_skip_threshold (skip LLM for high-confidence matches) "
            f"or max_llm_calls to cap cost.",
            UserWarning,
            stacklevel=2,
        )

    candidates_per_row = retriever.retrieve_with_scores(left_vals, right_vals, top_k=cfg.top_k)

    # --- Pass 1: embed_threshold short-circuit, collect rows needing LLM ---
    embed_matches: list[MatchResult] = []
    llm_queue: list[tuple] = []  # (left_val, candidates_with_scores)

    for left_val, candidates_with_scores in zip(left_vals, candidates_per_row):
        if not candidates_with_scores:
            continue

        best_candidate, best_score = candidates_with_scores[0]
        if best_score >= cfg.embed_skip_threshold:
            embed_matches.append(MatchResult(
                left_val=left_val,
                right_val=best_candidate,
                score=best_score,
                reasoning=f"skipped LLM — embed similarity {best_score:.4f} >= embed_skip_threshold {cfg.embed_skip_threshold}",
                embed_rank=0,
                match_method="embed_skip",
            ))
            stats.inc("n_embed_skipped")
            if verbose >= 2:
                print(
                    f'  Row: "{left_val}" -> "{best_candidate}" '
                    f'[score={best_score:.3f}, embed_rank=0, embed_skip]',
                    file=sys.stderr, flush=True,
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
    llm_matches = _score_rows(scorer, llm_queue, cfg, llm_concurrency, verbose)

    matches = embed_matches + llm_matches
    result = merger.merge(df1, df2, left_col, right_col, matches, how=how, return_reasoning=return_reasoning)
    result = result.drop(columns=["__left_key__", "__right_key__"], errors="ignore")

    # Always print one-line summary
    stats.elapsed_seconds = time.time() - t0
    print(stats.summary_line(), file=sys.stderr, flush=True)

    return result


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
    verbose: int = 0,
) -> list[MatchResult]:
    if not llm_queue:
        return []
    if llm_concurrency <= 1:
        return _score_sequential(scorer, llm_queue, cfg, verbose)
    if scorer._is_async:
        return _score_async(scorer, llm_queue, cfg, llm_concurrency, verbose)
    return _score_threaded(scorer, llm_queue, cfg, llm_concurrency, verbose)


def _score_sequential(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, verbose: int = 0) -> list[MatchResult]:
    matches = []
    iterator = _maybe_tqdm(llm_queue, len(llm_queue), "LLM", verbose)
    for left_val, candidates_with_scores in iterator:
        candidates = [c for c, _ in candidates_with_scores]
        candidates_debug = _make_debug(candidates_with_scores)
        results = scorer.score(
            left_val, candidates, cfg.context_str,
            llm_threshold=cfg.llm_threshold, match_all=cfg.match_all,
        )
        matches.extend(_process_results(results, left_val, candidates_with_scores, candidates_debug))
    return matches


def _score_threaded(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, llm_concurrency: int, verbose: int = 0) -> list[MatchResult]:
    """Parallel scoring for sync LLM functions using ThreadPoolExecutor."""
    result_map: dict[int, list[MatchResult]] = {}

    def _score_one(idx: int, left_val: str, candidates_with_scores: list):
        candidates = [c for c, _ in candidates_with_scores]
        results = scorer.score(
            left_val, candidates, cfg.context_str,
            llm_threshold=cfg.llm_threshold, match_all=cfg.match_all,
        )
        return idx, results, left_val, candidates_with_scores

    with ThreadPoolExecutor(max_workers=llm_concurrency) as executor:
        futures = {
            executor.submit(_score_one, i, lv, cws): i
            for i, (lv, cws) in enumerate(llm_queue)
        }
        for future in _maybe_tqdm(as_completed(futures), len(futures), "LLM", verbose):
            idx, results, left_val, candidates_with_scores = future.result()
            candidates_debug = _make_debug(candidates_with_scores)
            result_map[idx] = _process_results(results, left_val, candidates_with_scores, candidates_debug)

    matches = []
    for i in range(len(llm_queue)):
        matches.extend(result_map.get(i, []))
    return matches


def _score_async(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, llm_concurrency: int, verbose: int = 0) -> list[MatchResult]:
    """Parallel scoring for async LLM functions using asyncio + semaphore."""
    n = len(llm_queue)

    async def _run_all():
        sem = asyncio.Semaphore(llm_concurrency)
        pbar = None
        if verbose >= 1:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n, desc="LLM", file=sys.stderr)
            except ImportError:
                print(f"  [LLM] starting ({n} calls)", file=sys.stderr, flush=True)

        async def _score_one(left_val: str, candidates_with_scores: list):
            async with sem:
                candidates = [c for c, _ in candidates_with_scores]
                results = await scorer.score_async(
                    left_val, candidates, cfg.context_str,
                    llm_threshold=cfg.llm_threshold, match_all=cfg.match_all,
                )
                if pbar is not None:
                    pbar.update(1)
                return results, left_val, candidates_with_scores

        tasks = [_score_one(lv, cws) for lv, cws in llm_queue]
        out = await asyncio.gather(*tasks)
        if pbar is not None:
            pbar.close()
        return out

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
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
