import asyncio
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Union
import pandas as pd

from llm_join.config import ColumnConfig
from llm_join.retriever import EmbeddingRetriever
from llm_join.bm25 import BM25Retriever
from llm_join.hybrid import HybridRetriever
from llm_join.scorer import LLMScorer
from llm_join.merger import Merger, MatchResult
from llm_join.stats import JoinStats


def _maybe_tqdm(iterable, total: int, desc: str, verbose: int):
    """Wrap iterable in tqdm if verbose >= 1; else passthrough."""
    if verbose < 1:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc, file=sys.stderr)


def fuzzy_join(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    left_on: Union[str, list[str]],
    right_on: Union[str, list[str]],
    llm_fn: Callable,
    embed_fn: Callable,
    context: str,
    column_context: Optional[dict] = None,
    top_k: int = 5,
    llm_threshold: float = 0.7,
    how: str = "inner",
    batch_size: int = 32,
    embed_skip_threshold: float = 1.0,
    max_llm_calls: Optional[int] = None,
    max_retries: int = 3,
    return_reasoning: bool = False,
    match_all: bool = False,
    verbose: int = 0,
    retrieval: str = "embedding",
    bm25_stopwords: Optional[object] = None,
    llm_concurrency: int,
    embed_concurrency: int,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fuzzy join two DataFrames using embeddings + LLM scoring.

    verbose:
      0 = silent (default; one-line summary at end always prints)
      1 = progress bars (tqdm) + per-batch failure detail
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
        retrieval=retrieval,
        bm25_stopwords=bm25_stopwords,
    )

    embed_retriever = EmbeddingRetriever(
        embed_fn=cfg.embed_fn,
        batch_size=cfg.batch_size,
        embed_concurrency=cfg.embed_concurrency,
        verbose=verbose,
    )
    bm25_retriever = BM25Retriever(stopwords=cfg.bm25_stopwords)

    if cfg.retrieval == "embedding":
        retriever = embed_retriever
    elif cfg.retrieval == "bm25":
        retriever = bm25_retriever
    else:  # hybrid
        retriever = HybridRetriever(embed_retriever, bm25_retriever)
    scorer = LLMScorer(llm_fn, max_retries=cfg.max_retries, stats=stats, verbose=verbose)
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

    store = None
    checkpoint_matches: list = []
    pending_left_vals = left_vals
    if checkpoint_path is not None:
        from llm_join.checkpoint import CheckpointStore
        store = CheckpointStore(checkpoint_path)
        pending_left_vals = []
        for v in left_vals:
            if store.is_done(v):
                checkpoint_matches.extend(store.get_results(v))
            else:
                pending_left_vals.append(v)
        n_checkpointed = len(left_vals) - len(pending_left_vals)
        if n_checkpointed > 0:
            print(
                f"llm-join: resuming from checkpoint — {n_checkpointed} already done, "
                f"{len(pending_left_vals)} remaining",
                file=sys.stderr, flush=True,
            )

    raw_candidates = retriever.retrieve_with_scores(pending_left_vals, right_vals, top_k=cfg.top_k)
    candidates_per_row = [_normalize_candidates(row, cfg.retrieval) for row in raw_candidates]

    # --- Pass 1: embed_threshold short-circuit, collect rows needing LLM ---
    embed_matches: list[MatchResult] = []
    llm_queue: list[tuple] = []  # (left_val, candidates_with_scores)

    for left_val, candidates_with_scores in zip(pending_left_vals, candidates_per_row):
        if not candidates_with_scores:
            continue

        top = candidates_with_scores[0]
        best_candidate = top["candidate"]
        # embed_skip only fires on real embedding signal — bm25-only candidates can't trigger it
        best_score = top["embed_score"] if top["embed_score"] is not None else 0.0
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
    llm_matches = _score_rows(scorer, llm_queue, cfg, llm_concurrency, verbose, store=store)

    matches = checkpoint_matches + embed_matches + llm_matches
    result = merger.merge(df1, df2, left_col, right_col, matches, how=how, return_reasoning=return_reasoning)
    result = result.drop(columns=["__left_key__", "__right_key__"], errors="ignore")

    if store is not None:
        store.delete()

    # Always print one-line summary
    stats.elapsed_seconds = time.time() - t0
    print(stats.summary_line(), file=sys.stderr, flush=True)

    return result


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize_candidates(row, retrieval_mode: str) -> list[dict]:
    """Convert retriever output to uniform dict format.

    EmbeddingRetriever and BM25Retriever return list[tuple[str, float]].
    HybridRetriever returns list[dict] already enriched.
    """
    out: list[dict] = []
    for item in row:
        if isinstance(item, dict):
            out.append(item)
        else:
            cand, score = item
            if retrieval_mode == "embedding":
                out.append({
                    "candidate": cand,
                    "embed_score": float(score),
                    "bm25_score": None,
                    "embed_rank": len(out),
                    "bm25_rank": None,
                    "rrf_score": None,
                    "source": "embed",
                })
            else:  # bm25
                out.append({
                    "candidate": cand,
                    "embed_score": None,
                    "bm25_score": float(score),
                    "embed_rank": None,
                    "bm25_rank": len(out),
                    "rrf_score": None,
                    "source": "bm25",
                })
    return out


def _make_debug(candidates_with_scores: list) -> list:
    out = []
    for d in candidates_with_scores:
        entry = {"candidate": d["candidate"], "source": d["source"]}
        if d.get("embed_score") is not None:
            entry["embed_score"] = round(d["embed_score"], 4)
        if d.get("bm25_score") is not None:
            entry["bm25_score"] = round(d["bm25_score"], 4)
        if d.get("rrf_score") is not None:
            entry["rrf_score"] = round(d["rrf_score"], 4)
        out.append(entry)
    return out


def _process_results(
    results,
    left_val: str,
    candidates_with_scores: list,
    candidates_debug: list,
) -> list[MatchResult]:
    """Convert scorer output to MatchResult list, applying embed fallback on None."""
    if results is None:
        top = candidates_with_scores[0]
        best_candidate = top["candidate"]
        # Fallback score: embed_score if available, otherwise bm25, otherwise 0
        best_embed_score = top.get("embed_score") or top.get("bm25_score") or 0.0
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
    store=None,
) -> list[MatchResult]:
    if not llm_queue:
        return []
    if llm_concurrency <= 1:
        return _score_sequential(scorer, llm_queue, cfg, verbose, store=store)
    if scorer._is_async:
        return _score_async(scorer, llm_queue, cfg, llm_concurrency, verbose, store=store)
    return _score_threaded(scorer, llm_queue, cfg, llm_concurrency, verbose, store=store)


def _score_sequential(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, verbose: int = 0, store=None) -> list[MatchResult]:
    matches = []
    iterator = _maybe_tqdm(llm_queue, len(llm_queue), "LLM", verbose)
    for left_val, candidates_with_scores in iterator:
        candidates = [d["candidate"] for d in candidates_with_scores]
        candidates_debug = _make_debug(candidates_with_scores)
        results = scorer.score(
            left_val, candidates, cfg.context_str,
            llm_threshold=cfg.llm_threshold, match_all=cfg.match_all,
        )
        row_results = _process_results(results, left_val, candidates_with_scores, candidates_debug)
        if store is not None and row_results:
            store.save(left_val, row_results)
        matches.extend(row_results)
    return matches


def _score_threaded(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, llm_concurrency: int, verbose: int = 0, store=None) -> list[MatchResult]:
    """Parallel scoring for sync LLM functions using ThreadPoolExecutor."""
    result_map: dict[int, list[MatchResult]] = {}

    def _score_one(idx: int, left_val: str, candidates_with_scores: list):
        candidates = [d["candidate"] for d in candidates_with_scores]
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
            row_results = _process_results(results, left_val, candidates_with_scores, candidates_debug)
            if store is not None and row_results:
                store.save(left_val, row_results)
            result_map[idx] = row_results

    matches = []
    for i in range(len(llm_queue)):
        matches.extend(result_map.get(i, []))
    return matches


def _score_async(scorer: LLMScorer, llm_queue: list, cfg: ColumnConfig, llm_concurrency: int, verbose: int = 0, store=None) -> list[MatchResult]:
    """Parallel scoring for async LLM functions using asyncio + semaphore."""
    n = len(llm_queue)

    async def _run_all():
        sem = asyncio.Semaphore(llm_concurrency)
        pbar = None
        if verbose >= 1:
            from tqdm import tqdm
            pbar = tqdm(total=n, desc="LLM", file=sys.stderr)

        async def _score_one(left_val: str, candidates_with_scores: list):
            async with sem:
                candidates = [d["candidate"] for d in candidates_with_scores]
                results = await scorer.score_async(
                    left_val, candidates, cfg.context_str,
                    llm_threshold=cfg.llm_threshold, match_all=cfg.match_all,
                )
                if pbar is not None:
                    pbar.update(1)
                candidates_debug = _make_debug(candidates_with_scores)
                row_results = _process_results(results, left_val, candidates_with_scores, candidates_debug)
                if store is not None and row_results:
                    store.save(left_val, row_results)
                return row_results

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
    for row_results in raw:
        matches.extend(row_results)
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
