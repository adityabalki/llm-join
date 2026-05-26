import asyncio
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import faiss
import numpy as np


def _maybe_tqdm(iterable, total: int, desc: str, verbose: int):
    """Wrap iterable in tqdm if verbose >= 1; else passthrough."""
    if verbose < 1:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc, file=sys.stderr)


class EmbeddingRetriever:
    """FAISS-backed retrieval with batched, optionally-parallel embedding.

    The embed function may be sync or async. The library:
      * splits texts into batches of `batch_size`
      * runs up to `embed_concurrency` batches in parallel
      * picks ThreadPoolExecutor for sync funcs, asyncio.Semaphore for async funcs
    """

    def __init__(
        self,
        embed_fn: Callable,
        batch_size: int = 32,
        embed_concurrency: int = 10,
        verbose: int = 0,
    ):
        self._embed_fn = embed_fn
        self._batch_size = max(1, int(batch_size))
        self._embed_concurrency = max(1, int(embed_concurrency))
        self._is_async = inspect.iscoroutinefunction(embed_fn)
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Embedding orchestration
    # ------------------------------------------------------------------

    def _embed_all(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")

        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        n_batches = len(batches)
        if self._is_async:
            results = self._embed_async(batches)
        elif self._embed_concurrency <= 1 or n_batches <= 1:
            results = [
                self._embed_fn(b)
                for b in _maybe_tqdm(batches, n_batches, "Embedding", self._verbose)
            ]
        else:
            results = self._embed_threaded(batches)

        # Convert each batch result to float32 ndarray, then stack
        arrays = [np.asarray(r, dtype="float32") for r in results]
        return np.vstack(arrays) if arrays else np.zeros((0, 1), dtype="float32")

    def _embed_threaded(self, batches: list[list[str]]) -> list:
        """Parallel batch embedding for sync embed_fn via ThreadPoolExecutor."""
        from concurrent.futures import as_completed
        results: list = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self._embed_concurrency) as ex:
            futures = {ex.submit(self._embed_fn, b): i for i, b in enumerate(batches)}
            for fut in _maybe_tqdm(as_completed(futures), len(futures), "Embedding", self._verbose):
                idx = futures[fut]
                results[idx] = fut.result()
        return results

    def _embed_async(self, batches: list[list[str]]) -> list:
        """Parallel batch embedding for async embed_fn via asyncio + semaphore."""
        verbose = self._verbose
        n_batches = len(batches)

        async def _run_all():
            sem = asyncio.Semaphore(self._embed_concurrency)
            results: list = [None] * n_batches
            done_count = [0]

            pbar = None
            if verbose >= 1:
                from tqdm import tqdm
                pbar = tqdm(total=n_batches, desc="Embedding", file=sys.stderr)

            async def _embed_one(idx, batch):
                async with sem:
                    r = await self._embed_fn(batch)
                    results[idx] = r
                    done_count[0] += 1
                    if pbar is not None:
                        pbar.update(1)

            tasks = [_embed_one(i, b) for i, b in enumerate(batches)]
            await asyncio.gather(*tasks)
            if pbar is not None:
                pbar.close()
            return results

        # Handle Jupyter / Databricks where an event loop is already running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _run_all()).result()
        return asyncio.run(_run_all())

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_vals: list[str],
        corpus_vals: list[str],
        top_k: int = 5,
    ) -> list[list[str]]:
        if not corpus_vals:
            return [[] for _ in query_vals]

        top_k = min(top_k, len(corpus_vals))
        # Single embed pass for corpus + query (one fewer round-trip)
        all_texts = list(corpus_vals) + list(query_vals)
        all_vecs = self._embed_all(all_texts)
        corpus_vecs = all_vecs[: len(corpus_vals)]
        query_vecs = all_vecs[len(corpus_vals) :]

        faiss.normalize_L2(corpus_vecs)
        faiss.normalize_L2(query_vecs)

        dim = corpus_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_vecs)

        _, indices = index.search(query_vecs, top_k)

        return [[corpus_vals[i] for i in row if i >= 0] for row in indices]

    def retrieve_with_scores(
        self,
        query_vals: list[str],
        corpus_vals: list[str],
        top_k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """Returns list of (candidate, cosine_score) per query, sorted by score desc."""
        if not corpus_vals:
            return [[] for _ in query_vals]

        top_k = min(top_k, len(corpus_vals))
        all_texts = list(corpus_vals) + list(query_vals)
        all_vecs = self._embed_all(all_texts)
        corpus_vecs = all_vecs[: len(corpus_vals)]
        query_vecs = all_vecs[len(corpus_vals) :]

        faiss.normalize_L2(corpus_vecs)
        faiss.normalize_L2(query_vecs)

        dim = corpus_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_vecs)

        scores, indices = index.search(query_vecs, top_k)

        return [
            [
                (corpus_vals[idx], float(score))
                for idx, score in zip(row_indices, row_scores)
                if idx >= 0
            ]
            for row_indices, row_scores in zip(indices, scores)
        ]
