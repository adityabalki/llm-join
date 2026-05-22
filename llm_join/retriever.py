from typing import Callable
import faiss
import numpy as np


class EmbeddingRetriever:
    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray]):
        self._embed_fn = embed_fn

    def _embed(self, texts: list[str]) -> np.ndarray:
        arr = self._embed_fn(texts)
        return np.array(arr, dtype="float32")

    def retrieve(
        self,
        query_vals: list[str],
        corpus_vals: list[str],
        top_k: int = 5,
    ) -> list[list[str]]:
        if not corpus_vals:
            return [[] for _ in query_vals]

        top_k = min(top_k, len(corpus_vals))
        corpus_vecs = self._embed(corpus_vals)
        query_vecs = self._embed(query_vals)

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(corpus_vecs)
        faiss.normalize_L2(query_vecs)

        dim = corpus_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_vecs)

        _, indices = index.search(query_vecs, top_k)

        return [
            [corpus_vals[i] for i in row if i >= 0]
            for row in indices
        ]

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
        corpus_vecs = self._embed(corpus_vals)
        query_vecs = self._embed(query_vals)

        faiss.normalize_L2(corpus_vecs)
        faiss.normalize_L2(query_vecs)

        dim = corpus_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_vecs)

        scores, indices = index.search(query_vecs, top_k)

        return [
            [(corpus_vals[idx], float(score)) for idx, score in zip(row_indices, row_scores) if idx >= 0]
            for row_indices, row_scores in zip(indices, scores)
        ]
