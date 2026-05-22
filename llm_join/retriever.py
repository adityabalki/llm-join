from typing import Callable, Optional
import faiss
import numpy as np


class EmbeddingRetriever:
    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        embed_fn: Optional[Callable] = None,
    ):
        self._embed_fn = embed_fn
        self._model_name = embed_model
        self._model = None  # lazy-loaded

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            arr = self._embed_fn(texts)
            return np.array(arr, dtype="float32")
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")

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
