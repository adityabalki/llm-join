"""BM25 lexical retrieval — catches token-level matches embeddings miss.

Wraps the `bm25s` library. Same interface as EmbeddingRetriever so callers can
substitute or compose retrievers.
"""
from typing import Optional, Union

import bm25s


class BM25Retriever:
    """Lexical scoring via BM25.

    Strong on rare-token matches (codes, SKUs, IDs, drug names, acronyms in your
    private corpus). Weak on paraphrases / synonyms — pair with embeddings for
    those via HybridRetriever.
    """

    def __init__(self, stopwords: Optional[Union[str, list]] = None):
        # stopwords: None (default — keep all tokens), "en", or custom list
        self._stopwords = stopwords
        self._retriever: Optional[bm25s.BM25] = None
        self._corpus: list[str] = []

    def _tokenize(self, texts: list[str]):
        return bm25s.tokenize(texts, stopwords=self._stopwords, show_progress=False)

    def _index(self, corpus_vals: list[str]) -> None:
        self._corpus = list(corpus_vals)
        if not self._corpus:
            self._retriever = None
            return
        tokens = self._tokenize(self._corpus)
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens, show_progress=False)

    def retrieve_with_scores(
        self,
        query_vals: list[str],
        corpus_vals: list[str],
        top_k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """Returns list of (candidate, bm25_score) per query, sorted by score desc."""
        if not corpus_vals or not query_vals:
            return [[] for _ in query_vals]

        # (Re)index on each call so the interface mirrors EmbeddingRetriever.
        self._index(corpus_vals)
        if self._retriever is None:
            return [[] for _ in query_vals]

        top_k_capped = min(top_k, len(self._corpus))
        query_tokens = self._tokenize(list(query_vals))
        # bm25s returns (results, scores) — results are corpus indices
        result_idx, result_scores = self._retriever.retrieve(
            query_tokens, k=top_k_capped, show_progress=False,
        )

        out: list[list[tuple[str, float]]] = []
        for row_idx, row_scores in zip(result_idx, result_scores):
            row = [
                (self._corpus[int(i)], float(s))
                for i, s in zip(row_idx, row_scores)
                if int(i) >= 0 and float(s) > 0  # skip zero-score / invalid hits
            ]
            out.append(row)
        return out
