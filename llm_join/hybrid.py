"""Hybrid retrieval — embedding (semantic) + BM25 (lexical) combined via RRF.

Each retriever proposes top_k candidates. We compute a Reciprocal Rank Fusion
(RRF) score per unique candidate across both lists, then rank.

Sort order:
  1. rrf_score DESC                  (combined relevance signal)
  2. appeared_in_both DESC           (prefer dual-evidence over single-list)
  3. embed_score DESC                (final tie-break)

Then truncate strictly to top_k.

Output format is `list[list[dict]]` per query, where each dict has:
    candidate     str
    embed_score   Optional[float]   None if missing from embed list
    bm25_score    Optional[float]   None if missing from bm25 list
    rrf_score     float             combined fusion score
    embed_rank    Optional[int]     0-based rank in embed list, None if absent
    bm25_rank     Optional[int]     0-based rank in bm25 list, None if absent
    source        str               "embed", "bm25", or "both"
"""
from typing import Optional


# RRF constant — standard choice (Cormack et al. 2009). Lower k weights top ranks
# more heavily; higher k flattens. 60 is the convention.
_RRF_K = 60


class HybridRetriever:
    def __init__(self, embed_retriever, bm25_retriever):
        self._embed = embed_retriever
        self._bm25 = bm25_retriever

    def retrieve_with_scores(
        self,
        query_vals: list[str],
        corpus_vals: list[str],
        top_k: int = 5,
    ) -> list[list[dict]]:
        if not corpus_vals or not query_vals:
            return [[] for _ in query_vals]

        # Get top_k from each retriever
        embed_results = self._embed.retrieve_with_scores(query_vals, corpus_vals, top_k=top_k)
        bm25_results = self._bm25.retrieve_with_scores(query_vals, corpus_vals, top_k=top_k)

        out: list[list[dict]] = []
        for embed_row, bm25_row in zip(embed_results, bm25_results):
            merged = self._fuse(embed_row, bm25_row, top_k=top_k)
            out.append(merged)
        return out

    @staticmethod
    def _fuse(
        embed_row: list[tuple[str, float]],
        bm25_row: list[tuple[str, float]],
        top_k: int,
    ) -> list[dict]:
        # Per-candidate record keyed by candidate string
        records: dict[str, dict] = {}

        for rank, (c, score) in enumerate(embed_row):
            records.setdefault(c, {
                "candidate": c,
                "embed_score": None,
                "bm25_score": None,
                "embed_rank": None,
                "bm25_rank": None,
            })
            records[c]["embed_score"] = score
            records[c]["embed_rank"] = rank

        for rank, (c, score) in enumerate(bm25_row):
            records.setdefault(c, {
                "candidate": c,
                "embed_score": None,
                "bm25_score": None,
                "embed_rank": None,
                "bm25_rank": None,
            })
            records[c]["bm25_score"] = score
            records[c]["bm25_rank"] = rank

        # Compute RRF + source label
        for rec in records.values():
            rrf = 0.0
            sources: list[str] = []
            if rec["embed_rank"] is not None:
                rrf += 1.0 / (_RRF_K + rec["embed_rank"] + 1)
                sources.append("embed")
            if rec["bm25_rank"] is not None:
                rrf += 1.0 / (_RRF_K + rec["bm25_rank"] + 1)
                sources.append("bm25")
            rec["rrf_score"] = rrf
            rec["source"] = "both" if len(sources) == 2 else sources[0]

        # Sort: (rrf DESC, both-list DESC, embed_score DESC)
        def sort_key(rec):
            both_signal = 1 if rec["source"] == "both" else 0
            embed_score = rec["embed_score"] if rec["embed_score"] is not None else -1.0
            return (-rec["rrf_score"], -both_signal, -embed_score)

        sorted_records = sorted(records.values(), key=sort_key)
        return sorted_records[:top_k]
