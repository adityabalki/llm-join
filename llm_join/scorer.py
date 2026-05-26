import asyncio
import json
import re
import sys
import time
import warnings
from typing import Callable, Optional

from llm_join.merger import MatchResult
from llm_join.prompts import build_prompt
from llm_join.stats import JoinStats, is_rate_limit_error


class LLMScorer:
    def __init__(
        self,
        llm: Callable,
        max_retries: int = 3,
        stats: Optional[JoinStats] = None,
        verbose: int = 0,
    ):
        self._llm = llm
        self._is_async = asyncio.iscoroutinefunction(llm)
        self._max_retries = max_retries
        self._stats = stats
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Internal stats helpers (safe no-op if stats is None)
    # ------------------------------------------------------------------

    def _inc(self, field: str, n: int = 1) -> None:
        if self._stats is not None:
            self._stats.inc(field, n)

    def _record_exc(self, exc: Exception) -> None:
        if is_rate_limit_error(exc):
            self._inc("n_rate_limited")

    def _log_record(self, results, left_val: str) -> None:
        """verbose=2: print one line per scored record."""
        if self._verbose < 2 or not results:
            return
        for r in results:
            print(
                f'  Row: "{left_val}" -> "{r.right_val}" '
                f'[score={r.score:.3f}, embed_rank={r.embed_rank}, {r.match_method}]',
                file=sys.stderr,
                flush=True,
            )

    def _log_batch_failure(self, left_val: str, exc: Exception, attempt: int) -> None:
        """verbose>=1: per-batch failure detail (not raw warning spam)."""
        if self._verbose >= 1:
            print(
                f'  Retry {attempt}/{self._max_retries + 1} for "{left_val}": {exc!r}',
                file=sys.stderr,
                flush=True,
            )

    # ------------------------------------------------------------------
    # Sync path
    # ------------------------------------------------------------------

    def score(
        self,
        left_val: str,
        candidates: list[str],
        context_str: str,
        llm_threshold: float = 0.7,
        match_all: bool = False,
    ) -> Optional[list[MatchResult]]:
        if self._is_async:
            raise TypeError(
                "LLM is async; call score_async() instead, or pass a sync callable."
            )
        prompt = build_prompt(left_val, candidates, context_str)
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._llm(prompt)
                self._inc("n_llm_called")
                results = self._parse(left_val, candidates, raw, llm_threshold, match_all=match_all)
                self._log_record(results, left_val)
                return results
            except Exception as exc:
                last_exc = exc
                self._record_exc(exc)
                if attempt < self._max_retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s, ...
                    self._log_batch_failure(left_val, exc, attempt + 1)
                    if self._verbose == 0:
                        warnings.warn(
                            f"LLM call failed for '{left_val}' (attempt {attempt + 1}/{self._max_retries + 1}): "
                            f"{exc!r}. Retrying in {wait}s.",
                            UserWarning,
                            stacklevel=2,
                        )
                    time.sleep(wait)
        self._inc("n_failed")
        warnings.warn(
            f"LLM call failed for '{left_val}' after {self._max_retries + 1} attempts: {last_exc!r}. "
            "Falling back to top embed candidate.",
            UserWarning,
            stacklevel=2,
        )
        return None  # signals LLM failure — caller applies embed fallback

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def score_async(
        self,
        left_val: str,
        candidates: list[str],
        context_str: str,
        llm_threshold: float = 0.7,
        match_all: bool = False,
    ) -> Optional[list[MatchResult]]:
        prompt = build_prompt(left_val, candidates, context_str)
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._is_async:
                    raw = await self._llm(prompt)
                else:
                    raw = self._llm(prompt)
                self._inc("n_llm_called")
                results = self._parse(left_val, candidates, raw, llm_threshold, match_all=match_all)
                self._log_record(results, left_val)
                return results
            except Exception as exc:
                last_exc = exc
                self._record_exc(exc)
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    self._log_batch_failure(left_val, exc, attempt + 1)
                    if self._verbose == 0:
                        warnings.warn(
                            f"LLM call failed for '{left_val}' (attempt {attempt + 1}/{self._max_retries + 1}): "
                            f"{exc!r}. Retrying in {wait}s.",
                            UserWarning,
                            stacklevel=2,
                        )
                    await asyncio.sleep(wait)
        self._inc("n_failed")
        warnings.warn(
            f"LLM call failed for '{left_val}' after {self._max_retries + 1} attempts: {last_exc!r}. "
            "Falling back to top embed candidate.",
            UserWarning,
            stacklevel=2,
        )
        return None  # signals LLM failure — caller applies embed fallback

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(
        self,
        left_val: str,
        candidates: list[str],
        raw: str,
        llm_threshold: float,
        match_all: bool = False,
    ) -> list[MatchResult]:
        try:
            # strip markdown code fences if present
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            # retry: find JSON array anywhere in response
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    warnings.warn(f"LLM returned malformed JSON for '{left_val}': {raw!r}")
                    return []
            else:
                warnings.warn(f"LLM returned malformed JSON for '{left_val}': {raw!r}")
                return []

        if not isinstance(parsed, list):
            warnings.warn(f"LLM returned non-array JSON for '{left_val}': {raw!r}")
            return []

        # Collect all valid scored items above threshold, deduplicated by index
        seen_indices: set[int] = set()
        scored = []
        for item in parsed:
            idx = item.get("index", -1)
            if not (0 <= idx < len(candidates)):
                continue
            if idx in seen_indices:
                continue  # LLM returned duplicate index — skip
            seen_indices.add(idx)
            score = float(item.get("score", 0.0))
            if score >= llm_threshold:
                scored.append((score, idx, item.get("reasoning", "")))

        if not scored:
            return []

        if match_all:
            # Return every candidate above threshold, sorted best score first
            scored.sort(key=lambda x: x[0], reverse=True)
            total = len(scored)
            return [
                MatchResult(
                    left_val=left_val,
                    right_val=candidates[idx],
                    score=score,
                    reasoning=reasoning + f" [match_all: {total} candidates above threshold returned]",
                    embed_rank=idx,
                )
                for score, idx, reasoning in scored
            ]

        # Default: exactly ONE result — best score, break ties by embed rank (lowest idx = top FAISS hit)
        best_score = max(s for s, _, _ in scored)
        best_idx, best_reasoning = min(
            ((idx, reasoning) for score, idx, reasoning in scored if score == best_score),
            key=lambda x: x[0],
        )
        return [MatchResult(
            left_val=left_val,
            right_val=candidates[best_idx],
            score=best_score,
            reasoning=best_reasoning,
            embed_rank=best_idx,
        )]
