import asyncio
import json
import re
import time
import warnings
from typing import Callable, Optional

from llm_join.merger import MatchResult
from llm_join.prompts import build_prompt


class LLMScorer:
    def __init__(self, llm: Callable, max_retries: int = 3):
        self._llm = llm
        self._is_async = asyncio.iscoroutinefunction(llm)
        self._max_retries = max_retries

    def score(
        self,
        left_val: str,
        candidates: list[str],
        context_str: str,
        threshold: float = 0.7,
    ) -> Optional[MatchResult]:
        if self._is_async:
            raise TypeError(
                "LLM is async; call score_async() instead, or pass a sync callable."
            )
        prompt = build_prompt(left_val, candidates, context_str)
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._llm(prompt)
                return self._parse(left_val, candidates, raw, threshold)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s, ...
                    warnings.warn(
                        f"LLM call failed for '{left_val}' (attempt {attempt + 1}/{self._max_retries + 1}): "
                        f"{exc!r}. Retrying in {wait}s.",
                        UserWarning,
                        stacklevel=2,
                    )
                    time.sleep(wait)
        warnings.warn(
            f"LLM call failed for '{left_val}' after {self._max_retries + 1} attempts: {last_exc!r}. Skipping row.",
            UserWarning,
            stacklevel=2,
        )
        return None

    async def score_async(
        self,
        left_val: str,
        candidates: list[str],
        context_str: str,
        threshold: float = 0.7,
    ) -> Optional[MatchResult]:
        prompt = build_prompt(left_val, candidates, context_str)
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._is_async:
                    raw = await self._llm(prompt)
                else:
                    raw = self._llm(prompt)
                return self._parse(left_val, candidates, raw, threshold)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    warnings.warn(
                        f"LLM call failed for '{left_val}' (attempt {attempt + 1}/{self._max_retries + 1}): "
                        f"{exc!r}. Retrying in {wait}s.",
                        UserWarning,
                        stacklevel=2,
                    )
                    await asyncio.sleep(wait)
        warnings.warn(
            f"LLM call failed for '{left_val}' after {self._max_retries + 1} attempts: {last_exc!r}. Skipping row.",
            UserWarning,
            stacklevel=2,
        )
        return None

    def _parse(
        self,
        left_val: str,
        candidates: list[str],
        raw: str,
        threshold: float,
    ) -> Optional[MatchResult]:
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
                    return None
            else:
                warnings.warn(f"LLM returned malformed JSON for '{left_val}': {raw!r}")
                return None

        if not isinstance(parsed, list):
            warnings.warn(f"LLM returned non-array JSON for '{left_val}': {raw!r}")
            return None

        best_score = -1.0
        best_idx = -1
        best_reasoning = ""
        for item in parsed:
            idx = item.get("index", -1)
            score = float(item.get("score", 0.0))
            if score > best_score and 0 <= idx < len(candidates):
                best_score = score
                best_idx = idx
                best_reasoning = item.get("reasoning", "")

        if best_idx == -1 or best_score < threshold:
            return None

        return MatchResult(
            left_val=left_val,
            right_val=candidates[best_idx],
            score=best_score,
            reasoning=best_reasoning,
            embed_rank=best_idx,
        )
