import asyncio
import json
import re
import warnings
from typing import Callable, Optional

from llm_join.merger import MatchResult
from llm_join.prompts import build_prompt


class LLMScorer:
    def __init__(self, llm: Callable):
        self._llm = llm
        self._is_async = asyncio.iscoroutinefunction(llm)

    def score(
        self,
        left_val: str,
        candidates: list[str],
        context_str: str,
        threshold: float = 0.7,
    ) -> Optional[MatchResult]:
        prompt = build_prompt(left_val, candidates, context_str)
        if self._is_async:
            raw = asyncio.get_event_loop().run_until_complete(self._llm(prompt))
        else:
            raw = self._llm(prompt)
        return self._parse(left_val, candidates, raw, threshold)

    async def score_async(
        self,
        left_val: str,
        candidates: list[str],
        context_str: str,
        threshold: float = 0.7,
    ) -> Optional[MatchResult]:
        prompt = build_prompt(left_val, candidates, context_str)
        if self._is_async:
            raw = await self._llm(prompt)
        else:
            raw = self._llm(prompt)
        return self._parse(left_val, candidates, raw, threshold)

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
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    warnings.warn(f"LLM returned malformed JSON for '{left_val}': {raw!r}")
                    return None
            else:
                warnings.warn(f"LLM returned malformed JSON for '{left_val}': {raw!r}")
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
