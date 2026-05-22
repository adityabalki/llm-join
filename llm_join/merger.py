from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class MatchResult:
    left_val: str
    right_val: str
    score: float
    reasoning: str
    embed_rank: int = 0


class Merger:
    def merge(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        left_col: str,
        right_col: str,
        matches: list[MatchResult],
        how: str = "inner",
        return_reasoning: bool = False,
    ) -> pd.DataFrame:
        if not matches:
            if how == "inner":
                return pd.DataFrame(columns=list(df1.columns) + list(df2.columns))
            return df1.copy() if how == "left" else df2.copy()

        match_df = pd.DataFrame({
            left_col: [m.left_val for m in matches],
            right_col: [m.right_val for m in matches],
            "_llm_score": [m.score for m in matches],
            "_llm_reasoning": [m.reasoning for m in matches],
            "_embed_rank": [m.embed_rank for m in matches],
        })

        df2_with_key = df2.merge(match_df, on=right_col, how="right")
        result = df1.merge(df2_with_key, left_on=left_col, right_on=left_col, how=how)

        if not return_reasoning:
            result = result.drop(columns=["_llm_score", "_llm_reasoning", "_embed_rank"], errors="ignore")

        return result.reset_index(drop=True)
