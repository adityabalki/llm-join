from dataclasses import dataclass
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
        if how not in ("inner", "left", "right", "outer"):
            raise ValueError(f"how must be inner/left/right/outer, got '{how}'")
        if right_col in df1.columns and right_col != left_col:
            raise ValueError(
                f"right_col '{right_col}' already exists in df1; rename before merging"
            )

        if not matches:
            empty = pd.DataFrame(columns=list(df1.columns) + [
                c for c in df2.columns if c != right_col or right_col == left_col
            ])
            if how == "inner":
                return empty
            if how == "left":
                return df1.copy()
            if how == "right":
                return df2.copy()
            # outer: return both frames with NaN fills
            return pd.merge(df1, df2, left_on=left_col, right_on=right_col, how="outer")

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
