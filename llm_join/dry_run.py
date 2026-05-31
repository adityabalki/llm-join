"""dry_run mode for llm-join — validate params and estimate cost without calling llm_fn/embed_fn."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import pandas as pd


@dataclass
class DryRunResult:
    """Result of a dry_run=True fuzzy_join call."""
    validation_errors: list[str] = field(default_factory=list)
    n_left_total: int = 0
    n_left_unique: int = 0
    n_right_total: int = 0
    n_right_unique: int = 0
    n_llm_calls_max: int = 0
    n_llm_calls_min: int = 0
    avg_tokens_per_call: int = 0
    total_tokens_min: int = 0
    total_tokens_max: int = 0
    retrieval: str = "embedding"
    top_k: int = 5

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0

    def summary(self) -> str:
        lines = []
        if self.validation_errors:
            lines.append(f"dry_run: {len(self.validation_errors)} validation error(s):")
            for err in self.validation_errors:
                lines.append(f"  - {err}")
        else:
            lines.append("dry_run: no validation errors")
        lines.append(f"  left rows: {self.n_left_total} (unique: {self.n_left_unique})")
        lines.append(f"  right rows: {self.n_right_total} (unique: {self.n_right_unique})")
        lines.append(
            f"  LLM calls: {self.n_llm_calls_min}-{self.n_llm_calls_max}"
            f"  ({self.n_llm_calls_min} if embed_skip_threshold fires; {self.n_llm_calls_max} worst case)"
        )
        lines.append(f"  tokens/call (est.): ~{self.avg_tokens_per_call}")
        lines.append(
            f"  total input tokens (est.): {self.total_tokens_min}-{self.total_tokens_max}"
        )
        lines.append(
            "  note: token estimates assume ~4 chars/token (English text). "
            "Non-English text (Chinese, Japanese, Korean, Arabic, etc.) or "
            "code-heavy values will use more tokens than shown here."
        )
        return "\n".join(lines)


def estimate(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left_on: Union[str, list],
    right_on: Union[str, list],
    llm_fn: object,
    embed_fn: object,
    context: str,
    column_context: dict,
    top_k: int,
    llm_threshold: float,
    batch_size: int,
    embed_concurrency: int,
    embed_skip_threshold: float,
    max_llm_calls: Optional[int],
    max_retries: int,
    match_all: bool,
    retrieval: str,
    bm25_stopwords: object,
    checkpoint_path: Optional[str],
) -> DryRunResult:
    """Validate params and estimate LLM call/token counts. Never calls llm_fn or embed_fn."""
    errors: list[str] = []

    # 1. left_on column(s) exist in df1
    left_cols = [left_on] if isinstance(left_on, str) else left_on
    for col in left_cols:
        if col not in df1.columns:
            errors.append(f"Column {col!r} not found in df1")

    # 2. right_on column(s) exist in df2
    right_cols = [right_on] if isinstance(right_on, str) else right_on
    for col in right_cols:
        if col not in df2.columns:
            errors.append(f"Column {col!r} not found in df2")

    # 3. llm_fn is callable
    if not callable(llm_fn):
        errors.append(f"llm_fn must be callable, got {type(llm_fn).__name__!r}")

    # 4. embed_fn is callable
    if not callable(embed_fn):
        errors.append(f"embed_fn must be callable, got {type(embed_fn).__name__!r}")

    # 5. context not empty
    if not context or not context.strip():
        errors.append(
            "context must not be empty — describe what the columns represent and what kind of match to make. "
            "Example: \"pharmaceutical drug names — match generic INN names to US brand names\""
        )

    # 6. retrieval valid
    if retrieval not in ("embedding", "bm25", "hybrid"):
        errors.append(
            f"retrieval must be one of 'embedding', 'bm25', 'hybrid', got {retrieval!r}"
        )

    # 7. llm_threshold valid
    if not (0.0 < llm_threshold <= 1.0):
        errors.append(f"llm_threshold must be in (0, 1], got {llm_threshold}")

    # 8. top_k valid
    if top_k < 1:
        errors.append(f"top_k must be >= 1, got {top_k}")

    # 9. batch_size valid
    if batch_size < 1:
        errors.append(f"batch_size must be >= 1, got {batch_size}")

    # 10. embed_concurrency valid
    if embed_concurrency < 1:
        errors.append(f"embed_concurrency must be >= 1, got {embed_concurrency}")

    # 11. embed_skip_threshold valid
    if not (0.0 < embed_skip_threshold <= 1.0):
        errors.append(f"embed_skip_threshold must be in (0, 1], got {embed_skip_threshold}")

    # 12. max_llm_calls valid
    if max_llm_calls is not None and max_llm_calls < 1:
        errors.append(f"max_llm_calls must be >= 1, got {max_llm_calls}")

    # 13. checkpoint_path parent dir exists (if provided)
    if checkpoint_path is not None:
        parent = os.path.dirname(os.path.abspath(checkpoint_path))
        if not os.path.isdir(parent):
            errors.append(
                f"checkpoint_path parent directory does not exist: {parent!r}"
            )

    # --- Scale estimates (always computed, even if errors present) ---

    # Left vals
    n_left_total = len(df1)
    if isinstance(left_on, list):
        left_series = df1[left_on].astype(str).agg(" ".join, axis=1) if all(c in df1.columns for c in left_on) else pd.Series([""] * len(df1))
    else:
        left_series = df1[left_on].astype(str) if left_on in df1.columns else pd.Series([""] * len(df1))
    n_left_unique = left_series.nunique()

    # Right vals
    n_right_total = len(df2)
    if isinstance(right_on, list):
        right_series = df2[right_on].astype(str).agg(" ".join, axis=1) if all(c in df2.columns for c in right_on) else pd.Series([""] * len(df2))
    else:
        right_series = df2[right_on].astype(str) if right_on in df2.columns else pd.Series([""] * len(df2))
    n_right_unique = right_series.nunique()

    # LLM call bounds
    n_llm_calls_max = n_left_unique
    if max_llm_calls is not None and max_llm_calls >= 1:
        n_llm_calls_max = min(n_llm_calls_max, max_llm_calls)

    if embed_skip_threshold < 1.0:
        n_llm_calls_min = 0
    else:
        n_llm_calls_min = n_llm_calls_max

    # Token estimation
    # Basis: ~4 chars per token (OpenAI/Anthropic rule-of-thumb for English text).
    # Non-English corpora (Chinese/Japanese/Korean ~1.5 chars/token, code ~3 chars/token) will use more tokens than estimated.
    # overhead=50 covers prompt template boilerplate (role line, format instruction, pair labels).
    # Formula: context_tokens + avg_left_tokens + (top_k * avg_right_tokens) + overhead
    context_tokens = len(context) // 4
    if n_left_unique > 0:
        left_tokens = int(left_series.str.len().mean()) // 4
    else:
        left_tokens = 0
    if n_right_unique > 0:
        right_tokens = int(right_series.str.len().mean()) // 4
    else:
        right_tokens = 0
    overhead = 50  # prompt boilerplate: ~50 tokens for role + format instruction + pair labels
    safe_top_k = max(top_k, 1)
    avg_tokens_per_call = context_tokens + left_tokens + (safe_top_k * right_tokens) + overhead

    total_tokens_min = n_llm_calls_min * avg_tokens_per_call
    total_tokens_max = n_llm_calls_max * avg_tokens_per_call

    return DryRunResult(
        validation_errors=errors,
        n_left_total=n_left_total,
        n_left_unique=n_left_unique,
        n_right_total=n_right_total,
        n_right_unique=n_right_unique,
        n_llm_calls_max=n_llm_calls_max,
        n_llm_calls_min=n_llm_calls_min,
        avg_tokens_per_call=avg_tokens_per_call,
        total_tokens_min=total_tokens_min,
        total_tokens_max=total_tokens_max,
        retrieval=retrieval,
        top_k=top_k,
    )
