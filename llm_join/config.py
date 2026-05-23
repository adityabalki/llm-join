from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class ColumnConfig:
    left_col: str
    right_col: str
    embed_fn: Callable[[list[str]], np.ndarray]
    context: str = ""
    column_context: dict[str, str] = field(default_factory=dict)
    top_k: int = 5
    threshold: float = 0.7
    batch_size: int = 32
    embed_threshold: Optional[float] = None
    max_llm_calls: Optional[int] = None
    max_retries: int = 3

    def __post_init__(self):
        if not self.left_col:
            raise ValueError("left_col must not be empty")
        if not self.right_col:
            raise ValueError("right_col must not be empty")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.embed_threshold is not None and not 0.0 < self.embed_threshold <= 1.0:
            raise ValueError(f"embed_threshold must be in (0, 1], got {self.embed_threshold}")
        if self.max_llm_calls is not None and self.max_llm_calls < 1:
            raise ValueError(f"max_llm_calls must be >= 1, got {self.max_llm_calls}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    @property
    def context_str(self) -> str:
        parts = []
        if self.context:
            parts.append(self.context)
        for col in (self.left_col, self.right_col):
            if col in self.column_context:
                parts.append(f"{col}: {self.column_context[col]}")
        return ". ".join(parts)
