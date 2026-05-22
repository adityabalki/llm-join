from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColumnConfig:
    left_col: str
    right_col: str
    context: str = ""
    column_context: dict = field(default_factory=dict)
    top_k: int = 5
    threshold: float = 0.7
    embed_model: str = "all-MiniLM-L6-v2"
    embed_fn: Optional[object] = None  # callable(list[str]) -> np.ndarray
    batch_size: int = 32
    embed_threshold: Optional[float] = None
    max_llm_calls: Optional[int] = None

    def __post_init__(self):
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

    @property
    def context_str(self) -> str:
        parts = []
        if self.context:
            parts.append(self.context)
        for col in (self.left_col, self.right_col):
            if col in self.column_context:
                parts.append(f"{col}: {self.column_context[col]}")
        return ". ".join(parts)
