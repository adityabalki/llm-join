"""Stats tracking + summary printing for fuzzy_join runs."""
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class JoinStats:
    """Counters tracked across a fuzzy_join run. Thread-safe via internal lock."""
    n_left_total: int = 0           # df1 rows
    n_left_unique: int = 0          # after left-dedup
    n_right_total: int = 0          # df2 rows
    n_right_unique: int = 0         # after right-dedup
    n_embed_skipped: int = 0        # matched via embed_skip_threshold
    n_llm_called: int = 0           # successful LLM calls
    n_rate_limited: int = 0         # 429s encountered (including retries)
    n_failed: int = 0               # exhausted retries → embed_fallback used
    elapsed_seconds: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def inc(self, field_name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + n)

    def summary_line(self) -> str:
        return (
            f"llm-join: {self.n_left_total} left "
            f"(deduped to {self.n_left_unique}) | "
            f"{self.n_right_total} right (deduped to {self.n_right_unique}) | "
            f"{self.n_embed_skipped} embed_skipped | "
            f"{self.n_llm_called} LLM | "
            f"{self.n_rate_limited} rate-limited | "
            f"{self.n_failed} failed | "
            f"{self.elapsed_seconds:.1f}s total"
        )


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of rate-limit / 429 errors across providers.

    Avoids hard imports of openai / anthropic / etc. Uses class name + message.
    """
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "toomanyrequests" in name:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg
