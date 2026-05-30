"""Checkpoint store for llm-join — saves/resumes large join progress."""
import json
import os
import tempfile
import threading
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_join.merger import MatchResult

CHECKPOINT_VERSION = "0.6.0"


class CheckpointStore:
    """Thread-safe, atomic JSON checkpoint for fuzzy_join progress.

    - Parent dir must exist at construction time (raises ValueError if not).
    - Corrupt or version-mismatched files emit UserWarning and start fresh.
    - Writes are atomic: tempfile in same dir + os.replace.
    - Thread-safe via threading.Lock (used by threaded scoring path).
    """

    def __init__(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            raise ValueError(
                f"llm-join: checkpoint parent directory does not exist: {parent!r}"
            )
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, list] = {}  # left_val -> list[dict]
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            warnings.warn(
                f"llm-join: corrupt checkpoint at {self._path!r}. Starting fresh.",
                UserWarning,
                stacklevel=3,
            )
            return
        if raw.get("version") != CHECKPOINT_VERSION:
            warnings.warn(
                f"llm-join: checkpoint version mismatch "
                f"(file={raw.get('version')!r}, expected={CHECKPOINT_VERSION!r}). "
                f"Starting fresh.",
                UserWarning,
                stacklevel=3,
            )
            return
        self._data = raw.get("results", {})

    def is_done(self, left_val: str) -> bool:
        return left_val in self._data

    def get_results(self, left_val: str) -> list:
        """Return list[MatchResult] for a completed left_val."""
        from llm_join.merger import MatchResult
        raw_list = self._data.get(left_val, [])
        results = []
        for d in raw_list:
            results.append(MatchResult(
                left_val=d["left_val"],
                right_val=d["right_val"],
                score=d["score"],
                reasoning=d["reasoning"],
                embed_rank=d.get("embed_rank", 0),
                match_method=d.get("match_method", "llm"),
                candidates=d.get("candidates"),
            ))
        return results

    def save(self, left_val: str, results: list) -> None:
        """Thread-safe save of results for one left_val."""
        serialized = []
        for r in results:
            serialized.append({
                "left_val": r.left_val,
                "right_val": r.right_val,
                "score": r.score,
                "reasoning": r.reasoning,
                "embed_rank": r.embed_rank,
                "match_method": r.match_method,
                "candidates": r.candidates,
            })
        with self._lock:
            self._data[left_val] = serialized
            self._flush()

    def _flush(self) -> None:
        """Atomic write via tempfile + os.replace. Call with lock held."""
        parent = os.path.dirname(os.path.abspath(self._path))
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": CHECKPOINT_VERSION, "results": self._data}, f)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def delete(self) -> None:
        """Delete checkpoint file if it exists."""
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def n_done(self) -> int:
        return len(self._data)
