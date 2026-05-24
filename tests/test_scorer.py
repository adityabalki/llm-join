import json
import pytest
import pytest_asyncio
from llm_join.scorer import LLMScorer
from llm_join.merger import MatchResult


def make_llm(response: str):
    """Returns a sync llm callable that returns a fixed string."""
    def llm(prompt: str) -> str:
        return response
    return llm


def valid_response(scores):
    return json.dumps([
        {"index": i, "score": s["score"], "reasoning": s["reasoning"]}
        for i, s in enumerate(scores)
    ])


def test_returns_best_match():
    resp = valid_response([
        {"score": 0.95, "reasoning": "strong match"},
        {"score": 0.10, "reasoning": "different"},
    ])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc", "JPMorgan Chase"], "company names")
    assert len(result) == 1
    assert result[0].right_val == "The Goldman Sachs Group Inc"
    assert result[0].score == 0.95


def test_returns_empty_when_all_below_threshold():
    resp = valid_response([
        {"score": 0.10, "reasoning": "no match"},
        {"score": 0.05, "reasoning": "no match"},
    ])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("Goldman Sachs & Co.", ["JPMorgan", "Citibank"], "company names", llm_threshold=0.7)
    assert result == []  # LLM ran OK, nothing above threshold


def test_reasoning_captured():
    resp = valid_response([{"score": 0.9, "reasoning": "legal name variant"}])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc"], "company names")
    assert result[0].reasoning == "legal name variant"


def test_malformed_json_returns_empty_with_warning(recwarn):
    scorer = LLMScorer(make_llm("not json at all"))
    result = scorer.score("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc"], "company names")
    assert result == []  # parse failed → empty, not None (None = LLM exception)
    assert len(recwarn) > 0


def test_embed_rank_set_correctly():
    resp = valid_response([
        {"score": 0.3, "reasoning": "weak"},
        {"score": 0.9, "reasoning": "strong"},
    ])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("Goldman Sachs & Co.", ["JPMorgan", "The Goldman Sachs Group Inc"], "company names")
    assert result[0].embed_rank == 1  # index 1 had the best score


def test_tied_returns_one_result_best_embed_rank():
    # Two candidates tied at 0.90 — match_all=False → return exactly ONE (lowest index = best embed rank)
    resp = valid_response([
        {"score": 0.90, "reasoning": "match A"},
        {"score": 0.90, "reasoning": "match B"},
        {"score": 0.50, "reasoning": "weak"},
    ])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("Goldman Sachs & Co.", ["Entity A", "Entity B", "Entity C"], "company names")
    assert len(result) == 1          # exactly one result, no tie expansion
    assert result[0].score == 0.90
    assert result[0].right_val == "Entity A"  # index 0 wins tie-break (best embed rank)


def test_duplicate_index_deduplicated():
    # LLM returns same index twice — should only produce one result
    raw = json.dumps([
        {"index": 0, "score": 0.95, "reasoning": "first"},
        {"index": 0, "score": 0.95, "reasoning": "duplicate"},
    ])
    scorer = LLMScorer(make_llm(raw))
    result = scorer.score("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc"], "company names")
    assert len(result) == 1


def test_async_llm_raises_on_sync_score():
    async def async_llm(prompt: str) -> str:
        return "[]"
    scorer = LLMScorer(async_llm)
    with pytest.raises(TypeError, match="score_async"):
        scorer.score("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc"], "company names")


@pytest.mark.asyncio
async def test_score_async_with_async_llm():
    resp = valid_response([{"score": 0.9, "reasoning": "match"}])
    async def async_llm(prompt: str) -> str:
        return resp
    scorer = LLMScorer(async_llm)
    result = await scorer.score_async("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc"], "company names")
    assert result is not None
    assert len(result) == 1
    assert result[0].score == 0.9


def test_non_list_json_returns_empty_with_warning(recwarn):
    scorer = LLMScorer(make_llm('{"error": "model overloaded"}'))
    result = scorer.score("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc"], "company names")
    assert result == []
    assert len(recwarn) > 0


def test_llm_failure_returns_none(recwarn):
    # None = LLM raised exception (triggers embed fallback in join.py)
    def failing_llm(prompt: str) -> str:
        raise RuntimeError("API down")
    scorer = LLMScorer(failing_llm, max_retries=0)
    result = scorer.score("Goldman Sachs & Co.", ["The Goldman Sachs Group Inc"], "company names")
    assert result is None  # not [] — signals failure to caller
