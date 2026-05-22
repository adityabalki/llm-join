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
        {"score": 0.95, "reasoning": "same drug"},
        {"score": 0.10, "reasoning": "different"},
    ])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("aspirin", ["Bayer Aspirin", "ibuprofen"], "pharma")
    assert result.right_val == "Bayer Aspirin"
    assert result.score == 0.95


def test_returns_none_when_all_below_threshold():
    resp = valid_response([
        {"score": 0.10, "reasoning": "no match"},
        {"score": 0.05, "reasoning": "no match"},
    ])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("aspirin", ["ibuprofen", "tylenol"], "pharma", threshold=0.7)
    assert result is None


def test_reasoning_captured():
    resp = valid_response([{"score": 0.9, "reasoning": "brand contains generic"}])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("aspirin", ["Bayer Aspirin"], "pharma")
    assert result.reasoning == "brand contains generic"


def test_malformed_json_returns_none_with_warning(recwarn):
    scorer = LLMScorer(make_llm("not json at all"))
    result = scorer.score("aspirin", ["Bayer Aspirin"], "pharma")
    assert result is None
    assert len(recwarn) > 0


def test_embed_rank_set_correctly():
    resp = valid_response([
        {"score": 0.3, "reasoning": "weak"},
        {"score": 0.9, "reasoning": "strong"},
    ])
    scorer = LLMScorer(make_llm(resp))
    result = scorer.score("aspirin", ["ibuprofen", "Bayer Aspirin"], "pharma")
    assert result.embed_rank == 1  # index of best match in candidates list

def test_async_llm_raises_on_sync_score():
    async def async_llm(prompt: str) -> str:
        return "[]"
    scorer = LLMScorer(async_llm)
    with pytest.raises(TypeError, match="score_async"):
        scorer.score("aspirin", ["Bayer Aspirin"], "pharma")

@pytest.mark.asyncio
async def test_score_async_with_async_llm():
    resp = valid_response([{"score": 0.9, "reasoning": "match"}])
    async def async_llm(prompt: str) -> str:
        return resp
    scorer = LLMScorer(async_llm)
    result = await scorer.score_async("aspirin", ["Bayer Aspirin"], "pharma")
    assert result is not None
    assert result.score == 0.9

def test_non_list_json_returns_none_with_warning(recwarn):
    scorer = LLMScorer(make_llm('{"error": "model overloaded"}'))
    result = scorer.score("aspirin", ["Bayer Aspirin"], "pharma")
    assert result is None
    assert len(recwarn) > 0
