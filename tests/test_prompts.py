from llm_join.prompts import build_prompt

def test_prompt_contains_left_val():
    prompt = build_prompt("aspirin", ["Bayer Aspirin", "ibuprofen"], "pharma")
    assert "aspirin" in prompt

def test_prompt_contains_all_candidates():
    prompt = build_prompt("aspirin", ["Bayer Aspirin", "ibuprofen"], "pharma")
    assert "Bayer Aspirin" in prompt
    assert "ibuprofen" in prompt

def test_prompt_contains_context():
    prompt = build_prompt("aspirin", ["Bayer Aspirin"], "pharmaceutical drug names")
    assert "pharmaceutical drug names" in prompt

def test_prompt_instructs_json():
    prompt = build_prompt("aspirin", ["Bayer Aspirin"], "pharma")
    assert "JSON" in prompt

def test_prompt_indexes_candidates():
    prompt = build_prompt("x", ["a", "b", "c"], "")
    assert "0." in prompt
    assert "1." in prompt
    assert "2." in prompt

def test_prompt_empty_context():
    prompt = build_prompt("x", ["y"], "")
    assert "x" in prompt
    assert "y" in prompt
    assert "Context:" not in prompt

def test_prompt_whitespace_context_omitted():
    prompt = build_prompt("x", ["y"], "   ")
    assert "Context:" not in prompt
