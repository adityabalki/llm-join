_TEMPLATE = """\
You are a data matching assistant.
{context_line}For each pair below, score how likely LEFT matches RIGHT (0.0–1.0).
Respond ONLY as a JSON array with no extra text:
[{{"index": 0, "score": 0.95, "reasoning": "brief explanation"}}, ...]

Pairs:
{pairs}"""


def build_prompt(left_val: str, candidates: list[str], context_str: str) -> str:
    context_line = f"Context: {context_str}\n" if context_str.strip() else ""
    pairs = "\n".join(
        f'{i}. LEFT: "{left_val}" | RIGHT: "{c}"'
        for i, c in enumerate(candidates)
    )
    return _TEMPLATE.format(context_line=context_line, pairs=pairs)
