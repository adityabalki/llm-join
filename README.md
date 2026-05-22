# llm-join

Fuzzy join DataFrames using LLM scoring and embedding-based retrieval.

## Install

```bash
pip install llm-join
```

## Quick Start

```python
import openai
from llm_join import fuzzy_join

client = openai.OpenAI()

def llm(prompt):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

result = fuzzy_join(
    df1, df2,
    left_on="sku",
    right_on="supplier_sku",
    llm=llm,
    context="product SKUs from electronics catalog",
)
```

## Use Cases

Works anywhere values are semantically equivalent but textually different:

| Domain | Example |
|--------|---------|
| Supply chain | `"Sony WH-1000XM5"` ↔ `"SONY-WH1000XM5-BLK"` |
| Finance | `"AMZN"` ↔ `"Amazon Web Services Inc."` |
| Legal | `"Goldman Sachs & Co."` ↔ `"The Goldman Sachs Group Inc"` |
| Compliance | Customer names ↔ OFAC sanctions list |
| Pharma | Generic drug names ↔ brand names |
| HR | Resume skills ↔ skills taxonomy |

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `left_on` | required | Column(s) in df1 |
| `right_on` | required | Column(s) in df2 |
| `llm` | required | Callable `(prompt: str) -> str` |
| `context` | `""` | Global domain context string |
| `column_context` | `{}` | Per-column context `{"col": "description"}` |
| `top_k` | `5` | Embedding candidates per row |
| `threshold` | `0.7` | Min LLM score to accept match |
| `how` | `"inner"` | Join type: inner/left/right/outer |
| `embed_model` | `"all-MiniLM-L6-v2"` | sentence-transformers model |
| `embed_fn` | `None` | Custom embed callable `(list[str]) -> ndarray` |
| `batch_size` | `32` | LLM pairs per prompt |
| `embed_threshold` | `None` | Skip LLM if embed score is decisive |
| `max_llm_calls` | `None` | Hard cap on LLM calls |
| `return_reasoning` | `False` | Add `_llm_score`, `_llm_reasoning`, `_embed_rank` columns |

## Works with any LLM

```python
# Anthropic
import anthropic
client = anthropic.Anthropic()
llm = lambda p: client.messages.create(
    model="claude-opus-4-7", max_tokens=1024,
    messages=[{"role": "user", "content": p}]
).content[0].text

# Ollama (local)
import ollama
llm = lambda p: ollama.chat(
    model="llama3", messages=[{"role": "user", "content": p}]
)["message"]["content"]
```
