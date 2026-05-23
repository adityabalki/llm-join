# llm-join

**The pandas join that understands what your data means.**

`pd.merge` joins on exact values. `llm-join` joins on *meaning* — embeddings find candidates, your LLM decides if they match.

```python
from llm_join import fuzzy_join

result = fuzzy_join(df1, df2, left_on="vendor", right_on="supplier_name", llm=my_llm, embed_fn=my_embed)
```

---

## The Problem

You have two DataFrames. Same data, different text:

| Your system | Their system |
|-------------|--------------|
| `Goldman Sachs & Co.` | `The Goldman Sachs Group Inc` |
| `Sony WH-1000XM5` | `SONY-WH1000XM5-BLK` |
| `aspirin 100mg` | `Bayer Aspirin Tablet` |
| `Python programming` | `Python (language)` |

`pd.merge` returns nothing. Fuzzy string matching gets the wrong answer. You end up writing custom logic — or doing it by hand.

**llm-join solves this in one line.**

---

## Install

```bash
pip install llm-join
```

---

## Quick Start

```python
import pandas as pd
import openai
from llm_join import fuzzy_join

# Your data
df1 = pd.DataFrame({
    "vendor": ["Goldman Sachs & Co.", "Amazon Web Services", "Microsoft Corp"],
    "spend": [1_200_000, 890_000, 340_000]
})

df2 = pd.DataFrame({
    "supplier_name": ["The Goldman Sachs Group Inc", "Amazon.com Inc.", "Microsoft Corporation"],
    "category": ["Finance", "Cloud", "Software"]
})

# Wire up any LLM you already use
client = openai.OpenAI()
def llm(prompt):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# Join
result = fuzzy_join(
    df1, df2,
    left_on="vendor",
    right_on="supplier_name",
    llm=llm,
    context="company names — match legal entity variants and abbreviations",
)

print(result)
```

| vendor | spend | supplier_name | category |
|---|---:|---|---|
| Goldman Sachs & Co. | 1,200,000 | The Goldman Sachs Group Inc | Finance |
| Amazon Web Services | 890,000 | Amazon.com Inc. | Cloud |
| Microsoft Corp | 340,000 | Microsoft Corporation | Software |

---

## Why llm-join

| Tool | Matching method | Semantic understanding | Reasoning output | Bring your own model |
|------|----------------|----------------------|-----------------|---------------------|
| `pd.merge` | Exact string match | ✗ | ✗ | n/a |
| `fuzzywuzzy` / `rapidfuzz` | Character similarity | ✗ | ✗ | n/a |
| Embedding similarity alone | Vector cosine distance | ✓ | ✗ | partial |
| **llm-join** | **Embeddings + LLM decision** | **✓** | **✓** | **✓** |

`"aspirin"` vs `"acetylsalicylic acid"` — fuzzywuzzy scores this low. llm-join scores it 0.98. The LLM knows they're the same drug.

---

## How It Works

llm-join uses a **two-stage pipeline** — embeddings eliminate 99.99% of pairs for free, the LLM only sees plausible candidates.

### Stage 1 — Embed & Retrieve (cheap, milliseconds)

| Step | What happens | Cost |
|------|-------------|------|
| Embed left column | Convert all left values to vectors via your `embed_fn` | your embed API |
| Embed right column | Convert all right values to vectors (once, cached) | your embed API |
| FAISS search | For each left row, find top-K nearest right rows by cosine similarity | free — pure math |

```
10,000 left rows × 100,000 right rows = 1,000,000,000 possible pairs
                                        ↓  embed + faiss (top_k=5)
                                        50,000 candidate pairs
                                        99.995% of pairs eliminated for free
```

### Stage 2 — LLM Scores the Candidates (accurate, reasoned)

Your LLM sees a small batch of plausible candidates per row — not the full cross product.

```
Query: "aspirin"

Candidate              | Embed rank | LLM score | Decision
-----------------------|------------|-----------|----------
acetylsalicylic acid   |     1      |   0.98    | ✓ match
Bayer Aspirin Tablet   |     2      |   0.95    | ✓ match
aspirin 100mg tablet   |     3      |   0.91    | ✓ match
ibuprofen              |     4      |   0.03    | ✗ skip
naproxen sodium        |     5      |   0.02    | ✗ skip
```

---

## Cost & Scale

### Naive approach: every pair to the LLM

| Left rows | Right rows | Pairs to score | Cost (gpt-4o-mini ~$0.30/1M tokens) |
|----------:|----------:|--------------:|------------------------------------:|
| 1,000 | 10,000 | 10,000,000 | ~$150 |
| 10,000 | 100,000 | 1,000,000,000 | ~$15,000 |
| 100,000 | 1,000,000 | 100,000,000,000 | impossible |

### llm-join two-stage pipeline

| Setup | LLM calls | Estimated cost | vs. naive |
|-------|----------:|---------------:|----------:|
| 10k × 100k, `top_k=5` | 50,000 | ~$0.75 | **−99.995%** |
| 10k × 100k, `top_k=3` | 30,000 | ~$0.45 | **−99.997%** |
| 10k × 100k, `embed_threshold=0.95` | ~5,000 | ~$0.08 | **−99.9995%** |

### Cost control parameters

| Parameter | Default | Effect |
|-----------|--------:|--------|
| `top_k=3` | 5 | 40% fewer LLM tokens per join |
| `embed_threshold=0.95` | None | Skip LLM for obvious matches — saves 30–60% typically |
| `max_llm_calls=N` | None | Hard budget cap — warns and returns partial result if hit |

```python
result = fuzzy_join(
    df1, df2,
    left_on="vendor", right_on="supplier",
    llm=my_llm,
    embed_fn=my_embed,
    top_k=3,                # fewer candidates = fewer LLM tokens per row
    embed_threshold=0.95,   # skip LLM entirely if embedding match score > 0.95
    max_llm_calls=1000,     # hard cap — warns and returns partial result if hit
)
```

---

## Real-World Use Cases

| Domain | Left table | Right table | Problem solved |
|--------|-----------|-------------|----------------|
| **Supply chain** | Buyer catalog SKU | Supplier SKU | Match products across 50+ vendor catalogs |
| **Finance** | Expense report payee | GL account / vendor master | Reconcile transactions automatically |
| **Legal / M&A** | Contract party name | Corporate registry | Identify true legal entity |
| **Compliance** | Customer name | OFAC sanctions list | Sanctions screening at scale |
| **Pharma** | Generic drug name | Brand name | Generic ↔ brand equivalence |
| **HR** | Resume job title | Standard taxonomy | Normalize titles for comp analysis |
| **E-commerce** | Marketplace listing | Master product catalog | Deduplicate across platforms |
| **Research** | Author name | Citation database | Disambiguate authors |
| **Government** | Vendor name | Tax registry | Consolidate procurement spend |
| **Real estate** | Raw address input | Property records DB | Standardize and match addresses |

---

## Usage

### Basic join

```python
result = fuzzy_join(
    df1, df2,
    left_on="drug_name",
    right_on="brand_name",
    llm=my_llm,
    embed_fn=my_embed,
)
```

### With domain context

```python
result = fuzzy_join(
    df1, df2,
    left_on="drug_name",
    right_on="brand_name",
    llm=my_llm,
    embed_fn=my_embed,
    context="pharmaceutical drug names — match generic to brand equivalents",
    column_context={
        "drug_name": "generic drug name (e.g. aspirin, ibuprofen)",
        "brand_name": "US brand name sold in pharmacies",
    },
)
```

### See why matches were made

```python
result = fuzzy_join(
    df1, df2,
    left_on="vendor",
    right_on="supplier_name",
    llm=my_llm,
    embed_fn=my_embed,
    return_reasoning=True,
)

print(result[["vendor", "supplier_name", "_llm_score", "_llm_reasoning"]])
```

| vendor | supplier_name | _llm_score | _llm_reasoning |
|---|---|---:|---|
| Goldman Sachs & Co. | The Goldman Sachs Group Inc | 0.97 | same firm, legal name variant |

### Left join (keep unmatched rows)

```python
result = fuzzy_join(df1, df2, left_on="a", right_on="b", llm=my_llm, embed_fn=my_embed, how="left")
```

### Multi-column join key

```python
result = fuzzy_join(
    df1, df2,
    left_on=["drug", "dosage_form"],   # concatenated as join key
    right_on="product_description",
    llm=my_llm,
    embed_fn=my_embed,
)
```

---

## Works with Any LLM

Pass any callable `(prompt: str) -> str`. No provider lock-in.

```python
# OpenAI
import openai
client = openai.OpenAI()
llm = lambda p: client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": p}]
).choices[0].message.content

# Anthropic
import anthropic
client = anthropic.Anthropic()
llm = lambda p: client.messages.create(
    model="claude-opus-4-7", max_tokens=512,
    messages=[{"role": "user", "content": p}]
).content[0].text

# Google Gemini
import google.generativeai as genai
model = genai.GenerativeModel("gemini-2.0-flash")
llm = lambda p: model.generate_content(p).text

# Ollama (local, free)
import ollama
llm = lambda p: ollama.chat(
    model="llama3.2", messages=[{"role": "user", "content": p}]
)["message"]["content"]

# Any custom endpoint
import requests
llm = lambda p: requests.post(
    "https://your-llm-api.com/chat",
    json={"prompt": p}
).json()["response"]
```

Same pattern for `embed_fn` — pass any callable `(list[str]) -> np.ndarray`. Works with OpenAI embeddings, Cohere, local models, or any custom endpoint.

---

## vs. Alternatives

| | llm-join | pd.merge | fuzzywuzzy | Jellyjoin | LinkTransformer |
|---|:---:|:---:|:---:|:---:|:---:|
| Semantic matching | ✓ | ✗ | ✗ | ✓ | ✓ |
| LLM makes final decision | ✓ | ✗ | ✗ | ✗ | optional |
| Reasoning per match | ✓ | ✗ | ✗ | ✗ | ✗ |
| Domain context injection | ✓ | ✗ | ✗ | ✗ | ✗ |
| Bring-your-own LLM callable | ✓ | n/a | n/a | ✗ | ✗ |
| Bring-your-own embed callable | ✓ | n/a | n/a | ✗ | ✗ |
| Enterprise-safe (no forced downloads) | ✓ | ✓ | ✓ | partial | ✗ |
| Hard cost cap (`max_llm_calls`) | ✓ | n/a | n/a | ✗ | ✗ |
| License | **MIT** | BSD | MIT | MIT | **GPL-3.0** |
| Install size | ~50 MB | ~20 MB | ~30 MB | ~60 MB | **~3 GB** |
| String distance fallback | ✗ | ✗ | ✓ | ✓ | ✗ |
| Dedup / clustering | ✗ | ✗ | ✗ | ✗ | ✓ |

> **GPL-3.0** (LinkTransformer) restricts use in closed-source products.  
> **~3 GB** install (LinkTransformer) = torch + transformers + sentence-transformers + wandb + full HuggingFace stack.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `left_on` | required | Column name(s) in df1 |
| `right_on` | required | Column name(s) in df2 |
| `llm` | required | Callable `(prompt: str) -> str` — your LLM function |
| `embed_fn` | required | Callable `(list[str]) -> np.ndarray` — your embedding function |
| `context` | `""` | Global domain context injected into LLM prompt |
| `column_context` | `{}` | Per-column context dict `{"col": "description"}` |
| `top_k` | `5` | Embedding candidates retrieved per row before LLM scoring |
| `threshold` | `0.7` | Minimum LLM score (0–1) to accept a match |
| `how` | `"inner"` | Join type: `inner` / `left` / `right` / `outer` |
| `embed_threshold` | `None` | Skip LLM when embedding score is decisive (saves cost) |
| `max_llm_calls` | `None` | Hard cap on LLM calls — returns partial result with warning if hit |
| `return_reasoning` | `False` | Append `_llm_score`, `_llm_reasoning`, `_embed_rank` columns |

---

## License

MIT
