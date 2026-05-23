# llm-join

**The pandas join that understands what your data means.**

`pd.merge` joins on exact values. `llm-join` joins on *meaning* — using embeddings to find candidates and an LLM you already have to decide if they match.

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

### vs. `pd.merge`
Exact string match only. Fails on any variation in naming.

### vs. fuzzy string matching (`fuzzywuzzy`, `rapidfuzz`)
Character similarity, not semantic meaning. `"Apple Inc"` vs `"Apple Incorporated"` scores high. `"aspirin"` vs `"acetylsalicylic acid"` scores low — even though they are the same drug.

### vs. embedding similarity alone
Fast and cheap, but no reasoning. Can't explain *why* two values match or catch false positives confidently.

### llm-join
Embeddings narrow down candidates (fast, cheap). LLM makes the final call with context (accurate). You get the best of both.

---

## Real-World Use Cases

| Domain | Left table | Right table | Problem |
|--------|-----------|-------------|---------|
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

## How It Works

```
df1 (left)                    df2 (right)
"aspirin"          ──embed──▶  faiss index of right values
                   ◀──top_k──  ["Bayer Aspirin", "ibuprofen", "acetylsalicylic acid"]
                   ──llm──▶    score each candidate with context
                               best match above threshold → join row
```

1. **Embed** both columns using your `embed_fn`
2. **Retrieve** top-K candidates per row using faiss ANN search
3. **Score** each candidate with your LLM, with domain context injected into the prompt
4. **Merge** matched rows — same API as `pd.merge`

---

## Cost & Scale

### The problem with naive LLM joins

If you sent every possible pair to the LLM:

| Left rows | Right rows | Pairs to score | Cost (gpt-4o-mini ~$0.30/1M tokens) |
|-----------|------------|----------------|--------------------------------------|
| 1,000 | 10,000 | 10,000,000 | ~$150 |
| 10,000 | 100,000 | 1,000,000,000 | ~$15,000 |
| 100,000 | 1,000,000 | 100,000,000,000 | impossible |

**llm-join solves this with a two-stage pipeline.**

### Stage 1: Embeddings narrow the search (cheap)

Convert every value to a vector. Use faiss to find the top-K most similar candidates per row. This is pure math — no API calls, runs in milliseconds.

```
10,000 left rows × 100,000 right rows = 1,000,000,000 possible pairs
                                        ↓  embed + faiss (top_k=5)
                                        50,000 candidate pairs
                                        99.995% of pairs eliminated for free
```

### Stage 2: LLM scores only the hard cases (accurate)

Your LLM sees a small batch of plausible candidates per row — not the full cross product. It decides the final match with full semantic reasoning.

```
"aspirin" candidates after embedding:
  0. "Bayer Aspirin"        → LLM score: 0.95 ✓ match
  1. "acetylsalicylic acid" → LLM score: 0.98 ✓ match (even better)
  2. "aspirin 100mg tablet" → LLM score: 0.91 ✓ match
  3. "ibuprofen"            → LLM score: 0.03 ✗ skip
  4. "naproxen sodium"      → LLM score: 0.02 ✗ skip
```

### Real cost example

| Setup | LLM calls | Estimated cost |
|-------|-----------|----------------|
| 10k × 100k, top_k=5 | 50,000 | ~$0.75 |
| 10k × 100k, top_k=3 | 30,000 | ~$0.45 |
| 10k × 100k, embed_threshold=0.95 | ~5,000 (obvious matches skip LLM) | ~$0.08 |

### Further cost controls

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

| Parameter | Effect |
|-----------|--------|
| `top_k=3` (default 5) | 40% fewer LLM tokens |
| `embed_threshold=0.95` | Skip LLM for obvious matches — typically saves 30–60% |
| `max_llm_calls=N` | Budget guard — never exceeds N LLM calls |

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

### Control cost

```python
result = fuzzy_join(
    df1, df2,
    left_on="vendor",
    right_on="supplier_name",
    llm=my_llm,
    embed_threshold=0.95,   # skip LLM if embedding match is obvious
    max_llm_calls=500,      # hard cap — warns and returns partial result if hit
    top_k=3,                # fewer candidates = fewer LLM tokens
)
```

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

Pass any callable that takes a prompt string and returns a string.

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
