# llm-join

**The pandas join that understands what your data means.**

`pd.merge` joins on exact values. `llm-join` joins on *meaning* — using embeddings to find candidates and an LLM you already have to decide if they match.

---

## Table of Contents

- [The Problem](#the-problem)
- [Install](#install)
- [Quick Start](#quick-start)
- [Why llm-join](#why-llm-join)
- [Real-World Use Cases](#real-world-use-cases)
- [How It Works](#how-it-works)
- [Cost & Scale](#cost--scale)
  - [The problem with naive LLM joins](#the-problem-with-naive-llm-joins)
  - [Stage 1: Embeddings narrow the search](#stage-1-embeddings-narrow-the-search-cheap)
  - [Stage 2: LLM scores only the hard cases](#stage-2-llm-scores-only-the-hard-cases-accurate)
  - [Real cost example](#real-cost-example)
  - [Further cost controls](#further-cost-controls)
- [Usage](#usage)
  - [Basic join](#basic-join)
  - [With domain context](#with-domain-context)
  - [See why matches were made](#see-why-matches-were-made)
  - [Control cost](#control-cost)
  - [Left join (audit unmatched rows)](#left-join-audit-unmatched-rows)
  - [Multi-column join key](#multi-column-join-key)
  - [Chaining multiple joins](#chaining-multiple-joins)
- [Works with Any LLM](#works-with-any-llm)
- [Works with Any Embedding Function](#works-with-any-embedding-function)
- [vs. Alternatives](#vs-alternatives)
- [Parameters](#parameters)
- [License](#license)

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
| `MSFT Q4 license renewal` | `Microsoft Enterprise Agreement Q4-2024` |
| `Python programming` | `Python (language)` |

`pd.merge` returns nothing. Fuzzy string matching gets the wrong answer. You end up writing custom logic — or doing it by hand.

**llm-join solves this in one line.**

---

## Install

```bash
# From GitHub (not yet on PyPI)
pip install git+https://github.com/adityabalki/llm-join.git

# Or clone and install locally
git clone https://github.com/adityabalki/llm-join.git
cd llm-join
pip install -e .

# Or copy the wheel to air-gapped machines
pip install llm_join-0.1.0-py3-none-any.whl
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

# Wire up your embedding function
import numpy as np
def my_embed(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in response.data], dtype="float32")

# Join (inner by default — only rows that matched)
result = fuzzy_join(
    df1, df2,
    left_on="vendor",
    right_on="supplier_name",
    llm=llm,
    embed_fn=my_embed,
    context="company names — match legal entity variants and abbreviations",
    how="inner",   # "inner" | "left" | "right" | "outer"
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
Character similarity, not semantic meaning. `"iPhone 14 Pro"` vs `"iPhone 14 Pro Max"` scores high — but they are different products. `"CABLE-USBC-200CM-BLK"` vs `"USB-C charging cable 2m black"` scores near zero — even though they are the same item.

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
| **Retail / e-commerce** | Marketplace product listing | Master product catalog | Deduplicate listings across 50+ sellers |
| **Logistics** | Shipment description | Harmonized tariff code | Auto-classify goods at customs |
| **E-commerce** | Marketplace listing | Master product catalog | Deduplicate across platforms |
| **Research** | Author name | Citation database | Disambiguate authors |
| **Government** | Vendor name | Tax registry | Consolidate procurement spend |
| **Real estate** | Raw address input | Property records DB | Standardize and match addresses |

---

## How It Works

**Example:** A retailer's internal purchase orders use plain English product names. Their supplier sends a catalog with SKU codes. `pd.merge` matches nothing. llm-join bridges the gap.

```python
orders_df = pd.DataFrame({"product_name": [
    "USB-C charging cable 2m black",
    "ergonomic mesh office chair",
    "27-inch 4K monitor",
], "qty": [500, 30, 12]})

catalog_df = pd.DataFrame({"sku": [
    "CABLE-USBC-200CM-BLK",
    "CABLE-USBA-200CM-BLK",
    "CHAIR-MESH-ERG-ADJUSTABLE",
    "CHAIR-TASK-FIXED-BLK",
    "MON-27-4K-IPS-HDMI2",
], "unit_price": [8.99, 6.49, 349.00, 189.00, 429.00]})

result = fuzzy_join(
    orders_df, catalog_df,
    left_on="product_name", right_on="sku",
    llm=my_llm, embed_fn=my_embed,
    context="procurement — match buyer product descriptions to supplier SKU codes",
    top_k=3, threshold=0.7,
)
```

### Step 1 — Embed both columns

Every value is converted to a vector. No API call — pure math, milliseconds.

| Value | Meaning captured in vector |
|---|---|
| `"USB-C charging cable 2m black"` | cable / USB-C / length / color |
| `"ergonomic mesh office chair"` | seating / ergonomic / mesh |
| `"27-inch 4K monitor"` | display / size / resolution |
| `"CABLE-USBC-200CM-BLK"` | cable / USB-C / 200cm / black |
| `"CHAIR-MESH-ERG-ADJUSTABLE"` | seating / mesh / ergonomic |
| `"MON-27-4K-IPS-HDMI2"` | display / 27in / 4K |

### Step 2 — FAISS retrieves top-K candidates (no LLM)

For each left row, faiss finds the `top_k` closest right vectors by cosine similarity. Everything else is eliminated — no LLM call needed.

**Query: `"USB-C charging cable 2m black"` → top_k=3**

| Rank | Candidate | Embed Score | Reaches LLM? |
|---:|---|---:|---|
| 0 | `CABLE-USBC-200CM-BLK` | 0.89 | ✓ yes |
| 1 | `CABLE-USBA-200CM-BLK` | 0.71 | ✓ yes |
| 2 | `CHAIR-TASK-FIXED-BLK` | 0.34 | ✓ yes (shares "BLK") |
| — | `CHAIR-MESH-ERG-ADJUSTABLE` | 0.11 | ✗ eliminated |
| — | `MON-27-4K-IPS-HDMI2` | 0.08 | ✗ eliminated |

**Query: `"ergonomic mesh office chair"` → top_k=3**

| Rank | Candidate | Embed Score | Reaches LLM? |
|---:|---|---:|---|
| 0 | `CHAIR-MESH-ERG-ADJUSTABLE` | 0.91 | ✓ yes |
| 1 | `CHAIR-TASK-FIXED-BLK` | 0.74 | ✓ yes |
| 2 | `CABLE-USBC-200CM-BLK` | 0.19 | ✓ yes |
| — | `MON-27-4K-IPS-HDMI2` | 0.06 | ✗ eliminated |
| — | `CABLE-USBA-200CM-BLK` | 0.14 | ✗ eliminated |

**Query: `"27-inch 4K monitor"` → top_k=3**

| Rank | Candidate | Embed Score | Reaches LLM? |
|---:|---|---:|---|
| 0 | `MON-27-4K-IPS-HDMI2` | 0.94 | ✓ yes |
| 1 | `CABLE-USBC-200CM-BLK` | 0.22 | ✓ yes |
| 2 | `CHAIR-TASK-FIXED-BLK` | 0.17 | ✓ yes |
| — | `CHAIR-MESH-ERG-ADJUSTABLE` | 0.09 | ✗ eliminated |
| — | `CABLE-USBA-200CM-BLK` | 0.12 | ✗ eliminated |

**Result: 3 LLM calls instead of 3 × 5 = 15 pair-by-pair calls.**

### Step 3 — One LLM call per left row scores all candidates

All top-K candidates go into a single prompt. LLM returns a JSON array — one API call, all candidates scored.

**Prompt sent for `"USB-C charging cable 2m black"`:**
```
Context: procurement — match buyer product descriptions to supplier SKU codes

LEFT: "USB-C charging cable 2m black"

Score each candidate (0.0–1.0):
0. CABLE-USBC-200CM-BLK
1. CABLE-USBA-200CM-BLK
2. CHAIR-TASK-FIXED-BLK
```

**LLM response:**
```json
[
  {"index": 0, "score": 0.97, "reasoning": "USBC = USB-C, 200CM = 2m, BLK = black. Exact match on all three specs."},
  {"index": 1, "score": 0.38, "reasoning": "Correct length and color but USBA is USB-A, not USB-C. Wrong connector type."},
  {"index": 2, "score": 0.04, "reasoning": "This is a chair SKU. No relation to a cable."}
]
```

**Apply threshold=0.7:**

| Candidate | LLM Score | Decision |
|---|---:|---|
| `CABLE-USBC-200CM-BLK` | 0.97 | ✓ **best match — joined** |
| `CABLE-USBA-200CM-BLK` | 0.38 | ✗ below threshold |
| `CHAIR-TASK-FIXED-BLK` | 0.04 | ✗ below threshold |

### Step 4 — Merge matched rows

```python
print(result)
```

| product_name | qty | sku | unit_price |
|---|---:|---|---:|
| USB-C charging cable 2m black | 500 | CABLE-USBC-200CM-BLK | 8.99 |
| ergonomic mesh office chair | 30 | CHAIR-MESH-ERG-ADJUSTABLE | 349.00 |
| 27-inch 4K monitor | 12 | MON-27-4K-IPS-HDMI2 | 429.00 |

The LLM correctly rejected `CABLE-USBA-200CM-BLK` (wrong connector) even though embedding scored it 0.71 — this is the case where LLM reasoning earns its cost.

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

| | |
|---|---|
| Input pairs | 10,000 × 100,000 = **1,000,000,000** |
| After embed + faiss (`top_k=5`) | **50,000** candidate pairs |
| Eliminated for free | **99.995%** of all pairs |

### Stage 2: LLM scores only the hard cases (accurate)

Your LLM sees a small batch of plausible candidates per row — not the full cross product. It scores each candidate; the highest score above `threshold` wins. One best match per left row.

Query: `"USB-C charging cable 2m black"`

| Rank | Candidate | LLM Score | Decision |
|---:|---|---:|---|
| 0 | `CABLE-USBC-200CM-BLK` | 0.97 | ✓ **best match — joined** |
| 1 | `CABLE-USBC-100CM-BLK` | 0.71 | scored, not selected (wrong length) |
| 2 | `CABLE-USBA-200CM-BLK` | 0.38 | scored, not selected (wrong connector) |
| 3 | `CHAIR-TASK-FIXED-BLK` | 0.04 | ✗ below threshold |
| 4 | `MON-27-4K-IPS-HDMI2` | 0.01 | ✗ below threshold |

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
    orders_df, catalog_df,
    left_on="product_name",
    right_on="sku",
    llm=my_llm,
    embed_fn=my_embed,
    context="procurement — match buyer product descriptions to supplier SKU codes",
)
```

### With domain context

```python
result = fuzzy_join(
    orders_df, catalog_df,
    left_on="product_name",
    right_on="sku",
    llm=my_llm,
    embed_fn=my_embed,
    context="procurement — match buyer product descriptions to supplier SKU codes",
    column_context={
        "product_name": "plain English product description written by a buyer",
        "sku": "supplier stock-keeping unit code, typically uppercase with hyphens",
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

print(result[["vendor", "supplier_name", "_llm_score", "_llm_reasoning", "_embed_rank", "_match_method"]])
```

| vendor | supplier_name | _llm_score | _llm_reasoning | _embed_rank | _match_method |
|---|---|---:|---|---:|---|
| Goldman Sachs & Co. | The Goldman Sachs Group Inc | 0.97 | same firm, legal name variant | 0 | llm |

### Control cost

```python
result = fuzzy_join(
    df1, df2,
    left_on="vendor",
    right_on="supplier_name",
    llm=my_llm,
    embed_fn=my_embed,
    embed_threshold=0.95,   # skip LLM if embedding match is obvious
    max_llm_calls=500,      # hard cap — warns and returns partial result if hit
    top_k=3,                # fewer candidates = fewer LLM tokens
)
```

### Left join (audit unmatched rows)

`how="left"` keeps all left rows — unmatched ones get NaN right columns. Useful to see what the LLM failed to match.

```python
result = fuzzy_join(df1, df2, left_on="a", right_on="b", llm=my_llm, embed_fn=my_embed, how="left")

# Rows with no match
unmatched = result[result["b"].isna()]
```

> **Note:** `how="outer"` is useful for reconciliation — unmatched left rows are left values with no match above threshold; unmatched right rows are right values that were never selected as a best match for any left row. `cross` join is not supported (it would be the naive O(n×m) approach that llm-join is designed to avoid).

### Multi-column join key

```python
# orders_df has separate "product_name" and "category" columns
# catalog_df has a single "sku_description" column like "CABLE / USB-C / 200CM / BLK"

result = fuzzy_join(
    orders_df, catalog_df,
    left_on=["product_name", "category"],   # concatenated: "USB-C cable 2m · electronics"
    right_on="sku_description",
    llm=my_llm,
    embed_fn=my_embed,
    context="procurement — match buyer product + category to supplier SKU description",
)
```

### Chaining multiple joins

Each `fuzzy_join` returns a regular DataFrame — pipe them like `pd.merge`.

```python
# df1: transactions  (vendor + product columns)
# df2: vendor master
# df3: product catalog

# Step 1 — match vendors
step1 = fuzzy_join(
    df1, df2,
    left_on="vendor",
    right_on="supplier_name",
    llm=my_llm,
    embed_fn=my_embed,
    how="left",
    return_reasoning=True,
)
step1 = step1.rename(columns={
    "_llm_score":    "_vendor_score",
    "_llm_reasoning": "_vendor_reasoning",
    "_match_method": "_vendor_method",
})

# Step 2 — match products on result of step 1
result = fuzzy_join(
    step1, df3,
    left_on="product",
    right_on="catalog_item",
    llm=my_llm,
    embed_fn=my_embed,
    how="left",
    return_reasoning=True,
)
# result now has _vendor_score + _llm_score without column collision
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

## Works with Any Embedding Function

Pass any callable `(list[str]) -> np.ndarray` (shape `[n, dim]`, dtype `float32`).

```python
import numpy as np

# OpenAI embeddings
import openai
client = openai.OpenAI()
def my_embed(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in response.data], dtype="float32")

# Cohere
import cohere
co = cohere.Client("YOUR_KEY")
def my_embed(texts):
    response = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
    return np.array(response.embeddings, dtype="float32")

# sentence-transformers (local)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def my_embed(texts):
    return model.encode(texts, convert_to_numpy=True).astype("float32")

# Any custom endpoint
import requests
def my_embed(texts):
    response = requests.post(
        "https://your-embed-api.com/embed",
        json={"texts": texts}
    ).json()
    return np.array(response["embeddings"], dtype="float32")
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
| `context` | required | Domain context injected into LLM prompt — describe what the columns represent and what kind of match to make |
| `column_context` | `{}` | Per-column context dict `{"col": "description"}` |
| `top_k` | `5` | Embedding candidates retrieved per row before LLM scoring |
| `batch_size` | `32` | Reserved for future LLM batching (passed through to config) |
| `threshold` | `0.7` | Minimum LLM score (0–1) to accept a match |
| `how` | `"inner"` | Join type: `inner` (matched pairs only) / `left` (all left rows, NaN where unmatched) / `right` (all right rows, NaN where no left row matched them) / `outer` (full picture — both unmatched left and unmatched right rows). |
| `embed_threshold` | `None` | Skip LLM when embedding score is decisive (saves cost) |
| `max_llm_calls` | `None` | Hard cap on LLM calls — returns partial result with warning if hit |
| `max_retries` | `3` | Retry failed LLM calls with exponential backoff (1s, 2s, 4s…). Set `0` to disable. |
| `return_reasoning` | `False` | Append `_llm_score`, `_llm_reasoning`, `_embed_rank`, `_match_method` columns. `_match_method` is `"llm"` or `"embed_threshold"` — tells you which rows skipped the LLM. |

---

## License

MIT
