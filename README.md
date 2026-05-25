# llm-join

[![PyPI](https://img.shields.io/pypi/v/llm-join)](https://pypi.org/project/llm-join/) [![GitHub](https://img.shields.io/badge/GitHub-adityabalki%2Fllm--join-blue?logo=github)](https://github.com/adityabalki/llm-join)

**The pandas join that understands what your data means.**

`pd.merge` joins on exact values. `llm-join` joins on meaning using embeddings to find candidates and an LLM you already have to decide if they match.

```python
from llm_join import fuzzy_join

result = fuzzy_join(
    df1, df2,
    left_on="vendor", right_on="supplier_name",
    llm=my_llm, embed_fn=my_embed,
    context="company names",
    llm_concurrency=10,
)
```

---

## Table of Contents

- [The Problem](#the-problem)
- [Why llm-join](#why-llm-join)
- [Real-World Use Cases](#real-world-use-cases)
- [Install](#install)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Cost & Scale](#cost--scale)
- [Performance](#performance)
- [Usage](#usage)
  - [Basic join](#basic-join)
  - [With domain context](#with-domain-context)
  - [See why matches were made](#see-why-matches-were-made)
  - [Control cost](#control-cost)
  - [Left join (audit unmatched rows)](#left-join-audit-unmatched-rows)
  - [Multi-column join key](#multi-column-join-key)
  - [Match all results](#match-all-results)
  - [Chaining multiple joins](#chaining-multiple-joins)
- [Works with Any LLM](#works-with-any-llm)
- [Works with Any Embedding Function](#works-with-any-embedding-function)
- [Features](#features)
- [Parameters](#parameters)
- [License](#license)

---

## The Problem

You have two DataFrames. Same data, different text:

| Your system | Their system |
|-------------|--------------|
| `Goldman Sachs & Co.` | `The Goldman Sachs Group Inc` |
| `Sony WH-1000XM5` | `SONY-WH1000XM5-BLK` |
| `MSFT Q4 license renewal` | `Microsoft Enterprise Agreement Q4-2024` |
| `Python programming` | `Python (language)` |

`pd.merge` returns nothing. Fuzzy string matching gets the wrong answer. You end up writing custom logic  or doing it by hand.

**llm-join solves this in one function call.**

---

## Why llm-join

### vs. `pd.merge`
Exact string match only. Fails on any variation in naming.

### vs. fuzzy string matching (`fuzzywuzzy`, `rapidfuzz`)
Character similarity, not meaning. `"iPhone 14 Pro"` vs `"iPhone 14 Pro Max"` scores high but they are different products. `"CABLE-USBC-200CM-BLK"` vs `"USB-C charging cable 2m black"` scores near zero even though they are the same item.

### vs. embedding similarity alone
Fast and cheap, but no reasoning. Can't explain why two values match or catch false positives confidently.

### llm-join
Embeddings narrow the field cheaply. Your model scores only the plausible candidates with context you provide. Accurate where it matters, fast where it doesn't.

**Works with any provider:** OpenAI, Azure, Anthropic, Bedrock, Ollama, or any private model. No data leaves your infrastructure beyond what your own providers already see.

---

## Real-World Use Cases

| Domain | Left table | Right table | Problem |
|--------|-----------|-------------|---------|
| **Supply chain** | Buyer product description | Supplier SKU | Match products across 50+ vendor catalogs |
| **Finance** | Expense report payee | GL account / vendor master | Reconcile transactions automatically |
| **Legal / M&A** | Contract party name | Corporate registry | Identify true legal entity |
| **Compliance** | Customer name | OFAC sanctions list | Sanctions screening at scale |
| **Retail / e-commerce** | Marketplace product listing | Master product catalog | Deduplicate listings across 50+ sellers |
| **Logistics** | Shipment description | Harmonized tariff code | Auto-classify goods at customs |
| **Research** | Author name | Citation database | Disambiguate authors |
| **Government** | Vendor name | Tax registry | Consolidate procurement spend |
| **Real estate** | Raw address input | Property records DB | Standardize and match addresses |
| **Pharma** | Drug compound names in adverse event reports | WHO drug dictionary (WHO-DD) | Standardize brand, generic, and abbreviation variants "Tylenol 500mg", "acetaminophen", "APAP" all resolve to the same WHO entry for pharmacovigilance signal detection |
| **HR** | Job title in offer letter or LinkedIn import | Internal job architecture / grade catalog | Normalize "Senior Software Engineer II", "Sr. SWE", "L5 Engineer" to canonical job families for compensation benchmarking and workforce analytics |

---

## Install

```bash
pip install llm-join
```

Or from source:

```bash
git clone https://github.com/adityabalki/llm-join.git
cd llm-join
pip install -e .
```

---

## Quick Start

```python
import pandas as pd
import numpy as np
import openai
from llm_join import fuzzy_join

df1 = pd.DataFrame({
    "vendor": ["Goldman Sachs & Co.", "Amazon Web Services", "Microsoft Corp"],
    "spend": [1_200_000, 890_000, 340_000]
})

df2 = pd.DataFrame({
    "supplier_name": ["The Goldman Sachs Group Inc", "Amazon.com Inc.", "Microsoft Corporation"],
    "category": ["Finance", "Cloud", "Software"]
})

client = openai.OpenAI()

def my_llm(prompt):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def my_embed(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in response.data], dtype="float32")

result = fuzzy_join(
    df1, df2,
    left_on="vendor",
    right_on="supplier_name",
    llm=my_llm,
    embed_fn=my_embed,
    context="company names — match legal entity variants and abbreviations",
    llm_concurrency=10,   # how many LLM calls to run in parallel
    how="inner",
)

print(result)
```

| vendor | spend | supplier_name | category |
|---|---:|---|---|
| Goldman Sachs & Co. | 1,200,000 | The Goldman Sachs Group Inc | Finance |
| Amazon Web Services | 890,000 | Amazon.com Inc. | Cloud |
| Microsoft Corp | 340,000 | Microsoft Corporation | Software |

---

## How It Works

**Example:** A retailer's purchase orders use plain English product names. Their supplier sends a catalog with SKU codes. `pd.merge` matches nothing. llm-join bridges the gap.

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
    context="procurement match buyer product descriptions to supplier SKU codes",
    top_k=3, llm_threshold=0.7,
    llm_concurrency=10,
)
```

### Step 1 — Embed both columns

Every value gets converted to a vector.

| Value | Meaning captured |
|---|---|
| `"USB-C charging cable 2m black"` | cable / USB-C / length / color |
| `"ergonomic mesh office chair"` | seating / ergonomic / mesh |
| `"27-inch 4K monitor"` | display / size / resolution |
| `"CABLE-USBC-200CM-BLK"` | cable / USB-C / 200cm / black |
| `"CHAIR-MESH-ERG-ADJUSTABLE"` | seating / mesh / ergonomic |
| `"MON-27-4K-IPS-HDMI2"` | display / 27in / 4K |

### Step 2 — FAISS retrieves top-K candidates per row (no LLM)

For each left row, FAISS finds the `top_k` closest right vectors using cosine similarity. Everything else is eliminated no LLM call needed.

**Query: `"USB-C charging cable 2m black"` → top_k=3**

| Rank | Candidate | Embed Score | Reaches LLM? |
|---:|---|---:|---|
| 0 | `CABLE-USBC-200CM-BLK` | 0.89 | yes |
| 1 | `CABLE-USBA-200CM-BLK` | 0.71 | yes |
| 2 | `CHAIR-TASK-FIXED-BLK` | 0.34 | yes (shares "BLK") |
| — | `CHAIR-MESH-ERG-ADJUSTABLE` | 0.11 | eliminated |
| — | `MON-27-4K-IPS-HDMI2` | 0.08 | eliminated |

**Result: 3 LLM calls instead of 3 × 5 = 15 pair-by-pair calls.**

### Step 3 — One LLM call per left row scores all candidates

All top-K candidates go into a single prompt. The LLM scores each one and returns JSON one API call per row, all candidates scored together.

**Prompt sent for `"USB-C charging cable 2m black"`:**
```
Context: procurement match buyer product descriptions to supplier SKU codes

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

**Apply llm_threshold=0.7:**

| Candidate | LLM Score | Decision |
|---|---:|---|
| `CABLE-USBC-200CM-BLK` | 0.97 | best match joined |
| `CABLE-USBA-200CM-BLK` | 0.38 | below llm_threshold |
| `CHAIR-TASK-FIXED-BLK` | 0.04 | below llm_threshold |

### Step 4 — Merge matched rows

```python
print(result)
```

| product_name | qty | sku | unit_price |
|---|---:|---|---:|
| USB-C charging cable 2m black | 500 | CABLE-USBC-200CM-BLK | 8.99 |
| ergonomic mesh office chair | 30 | CHAIR-MESH-ERG-ADJUSTABLE | 349.00 |
| 27-inch 4K monitor | 12 | MON-27-4K-IPS-HDMI2 | 429.00 |

The LLM correctly rejected `CABLE-USBA-200CM-BLK` (wrong connector) even though the embedding scored it 0.71. That's where the LLM earns its cost.

---

## Cost & Scale

If you sent every possible pair to the LLM:

| Left rows | Right rows | Pairs to score | Cost (gpt-4o-mini) |
|-----------|------------|----------------|---------------------|
| 1,000 | 10,000 | 10,000,000 | ~$150 |
| 10,000 | 100,000 | 1,000,000,000 | ~$15,000 |
| 100,000 | 1,000,000 | 100,000,000,000 | not feasible |

llm-join avoids this with a two stage pipeline.

### Stage 1 — Embeddings narrow the search (cheap)

FAISS finds the top-K most similar candidates per row. Pure math, no API calls.

| | |
|---|---|
| Input pairs | 10,000 × 100,000 = **1,000,000,000** |
| After FAISS (`top_k=5`) | **50,000** candidate pairs |
| Eliminated for free | **99.995%** of all pairs |

### Stage 2 — LLM scores only the shortlist (accurate)

Your LLM sees a small batch of plausible candidates not the full cross product. One call per left row, all candidates scored in that call.

| Setup | LLM calls | Estimated cost |
|-------|-----------|----------------|
| 10k × 100k, top_k=5 | 50,000 | ~$0.75 |
| 10k × 100k, top_k=3 | 30,000 | ~$0.45 |
| 10k × 100k, embed_skip_threshold=0.95 | ~5,000 | ~$0.08 |

### Cost controls

```python
result = fuzzy_join(
    df1, df2,
    left_on="vendor", right_on="supplier",
    llm=my_llm, embed_fn=my_embed,
    context="...",
    llm_concurrency=10,
    top_k=3,                    # fewer candidates = fewer LLM tokens per row
    embed_skip_threshold=0.95,  # skip LLM entirely if embed score is high enough
    max_llm_calls=1000,         # hard cap — warns and returns partial result if hit
)
```

---

## Performance

Set `llm_concurrency` based on your API rate limit.

```python
# Sequential use for debugging or if you have strict rate limits
fuzzy_join(..., llm_concurrency=1)

# Good starting point for most APIs
fuzzy_join(..., llm_concurrency=10)

# High throughput — check your rate limits first
fuzzy_join(..., llm_concurrency=50)
```

- Sync function (`def my_llm`) → `ThreadPoolExecutor`
- Async function (`async def my_llm`) → `asyncio` + semaphore

If a call fails after all retries, the top embedding match is used as fallback.

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
    context="procurement match buyer product descriptions to supplier SKU codes",
    llm_concurrency=10,
)
```

### With domain context

Give the LLM more detail about each column to improve accuracy.

```python
result = fuzzy_join(
    orders_df, catalog_df,
    left_on="product_name",
    right_on="sku",
    llm=my_llm,
    embed_fn=my_embed,
    context="procurement match buyer product descriptions to supplier SKU codes",
    column_context={
        "product_name": "plain English product description written by a buyer",
        "sku": "supplier stock keeping unit code, typically uppercase with hyphens",
    },
    llm_concurrency=10,
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
    context="company names match legal entity variants",
    llm_concurrency=10,
    return_reasoning=True,
)

print(result[["vendor", "supplier_name", "_llm_score", "_llm_reasoning", "_match_method", "_llm_candidates"]])
```

| vendor | supplier_name | _llm_score | _llm_reasoning | _match_method | _llm_candidates |
|---|---|---:|---|---|---|
| Goldman Sachs & Co. | The Goldman Sachs Group Inc | 0.97 | same firm, legal name variant | llm | [{"candidate": "The Goldman Sachs...", "embed_score": 0.94}, ...] |

`_llm_candidates` shows exactly which candidates were sent to the LLM and their embedding scores useful for tuning `top_k` and `threshold`.

### Left join (audit unmatched rows)

`how="left"` keeps all left rows. Unmatched ones get NaN in the right columns useful for finding what failed to match.

```python
result = fuzzy_join(
    df1, df2,
    left_on="vendor", right_on="supplier_name",
    llm=my_llm, embed_fn=my_embed,
    context="company names — match legal entity variants",
    llm_concurrency=10,
    how="left",
)

unmatched = result[result["supplier_name"].isna()]
```

> `how="full"` is useful for reconciliation unmatched left rows have no match above threshold; unmatched right rows were never selected as a best match.

### Multi-column join key

Pass a list to `left_on` or `right_on` values are concatenated automatically. Works for any number of columns.

```python
# Match on product name + category combined
result = fuzzy_join(
    orders_df, catalog_df,
    left_on=["product_name", "category"],
    right_on="sku_description",
    llm=my_llm,
    embed_fn=my_embed,
    context="procurement match buyer product + category to supplier SKU description",
    llm_concurrency=10,
)
```

### Match all results

By default, only the best-scoring match above threshold is returned. Set `match_all=True` when one left value legitimately maps to multiple right values.

```python
# "aspirin" should match all its known names
result = fuzzy_join(
    drugs_df, synonyms_df,
    left_on="generic_name", right_on="name",
    llm=my_llm, embed_fn=my_embed,
    context="pharmaceutical drug names match generics to all known synonyms",
    llm_concurrency=10,
    match_all=True,
    top_k=10,
)
```

| generic_name | name | _llm_score |
|---|---|---:|
| aspirin | acetylsalicylic acid | 0.98 |
| aspirin | Bayer Aspirin Tablet | 0.95 |
| aspirin | aspirin 100mg tablet | 0.91 |

### Chaining multiple joins

Each `fuzzy_join` returns a regular DataFrame pipe them like `pd.merge`.

```python
# Step 1 — match vendors
step1 = fuzzy_join(
    df1, df2,
    left_on="vendor", right_on="supplier_name",
    llm=my_llm, embed_fn=my_embed,
    context="company names — match legal entity variants",
    llm_concurrency=10,
    how="left", return_reasoning=True,
)

# Step 2 — match products on result of step 1
result = fuzzy_join(
    step1, df3,
    left_on="product", right_on="catalog_item",
    llm=my_llm, embed_fn=my_embed,
    context="product names — match internal descriptions to catalog items",
    llm_concurrency=10,
    how="left", return_reasoning=True,
)
```

---

## Works with Any LLM

Pass any callable that takes a prompt string and returns a string.

```python
# OpenAI
import openai
client = openai.OpenAI()
def my_llm(prompt):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# Anthropic
import anthropic
client = anthropic.Anthropic()
def my_llm(prompt):
    return client.messages.create(
        model="claude-opus-4-5", max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

# Google Gemini
import google.generativeai as genai
model = genai.GenerativeModel("gemini-2.0-flash")
def my_llm(prompt):
    return model.generate_content(prompt).text

# Ollama (local, free)
import ollama
def my_llm(prompt):
    return ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": prompt}]
    )["message"]["content"]

# Async LLM (uses asyncio path automatically)
async def my_llm(prompt):
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---

## Works with Any Embedding Function

Pass any callable `(list[str]) -> np.ndarray` (shape `[n, dim]`, dtype `float32`).

```python
import numpy as np

# OpenAI
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

# sentence-transformers (local, no API key needed)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def my_embed(texts):
    return model.encode(texts, convert_to_numpy=True).astype("float32")
```

---

## Features

- **Any provider** — OpenAI, Azure, Anthropic, Bedrock, Vertex, Ollama, or any private model. Sync and async supported.
- **Enterprise friendly** - User supply the LLM and embedding calls. Your data stays within your own infrastructure and chosen providers.
- **Two-stage pipeline** — embeddings eliminate 99%+ of pairs; your model scores only the shortlist
- **Domain context** — `context` and `column_context` inject column level descriptions into every prompt
- **Full join semantics** — `inner`, `left`, `right`, `full`  same as `pd.merge`
- **Multi-column keys** — pass a list to `left_on` / `right_on`
- **No duplicate output rows** — one best match per left row; ties broken by embedding rank
- **Reasoning output** — `return_reasoning=True` adds `_llm_score`, `_llm_reasoning`, `_embed_rank`, `_match_method`, `_llm_candidates`
- **Cost controls** — `top_k`, `embed_skip_threshold`, `max_llm_calls`
- **Multi-match mode** — `match_all=True` returns all candidates above threshold
- **Parallel scoring** — `llm_concurrency` controls how many calls run at once
- **Retry with backoff** — failed calls retry at 1s, 2s, 4s; falls back to top embed match on total failure
- **MIT license**

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `left_on` | required | Column name(s) in df1. Pass a list for multi-column keys  any number of columns supported. |
| `right_on` | required | Column name(s) in df2. Pass a list for multi-column keys  any number of columns supported. |
| `llm` | required | Your LLM function. Sync: `(str) -> str`. Async: `async (str) -> str`. |
| `embed_fn` | required | Your embedding function. `(list[str]) -> np.ndarray` of shape `[n, dim]`, dtype `float32`. |
| `context` | required | Describe what the columns represent and what kind of match to make. Injected into every LLM prompt. |
| `llm_concurrency` | required | How many LLM calls to run in parallel. `1` = sequential. Start with `10` and adjust based on your API rate limit. |
| `column_context` | `{}` | Per-column descriptions `{"col": "description"}` adds extra detail to the prompt beyond `context`. |
| `top_k` | `5` | How many embedding candidates to retrieve per left row before LLM scoring. |
| `llm_threshold` | `0.7` | Minimum LLM score (0–1) to accept a match. Rows where LLM scores below this are not joined. |
| `how` | `"inner"` | Join type: `inner` / `left` / `right` / `full` (full outer join). |
| `embed_skip_threshold` | `1.0` | Skip LLM when top embed similarity is at or above this value. Default `1.0` means only identical vectors (exact same text) skip LLM. Lower it (e.g. `0.92`) to skip LLM for near-identical matches and save cost. |
| `max_llm_calls` | `None` | Hard cap on total LLM calls. Emits a warning and returns a partial result if hit. |
| `max_retries` | `3` | How many times to retry a failed LLM call (exponential backoff). Set `0` to disable. Falls back to top embed candidate on total failure. |
| `batch_size` | `32` | Batch size for `embed_fn` calls. |
| `match_all` | `False` | Return all candidates above threshold, not just the best. Use when one left value maps to multiple right values. |
| `return_reasoning` | `False` | Add debug columns: `_llm_score`, `_llm_reasoning`, `_embed_rank`, `_match_method`, `_llm_candidates`. |

---

## License

MIT
