# llm-join Design Spec
Date: 2026-05-23

## Overview

`llm-join` is a PyPI Python library for semantic fuzzy joining of DataFrames using LLMs. Users bring their own LLM callable. The module handles embedding-based candidate retrieval, prompt construction, batching, scoring, and merging.

**Core use case:** Join DataFrames where values are semantically equivalent but textually different — "Sony WH-1000XM5" = "SONY-WH1000XM5-BLK" = "Sony Headphones WH1000XM5".

## Use Cases by Domain

| Domain | Left Column | Right Column | Problem Solved |
|--------|-------------|--------------|----------------|
| **Supply Chain** | Buyer catalog SKU | Supplier SKU | Match products across 50+ vendor catalogs |
| **Procurement** | Purchase order vendor | Vendor master name | Consolidate spend by true vendor |
| **Finance** | Expense report payee | GL account name | Reconcile transactions to chart of accounts |
| **Accounting** | Bank transaction desc | Invoice vendor name | Auto-match bank feeds to invoices |
| **E-commerce** | Marketplace listing | Master product catalog | Deduplicate listings across platforms |
| **Retail** | POS product name | Inventory system SKU | Sync point-of-sale to warehouse |
| **Sales** | CRM contact name | Marketing lead name | Deduplicate leads before outreach |
| **Ad Tech** | Publisher name | DSP seat ID | Match inventory sources across exchanges |
| **Legal** | Contract party name | Corporate registry | Identify true legal entity in contracts |
| **Compliance** | Customer name | OFAC sanctions list | Sanctions screening / KYC |
| **Compliance** | Customer name | PEP database | Politically exposed persons check |
| **IP / Patents** | Patent assignee | Company master | Attribute patents to correct parent company |
| **Government** | Vendor name | Tax registry | Consolidate procurement spend |
| **Government** | Grant recipient | Voter / citizen records | Cross-agency entity resolution |
| **Pharma** | Drug generic name | Brand name | Generic ↔ brand equivalence mapping |
| **Pharma** | Clinical trial drug | Formulary name | Trial data ↔ insurance formulary join |
| **HR** | Job title (resume) | Standard job taxonomy | Normalize titles for compensation analysis |
| **HR** | Skills (resume) | Skills taxonomy | Map freetext skills to canonical ontology |
| **Media** | Content title | Streaming catalog | Match content across Netflix / Prime / Hulu |
| **Research** | Author name | Citation database | Disambiguate authors across publications |
| **News / NLP** | Named entity | Knowledge graph node | Ground extracted entities to Wikidata / DBpedia |
| **Real Estate** | Address (raw input) | Property records DB | Standardize and match addresses |

---

## Public API

```python
from llm_join import fuzzy_join

result = fuzzy_join(
    df1, df2,
    left_on="drug_name",           # str or list[str]
    right_on="brand_name",         # str or list[str]
    llm=my_fn,                     # callable: (prompt: str) -> str | Awaitable[str]
    context="pharma drug names",        # str — global domain context
    column_context={"drug_name": "..."},# dict[col, str] — per-column context (optional)
    top_k=5,                       # embedding candidates per row (default: 5)
    threshold=0.7,                 # min LLM score to accept match (default: 0.7)
    how="inner",                   # inner/left/right/outer (default: inner)
    embed_model="text-embedding-3-small",  # embedding model identifier
    batch_size=32,                 # LLM pairs per prompt (default: 32)
    embed_threshold=0.95,          # skip LLM if embed score is decisive (optional)
    max_llm_calls=None,            # hard cap on LLM calls, warns if hit (optional)
    return_reasoning=False,        # append _llm_score, _llm_reasoning, _embed_rank cols
)
```

**Output columns added when `return_reasoning=True`:**
- `_llm_score` — float 0–1, LLM confidence
- `_llm_reasoning` — string explanation from LLM
- `_embed_rank` — rank among top_k embedding candidates

**Minimal usage:**
```python
import openai
from llm_join import fuzzy_join

client = openai.OpenAI()
llm = lambda p: client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": p}]
).choices[0].message.content

result = fuzzy_join(df1, df2, left_on="drug", right_on="brand", llm=llm)
```

---

## Architecture

```
fuzzy_join()
    │
    ▼
ColumnConfig(s)
┌──────────────────────────────────┐
│ left_col, right_col              │
│ context_str (global + per-col)   │
│ top_k, threshold, embed_model    │
└──────────────────────────────────┘
    │
    ▼
EmbeddingRetriever
┌──────────────────────────────────┐
│ encode right_col → faiss index   │
│ encode left_col → ANN search     │
│ returns top_k candidates per row │
└──────────────────────────────────┘
    │  List[(left_val, [candidates])]
    ▼
LLMScorer
┌──────────────────────────────────┐
│ batch candidate pairs → prompt   │
│ inject context_str               │
│ call llm() — sync or async       │
│ parse JSON response → scores     │
│ retry once on malformed JSON     │
└──────────────────────────────────┘
    │  List[(left_val, best_match, score, reasoning)]
    ▼
Merger
┌──────────────────────────────────┐
│ filter by threshold              │
│ pd.merge on matched keys         │
│ append score/reasoning if flag   │
└──────────────────────────────────┘
    │
    ▼
DataFrame
```

**Boundaries:**
- `EmbeddingRetriever`: embeddings + faiss only, no LLM knowledge
- `LLMScorer`: prompt in, scored pairs out, no DataFrame knowledge
- `Merger`: scored pairs + DataFrames in, merged DataFrame out
- `ColumnConfig`: dataclass flowing through all stages

---

## File Structure

```
llm-join/
├── llm_join/
│   ├── __init__.py       # exports fuzzy_join
│   ├── join.py           # fuzzy_join() orchestrator
│   ├── config.py         # ColumnConfig dataclass
│   ├── retriever.py      # EmbeddingRetriever
│   ├── scorer.py         # LLMScorer
│   ├── merger.py         # Merger
│   └── prompts.py        # prompt templates
├── tests/
│   ├── test_retriever.py
│   ├── test_scorer.py
│   ├── test_merger.py
│   └── test_join.py      # integration with mock llm
├── pyproject.toml
└── README.md
```

**Dependencies:**
- `pandas`
- `numpy`
- `faiss-cpu`
- `sentence-transformers` (optional, for local embed models)
- No LLM SDK — user brings own

---

## LLM Prompt Design

Single prompt per batch — all top_k candidates for one left row sent together.

**Prompt template:**
```
You are a data matching assistant.
Context: {context_str}

For each pair below, score how likely LEFT matches RIGHT (0.0–1.0).
Respond ONLY as JSON array: [{"index": 0, "score": 0.95, "reasoning": "..."}]

Pairs:
0. LEFT: "{left_val}" | RIGHT: "{candidate_0}"
1. LEFT: "{left_val}" | RIGHT: "{candidate_1}"
...
```

**Parsing:**
- Expect JSON array, index-aligned
- Malformed JSON: retry once, then score 0.0 + emit warning
- `reasoning` stored only if `return_reasoning=True`

---

## Scale & Batching

| Scale | Rows | Strategy |
|-------|------|----------|
| Small | <10k | sync, batch_size=32 |
| Medium | 10k–1M | async batching, thread pool for sync llm |
| Large | 1M+ | embed_threshold pre-filter skips LLM for obvious matches/misses |

**Auto-detection:**
- `asyncio.iscoroutinefunction(llm)` → async path
- sync `llm` → `ThreadPoolExecutor` with `max_workers`

**Cost controls:**
- `embed_threshold`: skip LLM if top embed score > threshold (obvious match) or all candidates < (1 - threshold) (obvious non-match)
- `max_llm_calls`: hard cap, raises `UserWarning` when hit, returns partial result

---

## Context Parameter

Accepts string, dict, or both:

```python
# String only — global context
context="pharmaceutical drug names"

# Dict only — per-column context
context={"drug_name": "generic drug name", "brand_name": "US brand name"}

# Both — merged in prompt
context="pharmaceutical domain"
column_context={"drug_name": "generic", "brand_name": "brand"}
```

Final context string injected into prompt = global + per-column, concatenated.
