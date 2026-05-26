"""
Fast async fuzzy_join — reference pattern.

This example uses OpenAI's API directly via httpx with:
  * One AsyncClient reused across all calls (HTTP/2, connection pooling)
  * Async embed_fn (auto-detected by llm-join)
  * Async llm function (auto-detected by llm-join)
  * Large batch_size + high embed_concurrency
  * embed_skip_threshold to skip LLM for near-identical pairs
  * verbose=1 for progress bars + summary

Adapt the URL, headers, and model names to any provider:
  - Azure OpenAI: change base_url + add api-version param
  - Anthropic: change endpoint + payload schema
  - Bedrock: use boto3 instead of httpx
  - Ollama (local): point base_url to http://localhost:11434

For a 10k x 100k join with this pattern: ~5 min, ~$0.75 on gpt-4o-mini.
"""
import os
import numpy as np
import pandas as pd
import httpx
from llm_join import fuzzy_join


# ---------------------------------------------------------------------------
# Reusable HTTP client (single TCP/TLS handshake, kept warm across all calls)
# ---------------------------------------------------------------------------

API_KEY = os.environ["OPENAI_API_KEY"]

client = httpx.AsyncClient(
    base_url="https://api.openai.com",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    http2=True,
    timeout=httpx.Timeout(60.0, connect=10.0),
    limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
)


# ---------------------------------------------------------------------------
# Async embed function — receives ONE batch, returns np.float32 array.
# llm-join handles batching and parallel calls.
# ---------------------------------------------------------------------------

async def my_embed(texts: list[str]) -> np.ndarray:
    r = await client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": texts},
    )
    r.raise_for_status()
    data = r.json()["data"]
    return np.array([d["embedding"] for d in data], dtype="float32")


# ---------------------------------------------------------------------------
# Async LLM function — receives a prompt string, returns response string.
# ---------------------------------------------------------------------------

async def my_llm(prompt: str) -> str:
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 800,
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Run the join
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df1 = pd.DataFrame({
        "vendor": [
            "Goldman Sachs & Co.",
            "Amazon Web Services",
            "Microsoft Corp",
        ],
        "spend": [1_200_000, 890_000, 340_000],
    })

    df2 = pd.DataFrame({
        "supplier_name": [
            "The Goldman Sachs Group Inc",
            "Amazon.com Inc.",
            "Microsoft Corporation",
            "JPMorgan Chase",
        ],
        "category": ["Finance", "Cloud", "Software", "Finance"],
    })

    result = fuzzy_join(
        df1, df2,
        left_on="vendor",
        right_on="supplier_name",
        llm_fn=my_llm,
        embed_fn=my_embed,
        context="company names — match legal entity variants and abbreviations",
        # Speed knobs
        batch_size=512,             # texts per embed_fn call
        embed_concurrency=50,       # parallel embed batches
        llm_concurrency=40,         # parallel LLM calls
        # Cost / quality knobs
        embed_skip_threshold=0.95,  # skip LLM for high-confidence pairs
        llm_threshold=0.7,          # min LLM score to accept match
        top_k=5,                    # candidates sent per LLM call
        # Observability
        verbose=1,                  # progress bars + summary
        return_reasoning=True,
    )

    print(result)
