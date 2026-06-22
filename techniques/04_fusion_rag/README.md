# Fusion RAG

One query isn't always the best query. Fusion RAG takes your original question, generates 3 alternative phrasings of it, runs all 4 queries against the index, then merges the results using Reciprocal Rank Fusion (RRF).

The insight: the same information can be worded many different ways. If you only search with the user's exact phrasing, you might miss relevant chunks that use different terminology.

## How it works

1. User query sent to LLM → generates 3 alternative queries
2. All 4 queries (original + 3 variants) retrieve chunks independently
3. Results merged using Reciprocal Rank Fusion — chunks that rank well across multiple queries score higher
4. LLM answers from the fused, re-ranked context

## When to use

- Ambiguous or short queries ("what did they do in 2022?")
- Users who phrase things conversationally
- Domain where the same concept has multiple names
- You want better recall without adding complexity to the index

## When NOT to use

- Query is already very specific and well-formed
- Latency matters more than recall
- Simple documents where keyword matching is enough

## Trade-offs

| | |
|---|---|
| Speed | Medium — avg ~7.5s |
| Accuracy | High — better recall than naive |
| Cost | Moderate — 4 retrieval passes + query expansion LLM call |
| Strength | Robust to query phrasing variations |

## Benchmark

```
Latency         : ~7.5s average
Accuracy        : High
Expanded Queries: 4 (1 original + 3 generated)
Fusion Mode     : Reciprocal Rank Fusion
```

## Run it

```bash
cp ../../.env.example ../../.env
python fusion_rag.py
```
