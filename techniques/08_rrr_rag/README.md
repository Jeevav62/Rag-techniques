# RRR RAG (Rewrite → Retrieve → Read)

Users don't always ask questions the way documents are written. "Tell me about his job" is a terrible retrieval query. RRR fixes this at the source: before retrieving anything, an LLM rewrites the query into something semantically precise and retrieval-optimized.

It's a simple idea but it works surprisingly well, especially for conversational or vague inputs.

## How it works

1. User's original query sent to LLM with prompt: "rewrite this for semantic retrieval, do NOT answer"
2. Rewritten query used for retrieval (not the original)
3. Top-5 chunks retrieved using the improved query
4. LLM answers from retrieved chunks

You can see the rewritten query in the output — it's useful for debugging why retrieval succeeded or failed.

## When to use

- Chatbot interfaces where users type conversationally
- Vague or short queries that don't match document language
- When you're seeing retrieval miss despite the answer clearly being in the document
- Low-cost upgrade over Naive RAG — just one extra LLM call

## When NOT to use

- Queries are already well-formed and precise
- Latency is critical (extra LLM call = ~1-2s added)

## Trade-offs

| | |
|---|---|
| Speed | Medium — avg ~5.3s |
| Accuracy | High |
| Cost | Low-moderate — one extra small LLM call for rewriting |
| Strength | Simple to add, meaningful accuracy gain for bad queries |

## Benchmark

```
Latency        : ~5.3s average
Accuracy       : High
Query Rewriting: Enabled (visible in output)
Retrieved Chunks: 5
```

## Run it

```bash
cp ../../.env.example ../../.env
python rrr_rag.py
```
