# Rerank + Compress RAG

This combines two post-processing steps back to back: first LLM reranking (removes bad chunks), then sentence compression (removes bad sentences from the remaining chunks).

It's the most thorough context-cleaning pipeline in this cookbook. You start with 8 retrieved chunks, rerank to the best 2, then compress those 2 down to their most relevant sentences. What reaches the LLM is extremely clean.

The trade-off is two extra LLM calls — one for reranking, one embedded in the compression pass. But if your use case demands high precision and you have some latency budget, this is worth it.

## How it works

1. Documents chunked (512 tokens, 50 overlap)
2. Vector retrieval — top-8 chunks fetched (casting a wide net)
3. **Stage 1**: `LLMRerank` evaluates all 8 and keeps top-2
4. **Stage 2**: `SentenceEmbeddingOptimizer` scores sentences in those 2 chunks, keeps top 40%
5. LLM answers from the final compressed, reranked context

## When to use

- Accuracy is top priority and latency is acceptable
- Long documents with noisy, mixed-relevance chunks
- High-stakes QA where hallucinations from irrelevant context are costly
- Want maximum precision with minimum context tokens

## When NOT to use

- Latency-sensitive applications
- Simple documents — two-stage pipeline is overkill for clean text
- Budget-constrained — extra LLM calls add up at scale

## Trade-offs

| | |
|---|---|
| Speed | Slow — avg ~10-11s |
| Accuracy | Very High |
| Cost | Higher — 1 extra LLM call (reranker) + extra embedding pass (compression) |
| Strength | Highest-precision context of all techniques |

## Benchmark

```
Latency             : ~10-11s average
Accuracy            : Very High
Step 1 (Retrieve)   : 8 chunks
Step 2 (Rerank)     : Top 2 kept
Step 3 (Compress)   : Top 40% sentences kept
Final context tokens: Much lower than naive 8-chunk context
```

## Run it

```bash
cp ../../.env.example ../../.env
python rerank_compress_rag.py
```
