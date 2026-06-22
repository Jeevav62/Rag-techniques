# Hybrid RAG

Two retrieval strategies combined: vector search finds semantically similar chunks, BM25 finds keyword matches. Both run independently, then their results are fused using Reciprocal Rank Fusion.

This turned out to be the best all-around technique in testing. Vector search alone misses exact keyword matches. BM25 alone misses semantic meaning. Together they cover both cases well.

## How it works

1. Documents chunked (512 tokens, 50 overlap)
2. Two retrievers built from the same nodes:
   - **Dense (Vector)**: finds top-3 semantically similar chunks
   - **Sparse (BM25)**: finds top-3 keyword-matching chunks
3. Results merged via Reciprocal Rank Fusion
4. LLM answers from fused top chunks

## When to use

- Technical documents with specific terminology (logs, APIs, specs)
- Mixed queries — some semantic, some keyword-specific
- You want a reliable default that handles most document types
- Production systems where you want predictable retrieval behavior

## When NOT to use

- Very short documents where both retrievers pull the same chunks anyway
- Memory-constrained environments (maintains both vector index and BM25 index)

## Trade-offs

| | |
|---|---|
| Speed | Medium — avg ~5.6s |
| Accuracy | High — best balance in testing |
| Cost | Moderate — two retriever passes, one fusion |
| Strength | Handles keyword AND semantic queries reliably |

## Benchmark

```
Latency          : ~5.6s average
Accuracy         : High (best overall in our tests)
Dense retriever  : Top-3 chunks
Sparse retriever : Top-3 chunks
Fusion           : Reciprocal Rank
```

## Run it

```bash
cp ../../.env.example ../../.env
python hybrid_rag.py
```
