# Contextual Compression RAG

Retrieve more than you need, then use an LLM to keep only the best chunks. The idea is that vector similarity isn't perfect — it pulls in related chunks that might be off-topic. Adding a reranking step removes the noise before sending context to the final LLM.

It also uses sentence-window chunking, which means each chunk carries its surrounding sentences as metadata. This gives the LLM more context when answering.

## How it works

1. Documents parsed with `SentenceWindowNodeParser` (window of 3 sentences around each chunk)
2. Retriever pulls top-10 chunks (broad net intentionally)
3. `LLMRerank` evaluates all 10 and keeps the best 3
4. Final answer generated from those 3 compressed chunks

## When to use

- Long documents with lots of noise (annual reports, research papers, legal docs)
- You care more about accuracy than speed
- Queries are specific and need precise answers
- Dataset has high variance in chunk quality

## When NOT to use

- Latency is a hard constraint — this makes 2 LLM calls
- Simple documents where basic retrieval already works well

## Trade-offs

| | |
|---|---|
| Speed | Slow — avg ~10s (extra LLM rerank call) |
| Accuracy | High — LLM actively removes bad chunks |
| Cost | Higher — reranker = extra tokens |
| Strength | Precision. You send less context but better context |

## Benchmark

```
Latency  : ~10s average
Accuracy : High
Chunks before rerank : 10
Chunks after rerank  : 3
```

## Run it

```bash
cp ../../.env.example ../../.env
python contextual_rag.py
```
