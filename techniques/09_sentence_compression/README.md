# Sentence Compression RAG

Standard RAG retrieves whole chunks. But a chunk of 512 tokens might have the answer in 2 sentences — the other 20 sentences are just noise in the context window. Sentence Compression RAG fixes this by filtering at the sentence level after retrieval.

`SentenceEmbeddingOptimizer` scores every sentence in each retrieved chunk by its similarity to the query, then keeps only the top 50%. What the LLM sees is a smaller, cleaner version of the retrieved chunks.

## How it works

1. Documents chunked (512 tokens, 50 overlap)
2. Vector index built, top-5 chunks retrieved
3. `SentenceEmbeddingOptimizer` scores each sentence in each chunk against the query
4. Bottom 50% of sentences dropped (by embedding similarity)
5. LLM answers from compressed chunks

## When to use

- Long chunks with mixed relevance (some sentences on-topic, some off)
- When you want to reduce context window usage without losing accuracy
- API cost optimization — fewer tokens sent to the LLM
- Documents with dense paragraphs that cover multiple topics

## When NOT to use

- Short chunks where compression removes critical context
- Documents where sentence order matters for meaning (code, sequential instructions)

## Trade-offs

| | |
|---|---|
| Speed | Medium — avg ~6-7s |
| Accuracy | High |
| Cost | Savings — compression reduces tokens sent to final LLM |
| Strength | Reduces noise at sentence level, not just chunk level |

## Benchmark

```
Latency            : ~6-7s average
Accuracy           : High
Compression        : Keep top 50% sentences per chunk
Context tokens sent: Significantly less than naive RAG
```

## Run it

```bash
cp ../../.env.example ../../.env
python sentence_compression_rag.py
```
