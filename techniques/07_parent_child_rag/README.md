# Parent-Child RAG (Hierarchical Chunking)

Every chunk has a tension: small chunks are precise but lose context, large chunks have context but introduce noise. Parent-Child RAG solves this by keeping both.

Small child chunks (400 tokens) are used for retrieval — they're precise enough to match the query well. But the answer is generated from their parent chunks (2000 tokens), which have the broader context the LLM needs to answer fully.

## How it works

1. Documents parsed into two levels using `HierarchicalNodeParser`:
   - **Parent chunks**: 2000 tokens — broad context
   - **Child chunks**: 400 tokens — precise retrieval units
2. Index built from all nodes (both levels)
3. Retriever fetches top-10 chunks
4. Context built from retrieved nodes
5. LLM answers from that hierarchical context

## When to use

- Very long documents where single-level chunking loses context
- Questions that need surrounding context to answer ("what happened before X?")
- Legal, medical, or research documents where paragraph-level context matters
- When you're seeing correct retrieval but incomplete answers

## When NOT to use

- Short documents — the two-level hierarchy doesn't help when there's not much text
- When you need fast retrieval — maintaining larger chunks increases index size

## Trade-offs

| | |
|---|---|
| Speed | Slow — avg ~8.3s |
| Accuracy | Very High — best accuracy in testing |
| Cost | Moderate — more tokens in context due to parent chunk size |
| Strength | Answers are contextually grounded, not just matched |

## Benchmark

```
Latency          : ~8.3s average
Accuracy         : Very High (highest in testing)
Parent chunk size: 2000 tokens
Child chunk size : 400 tokens
Retrieved chunks : 10
```

## Run it

```bash
cp ../../.env.example ../../.env
python parent_child_rag.py
```
