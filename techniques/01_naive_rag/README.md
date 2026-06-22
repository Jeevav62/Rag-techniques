# Naive RAG

The simplest form of RAG. No tricks, no extra steps — just chunk, embed, retrieve, answer. It's the baseline every other technique improves on.

## How it works

1. Documents split into fixed-size chunks (512 tokens, 50 overlap)
2. Each chunk converted to a vector embedding
3. Query embedded using the same model
4. Top-5 most similar chunks retrieved via cosine similarity
5. LLM answers using those chunks as context

## When to use

- You're just getting started and want to understand RAG
- Your documents are clean, well-structured text
- Latency matters and accuracy requirements are moderate
- You want a fast baseline to compare other techniques against

## When NOT to use

- Queries are ambiguous or phrased in unexpected ways (try Fusion RAG)
- Documents are scanned PDFs or images (try Unstructured RAG)
- You need high precision on complex questions (try Contextual Compression)

## Trade-offs

| | |
|---|---|
| Speed | Fast — avg ~4.7s |
| Accuracy | Medium — works well on direct questions |
| Cost | Low — one embedding pass, one LLM call |
| Weakness | Retrieves chunks by similarity, not relevance — can pull noisy results |

## Benchmark

Tested on a personal document QA task (resume/bio-style questions):

```
Latency  : ~4.7s average
Accuracy : Medium
```

## Run it

```bash
# Set up env
cp ../../.env.example ../../.env
# Edit .env with your OPENAI_API_KEY

# Update SAMPLE_FILES and SAMPLE_QUERY at top of script
python naive_rag.py
```
