# HyDE RAG (Hypothetical Document Embeddings)

The problem with short queries: they don't look like documents. When you type "where does she work?", that query embedding is very different from the document chunk that says "Currently working at Acme Corp as a Senior Engineer." HyDE fixes this mismatch.

Instead of embedding the query and searching, HyDE asks the LLM to write a hypothetical answer first, then embeds that and searches. The hypothetical document looks like a real document, so it matches better.

## How it works

1. User query sent to LLM → generates a hypothetical document (a made-up answer)
2. Hypothetical document embedded (not the original query)
3. Retrieval using that embedding — finds chunks that look like the hypothetical answer
4. LLM answers again using the **actual retrieved chunks** (not the hypothetical)

The hypothetical answer is just used for retrieval. The final answer still comes from real document content.

## When to use

- Short, vague, or conversational queries ("what does he do?")
- Q&A over personal documents (resumes, bios, profiles)
- Cases where query-document vocabulary mismatch hurts retrieval
- When you're seeing good documents exist but wrong chunks being retrieved

## When NOT to use

- Queries are already specific and well-formed
- Documents are factual/technical where hypothetical generation could mislead retrieval
- Latency is a concern — the extra LLM call is slower than most

## Trade-offs

| | |
|---|---|
| Speed | Slow — avg ~12.5s (slowest in testing) |
| Accuracy | High — especially for short/vague queries |
| Cost | Higher — extra LLM call for hypothesis generation |
| Strength | Solves vocabulary mismatch between query and document style |

## Benchmark

```
Latency               : ~12.5s average (extra LLM call for hypothesis)
Accuracy              : High
Hypothetical Doc Used : Yes
Original Query Kept   : Yes (searched alongside hypothesis)
```

## Run it

```bash
cp ../../.env.example ../../.env
python hyde_rag.py
```
