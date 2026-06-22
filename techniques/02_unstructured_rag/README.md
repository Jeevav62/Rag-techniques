# Unstructured RAG

Naive RAG with a document parsing layer in front. The `unstructured` library handles OCR, layout detection, and text extraction — so you can feed it scanned PDFs, images, or mixed file types that a normal text reader would choke on.

## How it works

1. Files passed through `UnstructuredReader` (OCR + text extraction)
2. Extracted text chunked automatically
3. Vector index built from extracted content
4. Query retrieves top chunks
5. LLM answers from retrieved context

The key difference from Naive RAG: step 1. Everything else is identical. The value is in getting usable text out of messy files.

## When to use

- Scanned PDFs where text isn't selectable
- Images with text (receipts, forms, screenshots)
- Mixed uploads where you don't know the file type upfront
- Legacy documents that aren't digitally native

## When NOT to use

- Files are already clean text/markdown — overhead isn't worth it
- You need table-level extraction — Unstructured can struggle with complex tables

## Trade-offs

| | |
|---|---|
| Speed | Fast — avg ~4.9s |
| Accuracy | Medium — depends on OCR quality |
| Cost | Low — same as Naive RAG after extraction |
| Weakness | OCR errors propagate into chunks; no structural awareness |

## Benchmark

```
Latency  : ~4.9s average
Accuracy : Medium
```

## Run it

```bash
cp ../../.env.example ../../.env
# Edit SAMPLE_FILES to point at a PDF or image
python unstructured_rag.py
```
