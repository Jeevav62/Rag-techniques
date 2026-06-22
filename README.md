# Advanced RAG Cookbook

A hands-on collection of 10 RAG techniques — each with a plain-English explanation, working code, and real benchmark numbers. Built to help you understand the trade-offs so you can pick the right technique for your use case, not just copy-paste code.

Every technique in this repo was tested against the same document set. The latency numbers are from actual runs, not estimates.

---

## Techniques

| # | Technique | Accuracy | Speed (avg) | Best For |
|---|-----------|----------|-------------|----------|
| [01](techniques/01_naive_rag/) | Naive RAG | Medium | ~4.7s | Baseline, simple docs |
| [02](techniques/02_unstructured_rag/) | Unstructured RAG | Medium | ~4.9s | Scanned PDFs, images |
| [03](techniques/03_contextual_compression/) | Contextual Compression | High | ~10s | Long docs, high noise |
| [04](techniques/04_fusion_rag/) | Fusion RAG | High | ~7.5s | Ambiguous queries |
| [05](techniques/05_hybrid_rag/) | Hybrid RAG | High | ~5.6s | Best all-around |
| [06](techniques/06_hyde_rag/) | HyDE RAG | High | ~12.5s | Short/vague queries |
| [07](techniques/07_parent_child_rag/) | Parent-Child RAG | Very High | ~8.3s | Long docs, context-heavy |
| [08](techniques/08_rrr_rag/) | RRR RAG | High | ~5.3s | Conversational queries |
| [09](techniques/09_sentence_compression/) | Sentence Compression | High | ~6-7s | Noisy chunks, token savings |
| [10](techniques/10_rerank_compress/) | Rerank + Compress | Very High | ~10-11s | Max precision |

---

## Quick Decision Guide

**If latency matters and you need a solid default:** → [Hybrid RAG](techniques/05_hybrid_rag/)

**If your users type vague/conversational questions:** → [RRR RAG](techniques/08_rrr_rag/)

**If you have scanned PDFs or images:** → [Unstructured RAG](techniques/02_unstructured_rag/)

**If accuracy is top priority and latency is okay:** → [Parent-Child RAG](techniques/07_parent_child_rag/) or [Rerank + Compress](techniques/10_rerank_compress/)

**If you just want to understand RAG before trying anything fancy:** → [Naive RAG](techniques/01_naive_rag/)

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/advanced-rag-cookbook
cd advanced-rag-cookbook
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

`.env` file:
```
OPENAI_API_KEY=sk-...
```

### 3. (Optional) Install Tesseract for OCR

Only needed if you're using Unstructured RAG with scanned PDFs or images.

- **Mac**: `brew install tesseract`
- **Ubuntu**: `apt-get install tesseract-ocr`
- **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Running Individual Techniques

Each technique folder has a standalone script. Open the script, update the `SAMPLE_FILES` and `SAMPLE_QUERY` variables at the top, then run:

```bash
cd techniques/01_naive_rag
python naive_rag.py
```

Every script prints: answer, metrics (latency, token count), and the retrieved chunks.

---

## Running the Gradio Demo

The interactive demo lets you switch between all 10 techniques, upload your own documents, and compare outputs side by side.

```bash
python app.py
```

Opens at `http://localhost:7860` — upload any PDF/doc/image, pick a technique, ask a question.

---

## Stack

- **LLM**: GPT-4o-mini (OpenAI)
- **Embeddings**: text-embedding-3-large (OpenAI)
- **RAG Framework**: LlamaIndex 0.10+
- **Document Parsing**: Unstructured
- **Sparse Retrieval**: BM25 (rank-bm25)
- **UI**: Gradio

---

## Repo Structure

```
advanced-rag-cookbook/
├── app.py                    # Gradio demo — all 10 techniques
├── requirements.txt
├── .env.example
├── rag/                      # Package used by app.py
│   ├── naive_rag.py
│   ├── unstructured_rag.py
│   ├── contextual_rag.py
│   ├── fusion_rag.py
│   ├── hybrid_rag.py
│   ├── hyde_rag.py
│   ├── parent_child_rag.py
│   ├── rrr_rag.py
│   ├── sentence_compression_rag.py
│   └── rerank_compress_rag.py
└── techniques/               # Standalone scripts + READMEs per technique
    ├── 01_naive_rag/
    ├── 02_unstructured_rag/
    ├── 03_contextual_compression/
    ├── 04_fusion_rag/
    ├── 05_hybrid_rag/
    ├── 06_hyde_rag/
    ├── 07_parent_child_rag/
    ├── 08_rrr_rag/
    ├── 09_sentence_compression/
    └── 10_rerank_compress/
```
