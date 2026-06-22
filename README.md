<div align="center">

# 🧠 Advanced RAG Cookbook

### A practical collection of 10 Retrieval-Augmented Generation techniques — with working code, real benchmarks, and plain-English explanations.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10%2B-purple?style=for-the-badge)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green?style=for-the-badge&logo=openai)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge)](https://github.com/Jeevav62/Rag-techniques/pulls)

<br/>

**RAG** · **Advanced RAG** · **RAG Techniques** · **LlamaIndex** · **Hybrid RAG** · **HyDE** · **Fusion RAG** · **Parent-Child RAG** · **Retrieval Augmented Generation**

</div>

---

## What is this?

This cookbook covers **10 RAG techniques** — from the simplest naive baseline to multi-stage reranking pipelines. Each technique has:

- A README that explains **what it does, why it works, and when to use it** (no fluff)
- A **standalone Python script** you can run directly
- **Real benchmark numbers** from actual tests (not estimates)

Built with [LlamaIndex](https://www.llamaindex.ai/) + OpenAI. Tested against the same document set across all techniques so the comparisons are fair.

> If you've ever Googled "which RAG technique should I use" and got a vague answer — this repo is the practical answer to that question.

---

## Techniques

| # | Technique | Accuracy | Avg Latency | Best For |
|---|-----------|:--------:|:-----------:|----------|
| [01](techniques/01_naive_rag/) | **Naive RAG** | Medium | ~4.7s | Baseline, simple docs, getting started |
| [02](techniques/02_unstructured_rag/) | **Unstructured RAG** | Medium | ~4.9s | Scanned PDFs, images, OCR-heavy files |
| [03](techniques/03_contextual_compression/) | **Contextual Compression** | High | ~10s | Long docs, noisy datasets, high precision |
| [04](techniques/04_fusion_rag/) | **Fusion RAG** | High | ~7.5s | Ambiguous queries, better recall |
| [05](techniques/05_hybrid_rag/) | **Hybrid RAG** ⭐ | High | ~5.6s | Best all-around — keyword + semantic |
| [06](techniques/06_hyde_rag/) | **HyDE RAG** | High | ~12.5s | Short/vague queries, Q&A over profiles |
| [07](techniques/07_parent_child_rag/) | **Parent-Child RAG** | Very High | ~8.3s | Long documents, context-heavy answers |
| [08](techniques/08_rrr_rag/) | **RRR RAG** | High | ~5.3s | Conversational queries, chatbot interfaces |
| [09](techniques/09_sentence_compression/) | **Sentence Compression** | High | ~6-7s | Noisy chunks, reducing context tokens |
| [10](techniques/10_rerank_compress/) | **Rerank + Compress** | Very High | ~10-11s | Max precision, high-stakes QA |

---

## Quick Decision Guide

**Default choice with no constraints** → [Hybrid RAG](techniques/05_hybrid_rag/) — handles keyword and semantic queries, reliable across document types.

**Users type conversationally** → [RRR RAG](techniques/08_rrr_rag/) — rewrites vague queries before retrieval.

**Scanned PDFs or images** → [Unstructured RAG](techniques/02_unstructured_rag/) — OCR layer handles what text readers can't.

**Short, vague questions** → [HyDE RAG](techniques/06_hyde_rag/) — generates a hypothetical answer and retrieves against that.

**Long documents, need context in answers** → [Parent-Child RAG](techniques/07_parent_child_rag/) — small chunks for precision, large chunks for context.

**Accuracy is the only priority** → [Rerank + Compress](techniques/10_rerank_compress/) — two-stage post-processing, cleanest context.

**Just learning RAG** → [Naive RAG](techniques/01_naive_rag/) — understand the baseline before adding complexity.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Jeevav62/Rag-techniques.git
cd Rag-techniques
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env → add your OPENAI_API_KEY
```

### 4. (For OCR) Install Tesseract

Only needed for Unstructured RAG when working with scanned PDFs/images.

| OS | Command |
|---|---|
| macOS | `brew install tesseract` |
| Ubuntu | `sudo apt-get install tesseract-ocr` |
| Windows | [Download installer](https://github.com/UB-Mannheim/tesseract/wiki) |

---

## Running a Technique

Each `techniques/XX/` folder has a standalone script. Update `SAMPLE_FILES` and `SAMPLE_QUERY` at the top, then:

```bash
cd techniques/05_hybrid_rag
python hybrid_rag.py
```

Output includes: answer, latency, token count, and the exact chunks retrieved.

---

## Interactive Demo (Gradio)

Try all 10 techniques side by side with your own documents:

```bash
python app.py
```

Opens at `http://localhost:7860` — upload any PDF/doc/image, pick a technique, compare outputs.

---

## Benchmark Results

All techniques tested on the same personal document QA dataset (bio/resume-style questions — "where does this person work?", "what city are they in?", etc.)

| Technique | Accuracy | Avg Latency | Extra LLM Calls |
|-----------|----------|-------------|-----------------|
| Naive RAG | Medium | 4.7s | 0 |
| Unstructured RAG | Medium | 4.9s | 0 |
| Contextual Compression | High | 10.0s | 1 (reranker) |
| Fusion RAG | High | 7.5s | 1 (query expansion) |
| Hybrid RAG | High | 5.6s | 0 |
| HyDE RAG | High | 12.5s | 1 (hypothesis gen) |
| Parent-Child RAG | Very High | 8.3s | 0 |
| RRR RAG | High | 5.3s | 1 (query rewrite) |
| Sentence Compression | High | 6-7s | 0 |
| Rerank + Compress | Very High | 10-11s | 2 |

---

## Stack

| Component | Library |
|-----------|---------|
| RAG Framework | [LlamaIndex](https://www.llamaindex.ai/) 0.10+ |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-large |
| Document Parsing | [Unstructured](https://unstructured.io/) |
| Sparse Retrieval | BM25 (rank-bm25) |
| UI | [Gradio](https://www.gradio.app/) |

---

## Repo Structure

```
Rag-techniques/
├── app.py                          # Gradio demo — all 10 techniques
├── requirements.txt
├── .env.example
├── rag/                            # Package imported by app.py
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
└── techniques/                     # Standalone scripts + docs per technique
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

---

## Contributing

Found a bug? Want to add a technique? PRs are welcome.

1. Fork the repo
2. Add your technique in `techniques/XX_<name>/` with a README + script
3. Add the `run_rag()` function to `rag/<name>.py` and register it in `rag/__init__.py` and `app.py`
4. Open a PR

---

## License

MIT — use it, modify it, share it.

---

<div align="center">

If this helped you understand RAG better, consider giving it a ⭐

</div>
