"""
Rerank + Compress RAG — two-stage post-processing.
First LLM picks the best chunks, then sentence compression drops noisy sentences.
Highest precision, but costs two extra LLM passes.
"""
import time
import tiktoken
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank, SentenceEmbeddingOptimizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv(find_dotenv())

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

# ── Change these to point at your documents ──
SAMPLE_FILES = ["your_document.pdf"]
SAMPLE_QUERY = "What is the main topic of this document?"


def count_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def run(query: str, files: list[str]):
    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(unstructured_kwargs={"filename": file, "metadata": False})
        documents.extend(docs)

    if not documents:
        print("No documents loaded.")
        return

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    reranker = LLMRerank(top_n=2)
    compressor = SentenceEmbeddingOptimizer(percentile_cutoff=0.4)

    start = time.time()
    query_engine = index.as_query_engine(
        similarity_top_k=8,
        node_postprocessors=[reranker, compressor],
    )
    response = query_engine.query(query)
    latency = round(time.time() - start, 2)

    chunks = [node.text for node in response.source_nodes]
    context_tokens = sum(count_tokens(c) for c in chunks)

    print(f"\nANSWER\n{'-'*40}\n{response.response}")
    print(f"\nPIPELINE\n{'-'*40}")
    print(f"Step 1 — Vector Retrieve : 8 chunks")
    print(f"Step 2 — LLM Rerank      : Top 2 kept")
    print(f"Step 3 — Sentence Compress: Top 40% sentences kept")
    print(f"Final Context Tokens     : {context_tokens}")
    print(f"Latency                  : {latency}s")
    print(f"\nCHUNKS (FINAL)\n{'-'*40}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
