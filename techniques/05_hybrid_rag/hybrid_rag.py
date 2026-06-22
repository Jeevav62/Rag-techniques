"""
Hybrid RAG — vector search (semantic) + BM25 (keyword), fused together.
Best all-round technique. Handles both "what does X mean" and "find the word X".
"""
import time
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv(find_dotenv())

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

# ── Change these to point at your documents ──
SAMPLE_FILES = ["your_document.pdf"]
SAMPLE_QUERY = "What is the main topic of this document?"


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

    vector_index = VectorStoreIndex(nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=3)

    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
    )

    query_engine = RetrieverQueryEngine(retriever=hybrid_retriever)

    start = time.time()
    response = query_engine.query(query)
    latency = round(time.time() - start, 2)

    chunks = [node.text for node in response.source_nodes]

    print(f"\nANSWER\n{'-'*40}\n{response.response}")
    print(f"\nHYBRID RETRIEVAL\n{'-'*40}")
    print(f"Dense (Vector) : Top-3")
    print(f"Sparse (BM25)  : Top-3")
    print(f"Fusion Mode    : Reciprocal Rank")
    print(f"Final Chunks   : {len(chunks)}")
    print(f"Latency        : {latency}s")
    print(f"\nCHUNKS\n{'-'*40}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
