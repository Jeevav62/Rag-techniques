from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import UnstructuredReader

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def run_rag(query: str, files: list[str]):
    """
    Hybrid RAG (CSV-safe)
    - Dense Vector Retriever
    - Sparse BM25 Retriever
    - Reciprocal Rank Fusion
    """

    # 1️⃣ Load documents (🔥 metadata disabled)
    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(
            unstructured_kwargs={
                "filename": file,
                "metadata": False,   # 🔥 CRITICAL FIX
            }
        )
        documents.extend(docs)

    if not documents:
        return "❌ No documents loaded.", []

    # 2️⃣ Explicit chunking (SAFE)
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = node_parser.get_nodes_from_documents(documents)

    # 3️⃣ Global LLM & Embeddings (UNCHANGED)
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    Settings.embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-large"
    )

    # 4️⃣ Dense retriever (Vector)
    vector_index = VectorStoreIndex(nodes)
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=3
    )

    # 5️⃣ Sparse retriever (BM25)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=3
    )

    # 6️⃣ Hybrid Fusion retriever
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank"
    )

    # 7️⃣ Query engine
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever
    )

    # 8️⃣ Query
    response = query_engine.query(query)

    # 9️⃣ Extract hybrid-selected chunks
    chunks = [node.text for node in response.source_nodes]

    answer = f"""
🧠 ANSWER
{response.response}

-------------------------
📄 HYBRID RETRIEVAL DETAILS

Dense Retriever (Vector) : Top-3
Sparse Retriever (BM25)  : Top-3
Fusion Mode              : Reciprocal Rank
Final Chunks Used        : {len(chunks)}
"""

    return answer.strip(), chunks
