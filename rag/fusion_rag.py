import os
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import UnstructuredReader
from llama_index.core.node_parser import SentenceSplitter

# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")

# -------------------------------------------------
# Global LLM + Embeddings (UNCHANGED)
# -------------------------------------------------
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-large"
)


def run_rag(query: str, files: list[str]):
    """
    Fusion RAG (CSV-safe)
    - Query expansion
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
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # 3️⃣ Build vector index
    index = VectorStoreIndex(nodes)

    # 4️⃣ Base retriever
    base_retriever = index.as_retriever(
        similarity_top_k=5
    )

    # 5️⃣ Fusion retriever (REAL fusion)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[base_retriever],
        num_queries=4,
        similarity_top_k=5,
        mode="reciprocal_rerank",
        use_async=True
    )

    # 6️⃣ Query engine
    query_engine = RetrieverQueryEngine(
        retriever=fusion_retriever
    )

    # 7️⃣ Query
    response = query_engine.query(query)

    # 8️⃣ Extract fused chunks
    chunks = [node.text for node in response.source_nodes]

    answer = f"""
🧠 ANSWER
{response.response}

-------------------------
📄 FUSION DETAILS

Expanded Queries   : 4
Retrieved Chunks   : {len(chunks)}
Fusion Mode        : Reciprocal Rank Fusion
"""

    return answer.strip(), chunks
