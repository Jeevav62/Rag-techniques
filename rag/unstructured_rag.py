from dotenv import load_dotenv
from llama_index.readers.file import UnstructuredReader
from llama_index.core import VectorStoreIndex

load_dotenv()


def run_rag(query: str, files: list):
    """
    Unstructured Naive RAG
    - Supports images, PDFs, docs, etc.
    - OCR + text extraction
    - Vector search + QA
    """

    # 1️⃣ Load documents using Unstructured
    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(
            unstructured_kwargs={"filename": file}
        )
        documents.extend(docs)

    if not documents:
        return "❌ No content extracted from files.", []

    # 2️⃣ Build vector index
    index = VectorStoreIndex.from_documents(documents)

    # 3️⃣ Query
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    # 4️⃣ Extract chunks
    chunks = [node.text for node in response.source_nodes]

    # 5️⃣ Final answer
    answer = f"""
🧠 ANSWER
{response}

-------------------------
📄 SOURCE COUNT
Retrieved Chunks: {len(chunks)}
"""

    return answer.strip(), chunks
