from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.readers.file import UnstructuredReader
from llama_index.core.node_parser import SentenceSplitter


def run_rag(query: str, files: list[str]):
    """
    HyDE RAG (CSV-safe)
    - Query → hypothetical document
    - Retrieval using hypothetical embedding
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

    # 3️⃣ Initialize LLM
    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 4️⃣ Build vector index
    index = VectorStoreIndex(nodes)

    # 5️⃣ Base query engine
    base_query_engine = index.as_query_engine(
        similarity_top_k=5
    )

    # 6️⃣ HyDE query transform (UNCHANGED)
    hyde_transform = HyDEQueryTransform(
        llm=llm,
        include_original=True
    )

    # 7️⃣ HyDE query engine
    hyde_query_engine = TransformQueryEngine(
        query_engine=base_query_engine,
        query_transform=hyde_transform
    )

    # 8️⃣ Query
    response = hyde_query_engine.query(query)

    # 9️⃣ Extract chunks
    chunks = [node.node.text for node in response.source_nodes]

    answer = f"""
🧠 ANSWER
{response.response}

-------------------------
📄 HyDE DETAILS

Hypothetical Document Used : Yes
Original Query Included   : Yes
Retrieved Chunks          : {len(chunks)}
"""

    return answer.strip(), chunks
