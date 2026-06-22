from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import UnstructuredReader

load_dotenv()


def run_rag(query: str, files: list[str]):
    """
    Contextual Compression RAG (CSV-safe)
    - Sentence window chunking
    - Broad retrieval
    - LLM-based reranking
    """

    # 1️⃣ Load documents (🔥 metadata disabled)
    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(
            unstructured_kwargs={
                "filename": file,
                "metadata": True,   # 🔥 CRITICAL FIX
            }
        )
        documents.extend(docs)

    if not documents:
        return "❌ No documents loaded.", []

    # 2️⃣ Sentence-window chunking (SAFE)
    node_parser = SentenceWindowNodeParser(
        window_size=3,
        window_metadata_key="context_window",
        original_text_metadata_key="original_text",
    )

    nodes = node_parser.get_nodes_from_documents(documents)

    # 3️⃣ Build index
    index = VectorStoreIndex(nodes)

    # 4️⃣ Broad retriever (intentional)
    retriever = index.as_retriever(similarity_top_k=10)
    retrieved_nodes = retriever.retrieve(query)

    # 5️⃣ LLM-based contextual compression
    reranker = LLMRerank(
        top_n=3,
        llm=OpenAI(model="gpt-4o-mini")
    )

    compressed_nodes = reranker.postprocess_nodes(
        retrieved_nodes,
        query_str=query
    )

    # 6️⃣ Extract compressed context chunks
    context_chunks = [
        n.node.metadata.get("context_window", n.node.text)
        for n in compressed_nodes
    ]

    # 7️⃣ Final grounded answer
    context = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.
Do not guess. Do not add extra information.

Context:
{context}

Question:
{query}
"""

    llm = OpenAI(model="gpt-4o-mini")
    response = llm.complete(prompt)

    answer = f"""
🧠 ANSWER
{response.text.strip()}

-------------------------
📄 CONTEXT COMPRESSION

Retrieved Chunks (before) : {len(retrieved_nodes)}
Chunks After Compression  : {len(context_chunks)}
"""

    return answer.strip(), context_chunks
