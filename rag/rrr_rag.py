from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.readers.file import UnstructuredReader
from llama_index.core.node_parser import SentenceSplitter


def run_rag(query: str, files: list[str]):
    """
    RRR RAG (CSV-safe)
    - Rewrite → Retrieve → Read
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

    # 3️⃣ Configure LLM
    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    Settings.llm = llm

    # 4️⃣ Build vector index
    index = VectorStoreIndex(nodes)

    # 5️⃣ Base query engine (Retrieve + Read)
    base_query_engine = index.as_query_engine(
        similarity_top_k=5
    )

    # 6️⃣ Rewrite Transform (UNCHANGED LOGIC)
    class RewriteQueryTransform:
        def __init__(self, llm):
            self.llm = llm

        def run(self, query_bundle: QueryBundle, metadata=None):
            prompt = f"""
Rewrite the following user query to be clear, explicit,
and optimized for semantic document retrieval.
Do NOT answer the question.

User Query:
{query_bundle.query_str}

Rewritten Query:
"""
            rewritten_query = self.llm.complete(prompt).text.strip()

            return QueryBundle(
                query_str=rewritten_query,
                custom_embedding_strs=[rewritten_query],
            )

    rewrite_transform = RewriteQueryTransform(llm)

    # 7️⃣ RRR query engine
    rrr_query_engine = TransformQueryEngine(
        query_engine=base_query_engine,
        query_transform=rewrite_transform
    )

    # 8️⃣ Query
    response = rrr_query_engine.query(query)

    # 9️⃣ Extract chunks
    chunks = [node.text for node in response.source_nodes]

    answer = f"""
🧠 ANSWER
{response.response}

-------------------------
📄 RRR DETAILS

Query Rewriting : Enabled
Retrieved Chunks: {len(chunks)}
"""

    return answer.strip(), chunks
