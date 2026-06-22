"""
RRR RAG — Rewrite → Retrieve → Read.
LLM rewrites vague queries into precise retrieval-optimized ones before searching.
"""
import time
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv(find_dotenv())

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

# ── Change these to point at your documents ──
SAMPLE_FILES = ["your_document.pdf"]
SAMPLE_QUERY = "What is the main topic of this document?"


class RewriteQueryTransform:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query_bundle: QueryBundle, metadata=None):
        prompt = f"""Rewrite the following user query to be clear, explicit,
and optimized for semantic document retrieval.
Do NOT answer the question.

User Query:
{query_bundle.query_str}

Rewritten Query:"""
        rewritten = self.llm.complete(prompt).text.strip()
        print(f"\n[Rewritten Query] {rewritten}\n")
        return QueryBundle(
            query_str=rewritten,
            custom_embedding_strs=[rewritten],
        )


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

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.llm = llm

    index = VectorStoreIndex(nodes)
    base_query_engine = index.as_query_engine(similarity_top_k=5)

    rrr_query_engine = TransformQueryEngine(
        query_engine=base_query_engine,
        query_transform=RewriteQueryTransform(llm),
    )

    start = time.time()
    response = rrr_query_engine.query(query)
    latency = round(time.time() - start, 2)

    chunks = [node.text for node in response.source_nodes]

    print(f"\nANSWER\n{'-'*40}\n{response.response}")
    print(f"\nRRR DETAILS\n{'-'*40}")
    print(f"Query Rewriting  : Enabled")
    print(f"Retrieved Chunks : {len(chunks)}")
    print(f"Latency          : {latency}s")
    print(f"\nCHUNKS\n{'-'*40}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
