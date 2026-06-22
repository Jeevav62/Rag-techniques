"""
HyDE RAG — generate a hypothetical answer first, retrieve using that embedding.
Fixes the mismatch between short vague queries and long detailed documents.
"""
import time
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
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

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    index = VectorStoreIndex(nodes)

    base_query_engine = index.as_query_engine(similarity_top_k=5)
    hyde_transform = HyDEQueryTransform(llm=llm, include_original=True)
    hyde_query_engine = TransformQueryEngine(
        query_engine=base_query_engine,
        query_transform=hyde_transform,
    )

    start = time.time()
    response = hyde_query_engine.query(query)
    latency = round(time.time() - start, 2)

    chunks = [node.node.text for node in response.source_nodes]

    print(f"\nANSWER\n{'-'*40}\n{response.response}")
    print(f"\nHyDE DETAILS\n{'-'*40}")
    print(f"Hypothetical Doc Used : Yes")
    print(f"Original Query Kept   : Yes")
    print(f"Retrieved Chunks      : {len(chunks)}")
    print(f"Latency               : {latency}s")
    print(f"\nCHUNKS\n{'-'*40}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
