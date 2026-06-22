"""
Parent-Child RAG — two-tier chunking. Small chunks retrieved, large chunks used for context.
Precision of small chunks + context richness of large chunks.
"""
import time
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv(find_dotenv())

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.7)
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

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2000, 400])
    nodes = node_parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=10)

    start = time.time()
    retrieved_nodes = retriever.retrieve(query)
    latency_retrieve = round(time.time() - start, 2)

    if not retrieved_nodes:
        print("Not found in documents.")
        return

    context_chunks = [n.node.text for n in retrieved_nodes]
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant.

Using the context below, answer the user's question.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:"""

    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

    start = time.time()
    response = llm.complete(prompt)
    latency_llm = round(time.time() - start, 2)

    print(f"\nANSWER\n{'-'*40}\n{response.text.strip()}")
    print(f"\nPARENT-CHILD DETAILS\n{'-'*40}")
    print(f"Parent Chunk Size : 2000 tokens")
    print(f"Child Chunk Size  : 400 tokens")
    print(f"Retrieved Chunks  : {len(context_chunks)}")
    print(f"Retrieve Latency  : {latency_retrieve}s")
    print(f"LLM Latency       : {latency_llm}s")
    print(f"\nCHUNKS\n{'-'*40}")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
