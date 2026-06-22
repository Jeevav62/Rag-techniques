"""
Unstructured RAG — OCR + document parsing before retrieval.
Works on PDFs, scanned docs, images, mixed file types.
"""
import time
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv(find_dotenv())

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

# ── Change these to point at your documents ──
SAMPLE_FILES = ["your_scanned_document.pdf"]
SAMPLE_QUERY = "What is discussed in this document?"


def run(query: str, files: list[str]):
    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(unstructured_kwargs={"filename": file})
        documents.extend(docs)

    if not documents:
        print("No content extracted from files.")
        return

    start = time.time()
    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine().query(query)
    latency = round(time.time() - start, 2)

    chunks = [node.text for node in response.source_nodes]

    print(f"\nANSWER\n{'-'*40}\n{response}")
    print(f"\nMETRICS\n{'-'*40}")
    print(f"Retrieved Chunks : {len(chunks)}")
    print(f"Latency          : {latency}s")
    print(f"\nCHUNKS\n{'-'*40}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
