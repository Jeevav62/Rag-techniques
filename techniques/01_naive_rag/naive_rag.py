"""
Naive RAG — the baseline.
Chunk → embed → retrieve top-k → answer.
"""
import csv
import time
import tiktoken
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv(find_dotenv())

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

# ── Change these to point at your documents ──
SAMPLE_FILES = ["your_document.pdf"]
SAMPLE_QUERY = "What is the main topic of this document?"


def count_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def load_csv_safely(file_path: str):
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(file_path, encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            headers = rows[0]
            return [
                Document(text=" | ".join(f"{h}: {v}" for h, v in zip(headers, row)))
                for row in rows[1:]
            ]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Could not decode CSV with utf-8, utf-8-sig, or latin-1")


def run(query: str, files: list[str]):
    documents = []
    for file in files:
        if file.lower().endswith(".csv"):
            documents.extend(load_csv_safely(file))
        else:
            reader = UnstructuredReader()
            documents.extend(reader.load_data(unstructured_kwargs={"filename": file}))

    if not documents:
        print("No documents loaded.")
        return

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    start = time.time()
    response = index.as_query_engine(similarity_top_k=5).query(query)
    latency = round(time.time() - start, 2)

    chunks = [node.text for node in response.source_nodes]
    context_tokens = sum(count_tokens(c) for c in chunks)

    print(f"\nANSWER\n{'-'*40}\n{response.response}")
    print(f"\nMETRICS\n{'-'*40}")
    print(f"Retrieved Chunks : {len(chunks)}")
    print(f"Context Tokens   : {context_tokens}")
    print(f"Latency          : {latency}s")
    print(f"\nCHUNKS\n{'-'*40}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
