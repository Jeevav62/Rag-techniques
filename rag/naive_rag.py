import time
import csv
import tiktoken
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import UnstructuredReader

load_dotenv()

# -------------------------------------------------
# Global settings
# -------------------------------------------------
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")


def count_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# -------------------------------------------------
# SAFE CSV LOADER (WINDOWS PROOF)
# -------------------------------------------------
def load_csv_safely(file_path: str):
    """
    Reads CSV with encoding fallback and converts rows to text
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    for enc in encodings:
        try:
            with open(file_path, encoding=enc, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            docs = []
            headers = rows[0]

            for row in rows[1:]:
                text = " | ".join(
                    f"{h}: {v}" for h, v in zip(headers, row)
                )
                docs.append(Document(text=text))

            return docs

        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "Unable to decode CSV using utf-8, utf-8-sig, or latin-1"
    )


# -------------------------------------------------
# NAIVE RAG (FINAL)
# -------------------------------------------------
def run_rag(query: str, files: list[str]):
    documents = []

    for file in files:
        if file.lower().endswith(".csv"):
            docs = load_csv_safely(file)   # 🔥 FINAL FIX
        else:
            reader = UnstructuredReader()
            docs = reader.load_data(
                unstructured_kwargs={"filename": file}
            )

        documents.extend(docs)

    if not documents:
        return "❌ No documents loaded.", []

    # -------------------------------------------------
    # Chunking
    # -------------------------------------------------
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # -------------------------------------------------
    # Index + Query
    # -------------------------------------------------
    index = VectorStoreIndex(nodes)

    start = time.time()
    response = index.as_query_engine(
        similarity_top_k=5
    ).query(query)
    end = time.time()

    chunks = [node.text for node in response.source_nodes]

    answer = f"""
🧠 ANSWER
{response.response}

-------------------------
📊 METRICS
Retrieved Chunks : {len(chunks)}
Latency          : {round(end - start, 2)} seconds
"""

    return answer.strip(), chunks
