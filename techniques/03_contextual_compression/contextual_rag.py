"""
Contextual Compression RAG — retrieve broad, then compress with LLM reranking.
Sentence-window chunking keeps surrounding context; LLM trims to top 3.
"""
import time
from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv(find_dotenv())

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

# ── Change these to point at your documents ──
SAMPLE_FILES = ["your_document.pdf"]
SAMPLE_QUERY = "What is the main topic of this document?"


def run(query: str, files: list[str]):
    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(unstructured_kwargs={"filename": file, "metadata": True})
        documents.extend(docs)

    if not documents:
        print("No documents loaded.")
        return

    node_parser = SentenceWindowNodeParser(
        window_size=3,
        window_metadata_key="context_window",
        original_text_metadata_key="original_text",
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    retriever = index.as_retriever(similarity_top_k=10)
    retrieved_nodes = retriever.retrieve(query)

    reranker = LLMRerank(top_n=3, llm=OpenAI(model="gpt-4o-mini"))

    start = time.time()
    compressed_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=query)
    latency = round(time.time() - start, 2)

    context_chunks = [
        n.node.metadata.get("context_window", n.node.text)
        for n in compressed_nodes
    ]
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
Do not guess. Do not add extra information.

Context:
{context}

Question:
{query}"""

    llm = OpenAI(model="gpt-4o-mini")
    response = llm.complete(prompt)

    print(f"\nANSWER\n{'-'*40}\n{response.text.strip()}")
    print(f"\nCOMPRESSION STATS\n{'-'*40}")
    print(f"Retrieved (before) : {len(retrieved_nodes)} chunks")
    print(f"After Compression  : {len(context_chunks)} chunks")
    print(f"Latency            : {latency}s")
    print(f"\nCHUNKS\n{'-'*40}")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:300]}...")


if __name__ == "__main__":
    run(SAMPLE_QUERY, SAMPLE_FILES)
