import time
import tiktoken
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank, SentenceEmbeddingOptimizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")


def count_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def run_rag(query: str, files: list[str]):
    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(
            unstructured_kwargs={"filename": file, "metadata": False}
        )
        documents.extend(docs)

    if not documents:
        return "❌ No documents loaded.", []

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)

    reranker = LLMRerank(top_n=2)
    compressor = SentenceEmbeddingOptimizer(percentile_cutoff=0.4)

    start = time.time()
    query_engine = index.as_query_engine(
        similarity_top_k=8,
        node_postprocessors=[reranker, compressor]
    )
    response = query_engine.query(query)
    end = time.time()

    chunks = [node.text for node in response.source_nodes]

    context_tokens = sum(count_tokens(c) for c in chunks)

    answer = f"""
🧠 ANSWER
{response.response}

-------------------------
📊 METRICS

Retrieved (initial) : 8 chunks
After Rerank        : Top 2 chunks
After Compression   : Top 40% sentences kept
Final Context Tokens: {context_tokens}
Latency             : {round(end - start, 2)} seconds
"""

    return answer.strip(), chunks
