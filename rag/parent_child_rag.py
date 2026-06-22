from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import UnstructuredReader


def run_rag(query: str, files: list[str]):

    reader = UnstructuredReader()
    documents = []

    for file in files:
        docs = reader.load_data(
            unstructured_kwargs={
                "filename": file,
                "metadata": False,
            }
        )
        documents.extend(docs)

    if not documents:
        return "❌ No documents loaded.", []

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2000, 400]
    )
    nodes = node_parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)

    retriever = index.as_retriever(similarity_top_k=10)
    retrieved_nodes = retriever.retrieve(query)

    if not retrieved_nodes:
        return "Not found in documents.", []

    context_chunks = [n.node.text for n in retrieved_nodes]
    context = "\n\n".join(context_chunks)

    # ✅ UPDATED PROMPT
    prompt = f"""
You are a helpful assistant.

Using the context below, answer the user's question.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""

    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.complete(prompt)

    answer = f"""
🧠 ANSWER
{response.text.strip()}

-------------------------
📄 PARENT–CHILD DETAILS

Parent Chunk Size : 2000
Child Chunk Size  : 400
Retrieved Chunks  : {len(context_chunks)}
"""

    return answer.strip(), context_chunks
