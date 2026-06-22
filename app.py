import gradio as gr

from rag import (
    naive_rag,
    unstructured_rag,
    contextual_rag,
    fusion_rag,
    hybrid_rag,
    hyde_rag,
    parent_child_rag,
    rrr_rag,
    sentence_compression_rag,
    rerank_compress_rag,
)

RAG_MAP = {
    "Naive RAG": naive_rag,
    "Unstructured OCR RAG": unstructured_rag,
    "Contextual Compression RAG": contextual_rag,
    "Fusion RAG (RRF)": fusion_rag,
    "Hybrid RAG (Vector + BM25)": hybrid_rag,
    "HyDE RAG": hyde_rag,
    "Parent–Child RAG": parent_child_rag,
    "RRR RAG (Rewrite → Retrieve → Read)": rrr_rag,
    "Sentence Compression RAG": sentence_compression_rag,
    "Rerank + Compress RAG": rerank_compress_rag,
}


def run_selected_rag(rag_type, query, uploaded_files):
    if not uploaded_files:
        return "❌ Please upload at least one document.", ""

    if not query or not query.strip():
        return "❌ Please enter a query.", ""

    rag_fn = RAG_MAP.get(rag_type)
    if rag_fn is None:
        return "❌ Invalid RAG selection.", ""

    file_paths = [f.name for f in uploaded_files]

    try:
        answer, chunks = rag_fn(query, file_paths)
    except Exception as e:
        return f"❌ Error running {rag_type}:\n{str(e)}", ""

    formatted_chunks = "\n\n".join(
        [f"--- Chunk {i+1} ---\n{chunk}" for i, chunk in enumerate(chunks)]
    )

    return answer, formatted_chunks


with gr.Blocks(title="🧠 Advanced RAG Cookbook — Playground") as demo:
    gr.Markdown("## 🧠 Advanced RAG Cookbook — Playground")
    gr.Markdown(
        "Pick a RAG technique, upload your documents (PDF / DOC / TXT / images), "
        "ask a question, and see exactly which chunks were retrieved."
    )

    rag_selector = gr.Dropdown(
        choices=list(RAG_MAP.keys()),
        value="Naive RAG",
        label="RAG Technique"
    )

    file_upload = gr.File(
        file_count="multiple",
        label="Upload Documents"
    )

    query_input = gr.Textbox(
        label="Your Question",
        placeholder="e.g. What is the main topic of this document?",
        lines=2
    )

    run_btn = gr.Button("🚀 Run")

    gr.Markdown("---")

    answer_output = gr.Textbox(label="Answer", lines=8)
    chunks_output = gr.Textbox(label="Retrieved Chunks", lines=16)

    run_btn.click(
        fn=run_selected_rag,
        inputs=[rag_selector, query_input, file_upload],
        outputs=[answer_output, chunks_output]
    )

demo.launch()
