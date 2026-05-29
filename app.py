import gradio as gr
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ensure_vector_store():
    if not os.path.exists("chroma_db"):
        print("Building vector store...")
        from src.ingestion.scraper import run as scrape
        from src.ingestion.chunker import run as chunk
        from src.ingestion.embedder import run as embed
        scrape()
        chunk()
        embed()


ensure_vector_store()

from src.retrieval.hybrid import search as hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.generator import query as rag_query

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0d0d0d !important;
    color: #f5f5f5 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 48px 32px !important;
}

.header-block {
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 28px;
    margin-bottom: 36px;
}

.header-block h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 20px !important;
    font-weight: 500 !important;
    color: #f0f0f0 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}

.header-block p {
    font-size: 14px !important;
    color: #888 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    line-height: 1.7 !important;
    max-width: 560px !important;
}

.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #bdbdbd;
    border: 1px solid #2a2a2a;
    padding: 3px 9px;
    margin-right: 6px;
    margin-top: 14px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 10px;
}

textarea {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 0 !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    transition: border-color 0.15s !important;
    resize: none !important;
    line-height: 1.6 !important;
}

textarea:focus {
    border-color: #555 !important;
    outline: none !important;
    box-shadow: none !important;
}

button.primary {
    background: #f0f0f0 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 13px 32px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    margin-top: 10px !important;
}

button.primary:hover { background: #d8d8d8 !important; }
button.primary:disabled {
    background: #222 !important;
    color: #555 !important;
    cursor: not-allowed !important;
}

button.secondary {
    background: transparent !important;
    color: #666 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    padding: 7px 10px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 5px !important;
    line-height: 1.5 !important;
}

button.secondary:hover {
    color: #bbb !important;
    border-color: #333 !important;
    background: #111 !important;
}

/* Pipeline trace panel */
.pipeline-panel {
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    padding: 16px 20px;
    margin-top: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 2;
    color: #c0c0c0;
    min-height: 80px;
}

.pipeline-panel .step-done { color: #6a9955; }
.pipeline-panel .step-active { color: #f5f5f5; }
.pipeline-panel .step-pending { color: #333; }

/* Answer panel */
.prose {
    background: #121212 !important;
    border: 1px solid #3a3a3a !important;
    padding: 20px !important;
    min-height: 100px !important;
}

.prose p {
    color: #fafafa  !important;
    font-size: 14px !important;
    line-height: 1.85 !important;
    margin-bottom: 14px !important;
}

.prose code {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    background: #141414 !important;
    border: 1px solid #222 !important;
    padding: 2px 7px !important;
    color: #9cdcfe !important;
}

.prose pre {
    background: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    padding: 16px !important;
    font-size: 12px !important;
    overflow-x: auto !important;
    margin: 14px 0 !important;
}

.prose hr {
    border: none !important;
    border-top: 1px solid #1e1e1e !important;
    margin: 16px 0 !important;
}

.prose em { color: #888 !important; font-size: 11px !important; }

/* Sources */
.sources-panel {
    background: #0a0a0a !important;
    border: 1px solid #1e1e1e !important;
    padding: 16px 20px !important;
    min-height: 60px !important;
}

.sources-panel p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #d0d0d0  !important;
    line-height: 2 !important;
}

.sources-panel a {
    color: #7a9ec2 !important;
    text-decoration: none !important;
}

.sources-panel a:hover { color: #a0bcd8 !important; }

hr { border: none; border-top: 1px solid #1a1a1a; margin: 32px 0; }
footer { display: none !important; }
.show-api { display: none !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2a2a2a; }
"""


def run_pipeline(question: str):
    """
    Generator function — yields pipeline status updates as each stage
    completes, then yields the final answer. Gradio streams these live.
    """
    if not question.strip():
        yield "", "", ""
        return

    # Stage 1
    pipeline_log = (
        "**Pipeline Trace**\n\n"
        "`→` Hybrid search (BM25 + semantic)...  \n"
        "`·` Cross-encoder reranking  \n"
        "`·` LLM generation  \n"
        "`·` Citation validation  \n"
    )
    yield pipeline_log, "", ""

    t0 = time.time()
    raw_chunks = hybrid_search(question, k=10)
    retrieval_ms = (time.time() - t0) * 1000

    # Stage 2
    pipeline_log = (
        "**Pipeline Trace**\n\n"
        f"`✓` Hybrid search — {len(raw_chunks)} candidates "
        f"in {retrieval_ms:.0f}ms  \n"
        "`→` Cross-encoder reranking...  \n"
        "`·` LLM generation  \n"
        "`·` Citation validation  \n"
    )
    yield pipeline_log, "", ""

    t1 = time.time()
    chunks = rerank(question, raw_chunks, top_k=5)
    rerank_ms = (time.time() - t1) * 1000

    top_sources = list({c["source_url"].split("/")[-1] for c in chunks[:3]})

    # Stage 3
    pipeline_log = (
        "**Pipeline Trace**\n\n"
        f"`✓` Hybrid search — {len(raw_chunks)} candidates "
        f"in {retrieval_ms:.0f}ms  \n"
        f"`✓` Reranking — top 5 selected in {rerank_ms:.0f}ms  \n"
        f"`→` LLM generation (llama-3.3-70b)...  \n"
        "`·` Citation validation  \n"
    )
    yield pipeline_log, "", ""

    result = rag_query(question)
    answer = result["answer"]

    if result["insufficient_context"]:
        answer = "The retrieved documentation does not contain sufficient information to answer this question."

    # Stage 4
    valid_citations = [c for c in result["citations"] if c["valid"]]
    pipeline_log = (
        "**Pipeline Trace**\n\n"
        f"`✓` Hybrid search — {len(raw_chunks)} candidates "
        f"in {retrieval_ms:.0f}ms  \n"
        f"`✓` Reranking — top 5 selected in {rerank_ms:.0f}ms  \n"
        f"`✓` LLM generation — {result['tokens_used']} tokens, "
        f"{result['latency']['generation_ms']:.0f}ms  \n"
        f"`→` Citation validation...  \n"
    )
    yield pipeline_log, "", ""

    # Final
    citation_status = (
        f"{len(valid_citations)} valid citation(s)"
        if valid_citations else "no citations — INSUFFICIENT_CONTEXT"
    )
    total_ms = retrieval_ms + rerank_ms + result["latency"]["generation_ms"]

    pipeline_log = (
        "**Pipeline Trace**\n\n"
        f"`✓` Hybrid search — {len(raw_chunks)} candidates "
        f"in {retrieval_ms:.0f}ms  \n"
        f"`✓` Reranking — top 5 selected in {rerank_ms:.0f}ms  \n"
        f"`✓` LLM generation — {result['tokens_used']} tokens, "
        f"{result['latency']['generation_ms']:.0f}ms  \n"
        f"`✓` Citation validation — {citation_status}  \n\n"
        f"*Total: {total_ms:.0f}ms · "
        f"Prompt: {result['prompt_version']} · "
        f"Model: {result['model']}*"
    )

    latency = result.get("latency", {})
    meta = (
        f"\n\n---\n"
        f"*`retrieval {retrieval_ms:.0f}ms` &nbsp;"
        f"`rerank {rerank_ms:.0f}ms` &nbsp;"
        f"`generation {latency.get('generation_ms', 0):.0f}ms` &nbsp;"
        f"`{result['tokens_used']} tokens`*"
    )

    seen = set()
    source_lines = []
    for c in result["citations"]:
        if c["valid"] and c["source_url"] not in seen:
            seen.add(c["source_url"])
            source_lines.append(
                f"[{c['source_url']}]({c['source_url']})"
            )
    sources = "\n\n".join(source_lines) if source_lines else "*No sources cited.*"

    yield pipeline_log, answer + meta, sources


EXAMPLES = [
    "How do I load a dataset in streaming mode?",
    "What is the difference between map-style and iterable datasets?",
    "How does LoRA reduce the number of trainable parameters?",
    "How do I apply a function to every example in a dataset?",
    "How do I save a fine-tuned model locally?",
]

with gr.Blocks(css=CUSTOM_CSS, title="HF Docs RAG") as demo:

    gr.Markdown("""
<div class="header-block">
<h1>HF Docs RAG</h1>
<p>Production-grade retrieval-augmented generation over HuggingFace documentation.
Answers are grounded in retrieved source passages. Every claim is cited.</p>
<span class="tag">Hybrid BM25 + Semantic Search</span>
<span class="tag">Cross-encoder Reranking</span>
<span class="tag">Citation Enforcement</span>
<span class="tag">Groq / Llama 3.3-70b</span>
</div>
""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            gr.Markdown('<div class="section-label">Query</div>')
            question = gr.Textbox(
                show_label=False,
                placeholder="Ask a question about Transformers, Datasets, PEFT, or Tokenizers...",
                lines=3,
            )
            submit = gr.Button("Run Query", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown('<div class="section-label">Examples</div>')
            for ex in EXAMPLES:
                gr.Button(ex, variant="secondary", size="sm").click(
                    fn=lambda q=ex: q, outputs=question
                )

    gr.Markdown(
        '<div class="section-label" style="margin-top:28px">Pipeline</div>'
    )
    pipeline_out = gr.Markdown(
        value="*Waiting for query...*",
        elem_classes=["pipeline-panel"]
    )

    gr.Markdown(
        '<div class="section-label" style="margin-top:20px">Answer</div>'
    )
    answer_out = gr.Markdown(elem_classes=["prose"])

    gr.Markdown(
        '<div class="section-label" style="margin-top:20px">Sources</div>'
    )
    sources_out = gr.Markdown(elem_classes=["sources-panel"])

    gr.Markdown("""
---
<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#333;text-align:center;padding:4px 0">
query → hybrid search (BM25 + vector) → RRF fusion → cross-encoder rerank → llm generation → citation validation
</div>
""")

    submit.click(
        fn=run_pipeline,
        inputs=question,
        outputs=[pipeline_out, answer_out, sources_out],
    )
    question.submit(
        fn=run_pipeline,
        inputs=question,
        outputs=[pipeline_out, answer_out, sources_out],
    )

if __name__ == "__main__":
    demo.launch()