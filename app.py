import gradio as gr
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ensure_vector_store():
    if not os.path.exists("chroma_db"):
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

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500;600&family=Geist:wght@300;400;500&display=swap');

:root {
    --bg:        #080808;
    --surface:   #101010;
    --border:    #1f1f1f;
    --border2:   #2a2a2a;
    --text:      #e8e8e8;
    --text-dim:  #888;
    --text-faint:#444;
    --accent:    #e8e8e8;
    --green:     #4ade80;
    --amber:     #fbbf24;
    --blue:      #60a5fa;
    --red:       #f87171;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 14px !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ── Header ── */
.hdr {
    padding: 28px 36px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 24px;
}
.hdr-title {
    font-family: 'Geist Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: .12em;
    text-transform: uppercase;
}
.hdr-sub {
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1;
}
.hdr-tags {
    margin-left: auto;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.chip {
    font-family: 'Geist Mono', monospace;
    font-size: 9px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--text-faint);
    border: 1px solid var(--border);
    padding: 3px 8px;
}

/* ── Main layout ── */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 260px;
    min-height: calc(100vh - 73px);
}

/* ── Left panel ── */
.left-panel {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
}

.query-block {
    padding: 24px 28px 20px;
    border-bottom: 1px solid var(--border);
}

.field-label {
    font-family: 'Geist Mono', monospace;
    font-size: 9px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-bottom: 8px;
}

textarea {
    width: 100% !important;
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
    padding: 12px 14px !important;
    resize: none !important;
    transition: border-color .12s !important;
}
textarea:focus {
    border-color: #3a3a3a !important;
    outline: none !important;
    box-shadow: none !important;
}
textarea::placeholder { color: var(--text-faint) !important; }

/* Run button */
button.primary {
    margin-top: 12px !important;
    width: 100% !important;
    background: var(--text) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    padding: 13px !important;
    cursor: pointer !important;
    position: relative !important;
    transition: background .12s !important;
}
button.primary:hover  { background: #d0d0d0 !important; }
button.primary:active { background: #b8b8b8 !important; }

/* Loading pulse on button */
button.primary.generating {
    background: var(--surface) !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border2) !important;
    animation: pulse-btn 1.4s ease-in-out infinite !important;
}
@keyframes pulse-btn {
    0%, 100% { opacity: 1; }
    50%       { opacity: .5; }
}

/* Pipeline trace */
.trace-block {
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}
.trace-block .prose,
.trace-block .prose * {
    font-family: 'Geist Mono', monospace !important;
    font-size: 12px !important;
    line-height: 2 !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
}
.trace-block .prose code {
    font-family: 'Geist Mono', monospace !important;
    font-size: 11px !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    padding: 1px 6px !important;
    border-radius: 2px !important;
    color: var(--text-dim) !important;
}

/* Answer */
.answer-block {
    padding: 24px 28px;
    flex: 1;
}
.answer-block .prose p {
    color: var(--text) !important;
    font-size: 14px !important;
    line-height: 1.85 !important;
    margin-bottom: 14px !important;
}
.answer-block .prose code {
    font-family: 'Geist Mono', monospace !important;
    font-size: 12px !important;
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    padding: 2px 7px !important;
    color: var(--blue) !important;
}
.answer-block .prose pre {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    padding: 16px !important;
    font-size: 12px !important;
    overflow-x: auto !important;
    margin: 14px 0 !important;
    border-radius: 2px !important;
}
.answer-block .prose hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 16px 0 !important;
}
.answer-block .prose em {
    color: var(--text-faint) !important;
    font-size: 11px !important;
    font-style: normal !important;
    font-family: 'Geist Mono', monospace !important;
}
.answer-block .prose strong { color: var(--text) !important; }

/* ── Right panel ── */
.right-panel {
    display: flex;
    flex-direction: column;
}

.examples-block {
    padding: 20px 20px 16px;
    border-bottom: 1px solid var(--border);
}

button.secondary {
    background: transparent !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 10px !important;
    line-height: 1.5 !important;
    padding: 8px 10px !important;
    cursor: pointer !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 5px !important;
    transition: all .1s !important;
}
button.secondary:hover {
    color: var(--text) !important;
    border-color: var(--border2) !important;
    background: var(--surface) !important;
}

/* Sources panel */
.sources-block {
    padding: 20px;
    flex: 1;
    overflow-y: auto;
}
.sources-block .prose p {
    font-family: 'Geist Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-dim) !important;
    line-height: 1.8 !important;
    margin-bottom: 10px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid var(--border) !important;
    word-break: break-all !important;
}
.sources-block .prose p:last-child { border-bottom: none !important; }
.sources-block .prose a {
    color: var(--blue) !important;
    text-decoration: none !important;
}
.sources-block .prose a:hover { color: #93c5fd !important; }
.sources-block .prose strong {
    color: var(--text) !important;
    display: block !important;
    font-size: 11px !important;
    margin-bottom: 2px !important;
}

/* Stat bar */
.stat-bar {
    padding: 14px 20px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
}
.stat-key { color: var(--text-faint); }
.stat-val { color: var(--text-dim); }

/* Hide Gradio chrome */
footer, .show-api, #component-0 > .gap { display: none !important; }
.contain { padding: 0 !important; gap: 0 !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: var(--border2); }
"""

EXAMPLES = [
    "How do I load a dataset in streaming mode?",
    "What is the difference between map-style and iterable datasets?",
    "How does LoRA reduce the number of trainable parameters?",
    "How do I apply a function to every example in a dataset?",
    "How do I save a fine-tuned model to disk?",
    "What does Arrow format provide to HuggingFace Datasets?",
]


def run_pipeline(question: str):
    if not question.strip():
        yield "—", "", ""
        return

    # Stage 1 — retrieval
    yield (
        "`→` hybrid search (bm25 + semantic)  \n"
        "`·` cross-encoder reranking  \n"
        "`·` llm generation  \n"
        "`·` citation validation",
        "", ""
    )

    t0 = time.time()
    raw_chunks = hybrid_search(question, k=10)
    retrieval_ms = (time.time() - t0) * 1000

    # Stage 2 — reranking
    yield (
        f"`✓` hybrid search — {len(raw_chunks)} candidates · {retrieval_ms:.0f}ms  \n"
        "`→` cross-encoder reranking  \n"
        "`·` llm generation  \n"
        "`·` citation validation",
        "", ""
    )

    t1 = time.time()
    chunks = rerank(question, raw_chunks, top_k=5)
    rerank_ms = (time.time() - t1) * 1000

    # Stage 3 — generation
    yield (
        f"`✓` hybrid search — {len(raw_chunks)} candidates · {retrieval_ms:.0f}ms  \n"
        f"`✓` reranking — top 5 selected · {rerank_ms:.0f}ms  \n"
        "`→` llm generation (llama-3.3-70b-versatile)  \n"
        "`·` citation validation",
        "", ""
    )

    result = rag_query(question)
    answer = result["answer"]
    gen_ms = result["latency"]["generation_ms"]

    if result["insufficient_context"]:
        answer = "The retrieved documentation does not contain sufficient information to answer this question."

    # Stage 4 — citations
    valid_cites = [c for c in result["citations"] if c["valid"]]
    cite_status = f"{len(valid_cites)} citation(s) validated" if valid_cites else "no citations — INSUFFICIENT_CONTEXT"

    total_ms = retrieval_ms + rerank_ms + gen_ms

    trace = (
        f"`✓` hybrid search — {len(raw_chunks)} candidates · {retrieval_ms:.0f}ms  \n"
        f"`✓` reranking — top 5 selected · {rerank_ms:.0f}ms  \n"
        f"`✓` llm generation — {result['tokens_used']} tokens · {gen_ms:.0f}ms  \n"
        f"`✓` citation validation — {cite_status}  \n\n"
        f"*total {total_ms:.0f}ms · prompt/{result['prompt_version']} · {result['model']}*"
    )

    # Format answer with clean meta line
    meta = (
        f"\n\n---\n"
        f"*retrieval {retrieval_ms:.0f}ms · "
        f"rerank {rerank_ms:.0f}ms · "
        f"generation {gen_ms:.0f}ms · "
        f"{result['tokens_used']} tokens*"
    )

    # Format sources as titled entries, not [1][2]
    seen = set()
    source_parts = []
    for c in result["citations"]:
        if c["valid"] and c["source_url"] not in seen:
            seen.add(c["source_url"])
            # Extract a clean page name from the URL
            page = c["source_url"].rstrip("/").split("/")[-1].replace("_", " ").replace("-", " ").title()
            section = c["source_url"].split("/docs/")[-1].split("/")[0].upper() if "/docs/" in c["source_url"] else ""
            source_parts.append(
                f"**{section} — {page}**  \n"
                f"[{c['source_url']}]({c['source_url']})"
            )

    sources = "\n\n".join(source_parts) if source_parts else "*No sources cited.*"

    yield trace, answer + meta, sources


with gr.Blocks(css=CSS, title="HF Docs RAG") as demo:

    # Header
    gr.HTML("""
    <div class="hdr">
        <span class="hdr-title">HF Docs RAG</span>
        <span class="hdr-sub">Retrieval-augmented generation over HuggingFace documentation</span>
        <div class="hdr-tags">
            <span class="chip">BM25 + Semantic</span>
            <span class="chip">Cross-encoder Rerank</span>
            <span class="chip">Citation Enforced</span>
            <span class="chip">Groq · Llama 3.3-70b</span>
        </div>
    </div>
    """)

    with gr.Row(elem_classes=["main-grid"]):

        # Left panel
        with gr.Column(elem_classes=["left-panel"]):

            with gr.Group(elem_classes=["query-block"]):
                gr.HTML('<div class="field-label">Query</div>')
                question = gr.Textbox(
                    show_label=False,
                    placeholder="Ask anything about Transformers, Datasets, PEFT, or Tokenizers...",
                    lines=3,
                )
                submit = gr.Button("Run Query →", variant="primary")

            with gr.Group(elem_classes=["trace-block"]):
                gr.HTML('<div class="field-label">Pipeline Trace</div>')
                trace_out = gr.Markdown(
                    value="*Waiting for query...*",
                    elem_classes=["prose"],
                )

            with gr.Group(elem_classes=["answer-block"]):
                gr.HTML('<div class="field-label">Answer</div>')
                answer_out = gr.Markdown(elem_classes=["prose"])

        # Right panel
        with gr.Column(elem_classes=["right-panel"]):

            with gr.Group(elem_classes=["examples-block"]):
                gr.HTML('<div class="field-label">Examples</div>')
                for ex in EXAMPLES:
                    gr.Button(ex, variant="secondary", size="sm").click(
                        fn=lambda q=ex: q, outputs=question
                    )

            with gr.Group(elem_classes=["sources-block"]):
                gr.HTML('<div class="field-label">Sources</div>')
                sources_out = gr.Markdown(
                    value="*Sources appear here after a query.*",
                    elem_classes=["prose"],
                )

    submit.click(
        fn=run_pipeline,
        inputs=question,
        outputs=[trace_out, answer_out, sources_out],
    )
    question.submit(
        fn=run_pipeline,
        inputs=question,
        outputs=[trace_out, answer_out, sources_out],
    )

if __name__ == "__main__":
    demo.launch()