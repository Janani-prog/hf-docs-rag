---
title: HF Docs RAG
emoji: 🤗
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "5.32.1"
app_file: app.py
pinned: false
---

# hf-docs-rag

Production-grade Retrieval-Augmented Generation over HuggingFace documentation. Built to demonstrate real AI engineering practices — not a tutorial project.

**Live demo:** [jananijayalakshmi/hf-docs-rag](https://huggingface.co/spaces/jananijayalakshmi/hf-docs-rag)

---

## What this is

Most RAG demos retrieve the top-k chunks and hand them to an LLM. This system does considerably more:

- Runs **two retrieval methods in parallel** — BM25 keyword search and semantic vector search — then fuses their rankings using Reciprocal Rank Fusion. Neither method alone is sufficient: vector search understands intent but misses exact terms; BM25 handles exact terms but misses semantics.
- Passes the fused candidates through a **cross-encoder reranker** that evaluates query and chunk together as a pair, rather than independently. This is slower but significantly more precise.
- Enforces **citation grounding** at the output layer. The LLM is instructed to cite every claim with a numbered reference. If the retrieved context does not support an answer, the system returns `INSUFFICIENT_CONTEXT` rather than hallucinating.
- Versions **prompts as files** tracked in a registry. A prompt change is treated like a code change — committed, reviewed, and evaluated against a baseline before deployment.
- Runs **automated evaluation** on every push. A golden dataset of curated Q&A pairs measures faithfulness, answer relevancy, and context precision. If metrics drop below threshold, CI fails.
- Traces **every query end-to-end** through LangFuse, capturing per-stage latency, token usage, and citation validity.

---

## Architecture

```
Query
  │
  ├─► BM25 Search ──────────────────┐
  │   (rank_bm25, keyword matching)  │
  │                                  ▼
  └─► Vector Search ────────► RRF Fusion (top-10)
      (ChromaDB, MiniLM-L6-v2)       │
                                      ▼
                              Cross-Encoder Reranker
                              (ms-marco-MiniLM-L-6-v2, top-5)
                                      │
                                      ▼
                              Versioned Prompt + Context
                                      │
                                      ▼
                              Groq LLM (llama-3.3-70b)
                                      │
                                      ▼
                              Citation Validation
                                      │
                                      ▼
                              Structured Response
                              { answer, citations, latency, tokens }
```

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| LLM | Groq API — llama-3.3-70b-versatile | Free tier |
| Embeddings | all-MiniLM-L6-v2 | Runs locally, no API |
| Vector store | ChromaDB | Persistent, local |
| Keyword search | rank-bm25 | BM25Okapi, in-memory |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Runs locally |
| Evaluation | Custom scorer over Groq | Faithfulness, relevancy, precision |
| Tracing | LangFuse cloud | Per-stage spans, cost tracking |
| CI/CD | GitHub Actions | Unit tests + eval gate on every push |

Everything runs free. No paid APIs beyond Groq's free tier.

---

## Evaluation

Evaluation runs automatically on every push via GitHub Actions against a curated golden dataset of 12 question-answer pairs covering factual queries, multi-hop questions, and unanswerable questions.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.72 | 0.70 | PASS |
| Answer Relevancy | 0.90 | 0.70 | PASS |
| Context Precision | 0.57 | — | — |

Evaluated against 12 curated Q&A pairs covering factual queries, 
multi-hop questions, and unanswerable questions. Scores measured 
using llama-3.1-8b-instant as the eval model. CI fails automatically 
if either gated metric drops below threshold.
---

## Production metrics

From LangFuse traces across live queries:

| Metric | Value |
|---|---|
| Citation coverage | 93.3% |
| Insufficient context rate | 13.3% |
| Avg tokens per query | 2,396 |
| Avg cost per query | $0.0014 |
| Retrieval latency p50 | 1,703ms |
| Generation latency p50 | 1,191ms |

---

## Repository structure

```
src/
  ingestion/
    scraper.py          # Crawls HuggingFace docs, saves raw JSON
    chunker.py          # Splits docs into 500-token overlapping chunks
    embedder.py         # Embeds chunks into ChromaDB via MiniLM
  retrieval/
    bm25_search.py      # BM25Okapi keyword search
    vector_search.py    # Semantic search via ChromaDB
    hybrid.py           # Reciprocal Rank Fusion
    reranker.py         # Cross-encoder reranking
    generator.py        # LLM call with citation enforcement
    prompt_manager.py   # Loads prompts from versioned registry
  evaluation/
    ragas_eval.py       # Evaluation pipeline with custom scorer
  monitoring/
    langfuse_tracer.py  # Per-stage tracing
    dashboard.py        # Metrics dashboard from LangFuse API

prompts/
  v1/rag_answer.txt     # Prompt version 1
  v2/rag_answer.txt     # Prompt version 2 (current)
  registry.json         # Version registry

golden_dataset/
  golden_qa.json        # 12 curated Q&A pairs

tests/
  test_chunker.py       # 5 unit tests
  test_retrieval_integration.py

.github/workflows/
  ci.yml                # Unit tests on every push
```

---

## Running locally

```bash
git clone https://github.com/Janani-prog/hf-docs-rag
cd hf-docs-rag
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add GROQ_API_KEY to .env

python src/ingestion/scraper.py
python src/ingestion/chunker.py
python src/ingestion/embedder.py

# Ask a question
python ask.py "How do I load a dataset in streaming mode?"

# Run evaluation
python src/evaluation/ragas_eval.py

# View metrics dashboard
python src/monitoring/dashboard.py

# Launch UI
python app.py
```

---

## CI/CD

GitHub Actions runs on every push to `main`:

1. Installs dependencies
2. Runs unit tests (`pytest tests/test_chunker.py`)
3. Evaluation gate runs separately — fails if faithfulness or relevancy drops more than 5% from the saved baseline

Prompt changes, dependency updates, and chunking parameter changes all go through this gate before merging.

---

## Design decisions worth noting

**Why hybrid retrieval over pure vector search?** Vector search understands meaning but struggles with exact technical terms like class names and method signatures. BM25 handles these precisely. RRF combines both without requiring score normalization across incompatible scales.

**Why a cross-encoder reranker?** Bi-encoder models (used in vector search) embed query and document independently and compare vectors. Cross-encoders see the query and document together, allowing the model to reason about their relationship directly. More accurate, but too slow to run over the full corpus — so we apply it only to the top-10 candidates from hybrid search.

**Why citation enforcement?** Generating plausible-sounding answers is trivial. Grounding every claim in retrieved evidence is not. The system is explicitly designed to fail loudly (`INSUFFICIENT_CONTEXT`) rather than hallucinate silently. This is the difference between a demo and a system you can trust.

**Why versioned prompts?** A prompt change can alter system behavior as dramatically as a code change. Storing prompts as versioned files in a registry makes changes auditable, reversible, and testable against the eval baseline.