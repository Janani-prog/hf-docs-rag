---
title: HF Docs RAG
emoji: 🤗
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# HuggingFace Docs RAG

Production-grade Retrieval-Augmented Generation over HuggingFace documentation.

## What makes this production-grade

| Feature | Implementation |
|---|---|
| Hybrid retrieval | BM25 + semantic search via Reciprocal Rank Fusion |
| Reranking | Cross-encoder reranking on top-10 candidates |
| Citation enforcement | Refuses to answer if context is insufficient |
| Prompt versioning | Prompts stored as versioned files in registry.json |
| Evaluation pipeline | Faithfulness, relevancy, and precision scoring |
| CI/CD gating | GitHub Actions runs eval on every push |

## Stack

- LLM: Groq API (llama-3.3-70b-versatile)
- Embeddings: all-MiniLM-L6-v2 (local)
- Vector store: ChromaDB
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
- Tracing: LangFuse
- CI/CD: GitHub Actions

## Quick start

pip install -r requirements.txt
cp .env.example .env
python src/ingestion/scraper.py
python src/ingestion/chunker.py
python src/ingestion/embedder.py
python ask.py "How do I load a dataset in streaming mode?"
