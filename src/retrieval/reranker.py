from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Loaded once at module level
_model = None


def _load():
    global _model
    if _model is None:
        print(f"Loading reranker: {MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME)


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder reranking — the most impactful retrieval improvement.

    The difference from vector search:
    - Vector search: embeds query and chunks *separately*, then compares.
      Fast, but the model never sees them together.
    - Cross-encoder: takes (query, chunk) as a *pair* and scores their
      relevance jointly. Much more accurate because the model sees both
      at once and can reason about exact relationships.

    The tradeoff: slower (can't pre-compute), so we only run it on the
    top 10 candidates from hybrid search, not all 1605 chunks.
    """
    _load()

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = _model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]