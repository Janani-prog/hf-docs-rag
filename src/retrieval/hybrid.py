from src.retrieval.vector_search import search as vector_search
from src.retrieval.bm25_search import search as bm25_search

RRF_K = 60
# Alpha controls the balance between semantic and keyword search.
# 0.0 = pure BM25, 1.0 = pure semantic, 0.5 = equal weight.
# Technical docs benefit from slightly higher BM25 weight (exact term matching).
ALPHA = 0.6


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    alpha: float = ALPHA
) -> list[dict]:
    """
    Weighted Reciprocal Rank Fusion.

    Standard RRF weights both lists equally. This implementation
    allows tuning via alpha:
      combined = alpha * (1 / (k + vector_rank))
              + (1 - alpha) * (1 / (k + bm25_rank))

    alpha=0.6 means semantic search contributes 60% of the score,
    BM25 contributes 40%. For highly technical corpora with specific
    class/method names, lowering alpha improves exact-match retrieval.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + alpha * (1 / (RRF_K + rank))
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + (1 - alpha) * (1 / (RRF_K + rank))
        chunk_map[cid] = chunk

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    results = []
    for cid in ranked_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = round(scores[cid], 6)
        results.append(chunk)

    return results


def search(query: str, k: int = 10, alpha: float = ALPHA) -> list[dict]:
    vec_results = vector_search(query, k=k)
    bm25_results = bm25_search(query, k=k)
    combined = reciprocal_rank_fusion(vec_results, bm25_results, alpha=alpha)
    return combined[:k]