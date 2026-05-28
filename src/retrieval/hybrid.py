from src.retrieval.vector_search import search as vector_search
from src.retrieval.bm25_search import search as bm25_search

RRF_K = 60  # standard constant — dampens the impact of rank differences


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict]
) -> list[dict]:
    """
    Combines two ranked lists into one using Reciprocal Rank Fusion.

    Why RRF works well: instead of trying to normalize scores from two
    completely different scoring systems (cosine similarity vs BM25),
    we only use the *rank position* of each result. A chunk ranked 1st
    by both systems gets a very high combined score. A chunk ranked 10th
    by both gets a low score. This is robust and requires no tuning.

    Formula: RRF(d) = sum(1 / (k + rank)) across all lists
    """
    scores = {}  # chunk_id -> combined RRF score
    chunk_map = {}  # chunk_id -> chunk dict (for final assembly)

    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        chunk_map[cid] = chunk

    # Sort by combined RRF score
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    results = []
    for cid in ranked_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = round(scores[cid], 6)
        results.append(chunk)

    return results


def search(query: str, k: int = 10) -> list[dict]:
    """
    Hybrid search: BM25 + semantic vector search combined via RRF.
    Returns top-k chunks ranked by combined relevance.
    """
    vec_results = vector_search(query, k=k)
    bm25_results = bm25_search(query, k=k)
    combined = reciprocal_rank_fusion(vec_results, bm25_results)
    return combined[:k]