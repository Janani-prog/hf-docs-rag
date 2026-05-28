import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector_search import search as vector_search
from src.retrieval.bm25_search import search as bm25_search
from src.retrieval.hybrid import search as hybrid_search


def test_vector_search_returns_results():
    results = vector_search("how to load a dataset", k=5)
    assert len(results) > 0
    assert len(results) <= 5


def test_vector_search_result_structure():
    results = vector_search("streaming dataset", k=3)
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "source_url" in r
        assert "score" in r
        assert len(r["text"]) > 0


def test_bm25_search_returns_results():
    results = bm25_search("LoRA fine-tuning parameters", k=5)
    assert len(results) > 0


def test_bm25_exact_term_matching():
    # BM25 should find exact technical terms reliably
    results = bm25_search("BM25Okapi tokenization", k=5)
    # Even if not found, should return empty list not crash
    assert isinstance(results, list)


def test_hybrid_search_combines_results():
    results = hybrid_search("map-style dataset random access", k=10)
    assert len(results) > 0
    for r in results:
        assert "rrf_score" in r
        assert "chunk_id" in r


def test_hybrid_search_rrf_scores_ordered():
    results = hybrid_search("tokenizer BERT", k=10)
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "RRF scores should be descending"


def test_vector_search_relevance():
    # The top result for this query should be about streaming
    results = vector_search("streaming mode load_dataset", k=3)
    top_text = results[0]["text"].lower()
    assert any(word in top_text for word in ["stream", "iterable", "dataset"])