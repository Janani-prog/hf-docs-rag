import json
import os
from rank_bm25 import BM25Okapi

CHUNKS_FILE = "data/processed/chunks.jsonl"

# Module-level cache — BM25 index is built once and reused
_bm25 = None
_chunks = None


def _load():
    global _bm25, _chunks
    if _bm25 is not None:
        return

    _chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            _chunks.append(json.loads(line.strip()))

    # Tokenize: lowercase + whitespace split
    # Simple but effective for technical documentation
    tokenized = [c["text"].lower().split() for c in _chunks]
    _bm25 = BM25Okapi(tokenized)


def search(query: str, k: int = 10) -> list[dict]:
    """
    BM25 keyword search over all chunks.
    Excels at exact term matching — great for specific class names,
    method names, and technical jargon that vector search may miss.
    """
    _load()

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    # Get top-k indices sorted by score descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # skip zero-score results
            chunk = _chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)

    return results