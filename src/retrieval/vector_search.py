import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hf_docs"
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once, reused across queries
_model = None
_collection = None


def _load():
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)


def search(query: str, k: int = 10) -> list[dict]:
    """
    Embeds the query and finds the k most semantically similar chunks.
    Returns a list of dicts with text, metadata, and similarity score.
    """
    _load()

    query_embedding = _model.encode(query).tolist()

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source_url": results["metadatas"][0][i]["source_url"],
            "title": results["metadatas"][0][i]["title"],
            "section": results["metadatas"][0][i]["section"],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })

    return chunks