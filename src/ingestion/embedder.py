import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS_FILE = "data/processed/chunks.jsonl"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hf_docs"
BATCH_SIZE = 64  # embed 64 chunks at a time — fits comfortably in CPU memory

# all-MiniLM-L6-v2 is small (80MB), fast on CPU, and produces 384-dim vectors
# Good enough for our use case and runs entirely offline
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def run():
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First run downloads ~80MB — subsequent runs are instant)\n")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading chunks from {CHUNKS_FILE}")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")

    # ChromaDB persistent client — data survives between runs
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # If the collection already exists with the same number of docs, skip
    existing = client.list_collections()
    existing_names = [c.name for c in existing]

    if COLLECTION_NAME in existing_names:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == len(chunks):
            print(f"Collection '{COLLECTION_NAME}' already has {len(chunks)} chunks. Skipping.")
            return
        else:
            print(f"Collection exists but has {collection.count()} chunks (expected {len(chunks)}). Recreating.")
            client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # cosine similarity is better than euclidean for text embeddings
        metadata={"hnsw:space": "cosine"}
    )

    print("Embedding and storing chunks...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i: i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]

        # Strip fields that aren't needed in ChromaDB metadata
        # ChromaDB metadata values must be str, int, float, or bool
        metadatas = [{
            "source_url": c["source_url"],
            "title": c["title"],
            "section": c["section"],
            "chunk_index": c["chunk_index"],
            "token_count": c["token_count"],
            "slug": c["slug"]
        } for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"\nDone. {collection.count()} chunks stored in ChromaDB at '{CHROMA_DIR}/'")
    print("Your vector store is ready for search.")


if __name__ == "__main__":
    run()