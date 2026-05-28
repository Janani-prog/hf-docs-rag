import json
import os
import re
import tiktoken
from tqdm import tqdm

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CHUNKS_FILE = os.path.join(PROCESSED_DIR, "chunks.jsonl")

CHUNK_SIZE = 500      # tokens per chunk
CHUNK_OVERLAP = 50    # tokens of overlap between consecutive chunks

# We use the GPT-2 tokenizer just for counting tokens — it's a close enough
# approximation and works offline without any API calls
tokenizer = tiktoken.get_encoding("gpt2")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def clean_content(text: str) -> str:
    """
    Strips HuggingFace site-wide nav and signup banner noise.

    Two noise patterns exist on every page:
    1. Nav dump: long line of product names with no spaces
    2. Signup banner: 'Join the Hugging Face community...' block
    """
    # Find the first real heading
    match = re.search(r"^#{1,3} ", text, re.MULTILINE)
    if match:
        text = text[match.start():]

    # Known noise phrases that appear in the banner block
    NOISE_PHRASES = [
        "🏡",
        "View all docs",
        "Join the Hugging Face community",
        "get access to the augmented documentation",
        "Collaborate on models, datasets and Spaces",
        "Faster examples with accelerated inference",
        "Switch between documentation themes",
        "to get started",
        "huggingface_logo",
        "![Hugging Face",
    ]

    cleaned_lines = []
    for line in text.splitlines():
        is_noise = (
            any(phrase in line for phrase in NOISE_PHRASES) or
            (len(line) > 200 and line.count(" ") < len(line) / 15)
        )
        if not is_noise:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_chunks(text: str, source_meta: dict) -> list[dict]:
    """
    Splits a document into overlapping token-based chunks.

    Why overlapping? If a key sentence falls right at a chunk boundary,
    overlap ensures it appears in at least one chunk fully intact.

    Strategy:
    1. Split on paragraph boundaries first (preserves natural breaks)
    2. Accumulate paragraphs until we hit CHUNK_SIZE tokens
    3. When we do, save the chunk and backtrack CHUNK_OVERLAP tokens
       worth of text to start the next chunk (the overlap)
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_tokens = []   # list of token ids for current chunk
    current_text_parts = []  # list of paragraph strings for current chunk
    chunk_index = 0

    for para in paragraphs:
        para_tokens = tokenizer.encode(para)

        # Hard split: if a single paragraph exceeds CHUNK_SIZE on its own,
        # break it into word-level chunks before processing
        # This handles cases like nav text or code blocks with no line breaks
        if len(para_tokens) > CHUNK_SIZE:
            words = para.split()
            sub_para = []
            sub_tokens = []
            for word in words:
                word_tokens = tokenizer.encode(" " + word)
                if sub_tokens and len(sub_tokens) + len(word_tokens) > CHUNK_SIZE:
                    paragraphs.append(" ".join(sub_para))
                    sub_para = [word]
                    sub_tokens = word_tokens
                else:
                    sub_para.append(word)
                    sub_tokens.extend(word_tokens)
            if sub_para:
                paragraphs.append(" ".join(sub_para))
            continue  # skip the original oversized paragraph

        # If adding this paragraph would exceed CHUNK_SIZE, flush current chunk
        if current_tokens and (len(current_tokens) + len(para_tokens)) > CHUNK_SIZE:
            chunk_text = "\n\n".join(current_text_parts)
            chunks.append({
                "chunk_id": f"{source_meta['slug']}__chunk_{chunk_index}",
                "text": chunk_text,
                "token_count": len(current_tokens),
                "chunk_index": chunk_index,
                **source_meta
            })

            # Prepare overlap: keep last CHUNK_OVERLAP tokens as the start
            if CHUNK_OVERLAP > 0 and len(current_tokens) >= CHUNK_OVERLAP:
                overlap_tokens = current_tokens[-CHUNK_OVERLAP:]
                try:
                    overlap_text = tokenizer.decode(overlap_tokens)
                except Exception:
                    overlap_text = ""
                current_tokens = overlap_tokens.copy()
                current_text_parts = [overlap_text] if overlap_text.strip() else []
            else:
                current_tokens = []
                current_text_parts = []

            chunk_index += 1

        # Add paragraph to current chunk
        current_tokens.extend(para_tokens)
        current_text_parts.append(para)

    # Don't forget the last chunk
    if current_text_parts:
        chunk_text = "\n\n".join(current_text_parts)
        chunks.append({
            "chunk_id": f"{source_meta['slug']}__chunk_{chunk_index}",
            "text": chunk_text,
            "token_count": len(current_tokens),
            "chunk_index": chunk_index,
            **source_meta
        })

    return chunks


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]
    print(f"Found {len(raw_files)} raw documents\n")

    all_chunks = []

    for filename in tqdm(raw_files):
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, encoding="utf-8") as f:
            doc = json.load(f)

        content = clean_content(doc["content"])

        if len(content) < 100:  # skip near-empty docs after cleaning
            continue

        source_meta = {
            "slug": filename.replace(".json", ""),
            "source_url": doc["url"],
            "title": doc["title"],
            "section": doc.get("section", "unknown"),
            "scraped_at": doc.get("scraped_at", "")
        }

        chunks = split_into_chunks(content, source_meta)
        all_chunks.extend(chunks)

    # Save as JSONL — one chunk per line, easy to stream later
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_chunks)} chunks from {len(raw_files)} documents")
    print(f"Saved to {CHUNKS_FILE}")

    # Quick stats
    token_counts = [c["token_count"] for c in all_chunks]
    print(f"Avg chunk size: {sum(token_counts) // len(token_counts)} tokens")
    print(f"Min: {min(token_counts)}  Max: {max(token_counts)}")


if __name__ == "__main__":
    run()