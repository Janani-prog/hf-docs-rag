import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import clean_content, split_into_chunks


def test_clean_content_removes_nav():
    noisy = "# Title\n🏡 View all docsAWSAccelerate\n## Real content\nThis is the article."
    cleaned = clean_content(noisy)
    assert "🏡" not in cleaned
    assert "Real content" in cleaned


def test_clean_content_removes_banner():
    noisy = "# Title\n![Hugging Face logo]\nJoin the Hugging Face community\n## Actual content\nHello."
    cleaned = clean_content(noisy)
    assert "Join the Hugging Face community" not in cleaned
    assert "Actual content" in cleaned


def test_split_respects_chunk_size():
    long_text = " ".join(["word"] * 2000)
    meta = {
        "slug": "test_doc",
        "source_url": "https://example.com",
        "title": "Test",
        "section": "test",
        "scraped_at": "2026-01-01"
    }
    chunks = split_into_chunks(long_text, meta)
    for chunk in chunks:
        # Allow some tolerance for overlap
        assert chunk["token_count"] <= 600, f"Chunk too large: {chunk['token_count']}"


def test_split_preserves_metadata():
    text = "## Section\n\nSome content here.\n\nMore content."
    meta = {
        "slug": "test_doc",
        "source_url": "https://example.com/test",
        "title": "Test Doc",
        "section": "transformers",
        "scraped_at": "2026-01-01"
    }
    chunks = split_into_chunks(text, meta)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["source_url"] == "https://example.com/test"
        assert chunk["title"] == "Test Doc"
        assert "chunk_id" in chunk


def test_split_produces_overlapping_chunks():
    # With enough text to create multiple chunks, verify chunk IDs are sequential
    long_text = "\n\n".join([f"Paragraph {i}: " + "content " * 80 for i in range(20)])
    meta = {
        "slug": "overlap_test",
        "source_url": "https://example.com",
        "title": "Overlap Test",
        "section": "test",
        "scraped_at": "2026-01-01"
    }
    chunks = split_into_chunks(long_text, meta)
    assert len(chunks) > 1
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))