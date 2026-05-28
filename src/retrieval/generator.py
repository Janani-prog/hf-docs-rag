import os
import re
from groq import Groq
from dotenv import load_dotenv
from src.retrieval.hybrid import search as hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.prompt_manager import get_prompt, get_current_version

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHUNKS = 5
PROMPT_PATH = "prompts/v1/rag_answer.txt"


def load_prompt_template() -> str:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def build_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a numbered context block.
    The numbers correspond to citation markers [1], [2] etc in the answer.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] SOURCE: {chunk['source_url']}\n"
            f"TITLE: {chunk['title']}\n\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Parses [N] markers from the answer and maps them back to source chunks.
    Any citation number that doesn't match a real chunk is flagged as invalid.
    """
    cited_numbers = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    citations = []

    for n in sorted(cited_numbers):
        if 1 <= n <= len(chunks):
            chunk = chunks[n - 1]
            citations.append({
                "number": n,
                "source_url": chunk["source_url"],
                "title": chunk["title"],
                "snippet": chunk["text"][:150] + "...",
                "valid": True
            })
        else:
            citations.append({
                "number": n,
                "valid": False,
                "reason": f"[{n}] cited but only {len(chunks)} chunks provided"
            })

    return citations


def query(question: str) -> dict:
    """
    Full RAG pipeline for a single question:
    1. Retrieve top-k relevant chunks via vector search
    2. Build numbered context block
    3. Call Groq LLM with prompt + context
    4. Parse citations and validate them
    5. Return structured response
    """
    # Step 1: retrieve
    raw_chunks = hybrid_search(question, k=10)
    chunks = rerank(question, raw_chunks, top_k=5)

    # Step 2: build context
    context = build_context(chunks)


    # Step 3: load prompt from registry (version-controlled)
    prompt_version = get_current_version()
    template = get_prompt(prompt_version)
    prompt = template.replace("{context}", context).replace("{question}", question)

    # Step 4: call LLM
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic — we want factual, not creative
    )

    answer = response.choices[0].message.content.strip()
    usage = response.usage

    # Step 5: validate citations
    citations = extract_citations(answer, chunks)
    invalid = [c for c in citations if not c["valid"]]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "citation_valid": len(invalid) == 0,
        "insufficient_context": answer.strip() == "INSUFFICIENT_CONTEXT",
        "retrieved_chunks": len(chunks),
        "tokens_used": usage.total_tokens,
        "model": GROQ_MODEL,
        "prompt_version": prompt_version,   # add this line
    }