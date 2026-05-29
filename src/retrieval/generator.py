import os
import re
import time
from groq import Groq
from dotenv import load_dotenv
from src.retrieval.hybrid import search as hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.prompt_manager import get_prompt, get_current_version
from src.monitoring.langfuse_tracer import RAGTrace, flush

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHUNKS = 5


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] SOURCE: {chunk['source_url']}\n"
            f"TITLE: {chunk['title']}\n\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
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
    trace = RAGTrace(question=question, model=GROQ_MODEL)

    # Step 1: Hybrid retrieval
    t0 = time.time()
    raw_chunks = hybrid_search(question, k=10)
    chunks = rerank(question, raw_chunks, top_k=MAX_CONTEXT_CHUNKS)
    retrieval_ms = (time.time() - t0) * 1000
    trace.log_retrieval(question, chunks, retrieval_ms)

    # Step 2: Build context
    context = build_context(chunks)

    # Step 3: Load versioned prompt
    prompt_version = get_current_version()
    template = get_prompt(prompt_version)
    prompt = template.replace("{context}", context).replace("{question}", question)

    # Step 4: Call LLM
    t1 = time.time()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    generation_ms = (time.time() - t1) * 1000
    answer = response.choices[0].message.content.strip()
    usage = response.usage

    trace.log_generation(
        prompt, answer,
        tokens=usage.total_tokens,
        latency_ms=generation_ms,
        prompt_version=prompt_version,
        model=GROQ_MODEL,
    )

    # Step 5: Validate citations
    citations = extract_citations(answer, chunks)
    invalid = [c for c in citations if not c["valid"]]
    coverage = len([c for c in citations if c["valid"]]) / max(len(citations), 1)
    trace.log_citation_validation(len(invalid) == 0, len(citations), coverage)
    trace.finalize(answer, retrieval_ms + generation_ms)

    flush()

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "citation_valid": len(invalid) == 0,
        "insufficient_context": answer.strip() == "INSUFFICIENT_CONTEXT",
        "retrieved_chunks": len(chunks),
        "tokens_used": usage.total_tokens,
        "model": GROQ_MODEL,
        "prompt_version": prompt_version,
        "latency": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(retrieval_ms + generation_ms, 2),
        }
    }