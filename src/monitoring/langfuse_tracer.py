import os
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _client


class RAGTrace:
    """
    LangFuse 4.x uses start_as_current_observation() context managers.
    We store span data and log it all as a single trace at the end.
    """

    def __init__(self, question: str, model: str):
        self.question = question
        self.model = model
        self.events = []
        self.enabled = False
        try:
            get_client()
            self.enabled = True
        except Exception as e:
            print(f"LangFuse unavailable (non-fatal): {e}")

    def log_retrieval(self, query: str, chunks: list[dict], latency_ms: float):
        self.events.append({
            "stage": "retrieval",
            "num_chunks": len(chunks),
            "top_sources": [c["source_url"] for c in chunks[:3]],
            "latency_ms": round(latency_ms, 2),
        })

    def log_generation(self, prompt: str, answer: str, tokens: int,
                       latency_ms: float, prompt_version: str, model: str):
        self.events.append({
            "stage": "generation",
            "prompt_version": prompt_version,
            "tokens_used": tokens,
            "latency_ms": round(latency_ms, 2),
            "insufficient_context": answer.strip() == "INSUFFICIENT_CONTEXT",
            "estimated_cost_usd": round(tokens * 0.00000059, 6),
        })

    def log_citation_validation(self, citation_valid: bool,
                                num_citations: int, coverage: float):
        self.events.append({
            "stage": "citations",
            "valid": citation_valid,
            "count": num_citations,
            "coverage": round(coverage, 4),
        })

    def finalize(self, answer: str, total_ms: float):
        if not self.enabled:
            return
        try:
            client = get_client()
            # LangFuse 4.x: use start_as_current_observation as context manager
            with client.start_as_current_observation(
                name="rag_query",
                input=self.question,
                output=answer[:500],
                metadata={
                    "model": self.model,
                    "total_ms": round(total_ms, 2),
                    "pipeline_stages": self.events,
                }
            ):
                pass  # all data passed via metadata above
            print(f"LangFuse trace logged ✓ (total: {round(total_ms)}ms)")
        except Exception as e:
            print(f"LangFuse finalize failed (non-fatal): {e}")


def flush():
    try:
        if _client:
            _client.flush()
    except Exception:
        pass