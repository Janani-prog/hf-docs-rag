from pydantic import BaseModel, Field
from typing import Optional


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    token_count: int
    chunk_index: int
    slug: str
    source_url: str
    title: str
    section: str
    scraped_at: str
    score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    number: int
    source_url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    valid: bool
    reason: Optional[str] = None


class RAGResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    citation_valid: bool
    insufficient_context: bool
    retrieved_chunks: int
    tokens_used: int
    model: str
    prompt_version: str
    latency: dict[str, float]


class PipelineMetrics(BaseModel):
    total_traces: int
    retrieval_latency_p50: float
    retrieval_latency_p95: float
    generation_latency_p50: float
    generation_latency_p95: float
    total_latency_p50: float
    total_latency_p95: float
    avg_tokens_per_query: float
    avg_cost_per_query_usd: float
    citation_coverage_pct: float
    insufficient_context_rate_pct: float
    error_rate_pct: float