"""
Metrics dashboard — pulls trace data from LangFuse and computes:
- Latency: p50, p95 per pipeline stage
- Cost: total and per-query estimate
- Citation coverage: % of responses with valid citations
- Failure rate: % returning INSUFFICIENT_CONTEXT or errors
"""
import os
import json
import statistics
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def fetch_traces(limit: int = 100) -> list[dict]:
    """
    Fetches recent traces from LangFuse REST API.
    LangFuse exposes a /api/public/traces endpoint — no SDK needed.
    """
    import requests
    from requests.auth import HTTPBasicAuth

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("LangFuse keys not found in .env — cannot fetch traces.")
        return []

    url = f"{host}/api/public/traces"
    params = {"limit": limit, "orderBy": "timestamp.desc"}

    try:
        response = requests.get(
            url,
            params=params,
            auth=HTTPBasicAuth(public_key, secret_key),
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"Failed to fetch traces: {e}")
        return []


def fetch_observations(trace_id: str) -> list[dict]:
    """Fetches all spans/events for a single trace."""
    import requests
    from requests.auth import HTTPBasicAuth

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    url = f"{host}/api/public/observations"
    params = {"traceId": trace_id}

    try:
        response = requests.get(
            url,
            params=params,
            auth=HTTPBasicAuth(public_key, secret_key),
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception:
        return []


def percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return round(sorted_data[idx], 2)


def compute_metrics(traces: list[dict]) -> dict:
    """
    Extracts metrics from trace metadata we stored during pipeline runs.
    Falls back to trace-level timing if observation data is sparse.
    """
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    tokens_list = []
    citation_valid_count = 0
    insufficient_count = 0
    error_count = 0
    total = len(traces)

    for trace in traces:
        metadata = trace.get("metadata") or {}
        pipeline_stages = metadata.get("pipeline_stages", [])

        for stage in pipeline_stages:
            if stage.get("stage") == "retrieval":
                ms = stage.get("latency_ms", 0)
                if ms > 0:
                    retrieval_latencies.append(ms)

            elif stage.get("stage") == "generation":
                ms = stage.get("latency_ms", 0)
                tokens = stage.get("tokens_used", 0)
                if ms > 0:
                    generation_latencies.append(ms)
                if tokens > 0:
                    tokens_list.append(tokens)
                if stage.get("insufficient_context"):
                    insufficient_count += 1

            elif stage.get("stage") == "citations":
                if stage.get("valid"):
                    citation_valid_count += 1

        total_ms = metadata.get("total_ms", 0)
        if total_ms > 0:
            total_latencies.append(total_ms)

    avg_tokens = sum(tokens_list) / max(len(tokens_list), 1)
    # llama-3.3-70b on Groq: ~$0.59 per 1M tokens
    cost_per_query = avg_tokens * 0.00000059

    return {
        "total_traces": total,
        "retrieval_latency": {
            "p50": percentile(retrieval_latencies, 50),
            "p95": percentile(retrieval_latencies, 95),
            "samples": len(retrieval_latencies),
        },
        "generation_latency": {
            "p50": percentile(generation_latencies, 50),
            "p95": percentile(generation_latencies, 95),
            "samples": len(generation_latencies),
        },
        "total_latency": {
            "p50": percentile(total_latencies, 50),
            "p95": percentile(total_latencies, 95),
            "samples": len(total_latencies),
        },
        "tokens": {
            "avg_per_query": round(avg_tokens, 0),
            "total": sum(tokens_list),
        },
        "cost": {
            "avg_per_query_usd": round(cost_per_query, 6),
            "total_usd": round(cost_per_query * total, 4),
        },
        "citation_coverage": round(
            citation_valid_count / max(total, 1) * 100, 1
        ),
        "insufficient_context_rate": round(
            insufficient_count / max(total, 1) * 100, 1
        ),
        "error_rate": round(
            error_count / max(total, 1) * 100, 1
        ),
    }


def print_dashboard(metrics: dict):
    total = metrics["total_traces"]

    print("\n" + "=" * 58)
    print("  PIPELINE METRICS DASHBOARD")
    print("=" * 58)
    print(f"  Traces analyzed:     {total}")
    print()

    print("  LATENCY")
    print("  " + "-" * 44)
    rl = metrics["retrieval_latency"]
    gl = metrics["generation_latency"]
    tl = metrics["total_latency"]
    print(f"  {'Stage':<20} {'p50':>8}  {'p95':>8}")
    print(f"  {'Retrieval':<20} {rl['p50']:>7.0f}ms  {rl['p95']:>7.0f}ms")
    print(f"  {'Generation':<20} {gl['p50']:>7.0f}ms  {gl['p95']:>7.0f}ms")
    print(f"  {'Total':<20} {tl['p50']:>7.0f}ms  {tl['p95']:>7.0f}ms")
    print()

    print("  TOKENS & COST")
    print("  " + "-" * 44)
    t = metrics["tokens"]
    c = metrics["cost"]
    print(f"  Avg tokens/query:    {t['avg_per_query']:.0f}")
    print(f"  Total tokens used:   {t['total']:,}")
    print(f"  Avg cost/query:      ${c['avg_per_query_usd']:.6f}")
    print(f"  Total cost:          ${c['total_usd']:.4f}")
    print()

    print("  QUALITY")
    print("  " + "-" * 44)
    print(f"  Citation coverage:   {metrics['citation_coverage']}%")
    print(f"  Insufficient ctx:    {metrics['insufficient_context_rate']}%")
    print(f"  Error rate:          {metrics['error_rate']}%")
    print()
    print("=" * 58)

    # Save to file for CI artifacts
    os.makedirs("data/eval_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/eval_results/dashboard_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {out_path}\n")


def run(limit: int = 100):
    print(f"Fetching last {limit} traces from LangFuse...")
    traces = fetch_traces(limit=limit)

    if not traces:
        print("No traces found. Run some queries first with `python ask.py`")
        return

    print(f"Found {len(traces)} traces. Computing metrics...\n")
    metrics = compute_metrics(traces)
    print_dashboard(metrics)
    return metrics


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(limit=limit)