import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from src.retrieval.generator import query as rag_query
from src.retrieval.hybrid import search as hybrid_search
from src.retrieval.reranker import rerank

load_dotenv()

GOLDEN_DATASET_PATH = "golden_dataset/golden_qa.json"
RESULTS_DIR = "data/eval_results"
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
EVAL_MODEL = "llama-3.1-8b-instant"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def call_llm(prompt: str) -> str:
    """Single LLM call with retry on rate limit."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return "ERROR"
    return "ERROR"


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: are the claims in the answer supported by the retrieved context?
    We ask the LLM to count supported vs unsupported claims.
    Score = supported_claims / total_claims
    """
    if answer == "INSUFFICIENT_CONTEXT":
        return 1.0  # correct refusal is perfectly faithful

    context_text = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
    prompt = f"""Given the context below, evaluate whether the answer is faithful.

CONTEXT:
{context_text}

ANSWER:
{answer}

Count how many claims in the answer are supported by the context vs not supported.
Respond with ONLY a JSON object like: {{"supported": 3, "total": 4}}
No explanation, just the JSON."""

    result = call_llm(prompt)
    try:
        # Extract JSON even if there's surrounding text
        start = result.find("{")
        end = result.rfind("}") + 1
        data = json.loads(result[start:end])
        total = max(data.get("total", 1), 1)
        return round(data.get("supported", 0) / total, 4)
    except Exception:
        return 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer relevancy: does the answer actually address the question?
    Score = 1.0 (fully relevant), 0.5 (partially), 0.0 (not relevant)
    """
    if answer == "INSUFFICIENT_CONTEXT":
        return 1.0  # correct refusal is relevant behavior

    prompt = f"""Does the following answer address the question asked?

QUESTION: {question}

ANSWER: {answer}

Rate relevancy as:
- 1.0 if the answer directly and completely addresses the question
- 0.5 if the answer is partially relevant or incomplete
- 0.0 if the answer does not address the question at all

Respond with ONLY a number: 0.0, 0.5, or 1.0"""

    result = call_llm(prompt)
    try:
        return round(float(result.strip()), 4)
    except Exception:
        return 0.0


def score_context_precision(question: str, contexts: list[str]) -> float:
    """
    Context precision: what fraction of retrieved chunks are actually relevant?
    Score = relevant_chunks / total_chunks
    """
    if not contexts:
        return 0.0

    relevant = 0
    for ctx in contexts:
        prompt = f"""Is the following context chunk relevant to answering the question?

QUESTION: {question}

CONTEXT CHUNK: {ctx[:600]}

Respond with ONLY: yes or no"""
        result = call_llm(prompt).lower()
        if "yes" in result:
            relevant += 1
        time.sleep(0.5)  # avoid rate limiting per-chunk calls

    return round(relevant / len(contexts), 4)


def run_pipeline_on_question(question: str) -> tuple[str, list[str]]:
    import time
    for attempt in range(3):
        try:
            raw_chunks = hybrid_search(question, k=10)
            chunks = rerank(question, raw_chunks, top_k=5)
            contexts = [c["text"] for c in chunks]
            result = rag_query(question)
            return result["answer"], contexts
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def evaluate_all(golden: list[dict]) -> list[dict]:
    results = []
    print(f"\nEvaluating {len(golden)} questions...\n")

    for i, item in enumerate(golden):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1}/{len(golden)}] {q[:55]}...")

        try:
            answer, contexts = run_pipeline_on_question(q)
        except Exception as e:
            print(f"    Pipeline error: {e}")
            answer, contexts = "ERROR", []

        faith = score_faithfulness(answer, contexts)
        relevancy = score_answer_relevancy(q, answer)
        precision = score_context_precision(q, contexts)

        print(f"    faith={faith:.2f}  relevancy={relevancy:.2f}  precision={precision:.2f}")
        time.sleep(1)  # breathing room between questions

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "category": item.get("category", "factual"),
        })

    return results


def print_results_table(scores: dict, all_pass: bool):
    print("\n" + "=" * 55)
    print("  RAGAS-STYLE EVALUATION RESULTS")
    print("=" * 55)
    metrics = [
        ("Faithfulness",      "faithfulness",      FAITHFULNESS_THRESHOLD),
        ("Answer Relevancy",  "answer_relevancy",  RELEVANCY_THRESHOLD),
        ("Context Precision", "context_precision", None),
    ]
    for label, key, threshold in metrics:
        value = scores.get(key, 0)
        score = round(float(value), 4)
        if threshold:
            status = "✓ PASS" if score >= threshold else "✗ FAIL"
        else:
            status = ""
        print(f"  {label:<22} {score:.4f}  {status}")
    print("=" * 55)
    if all_pass:
        print("  ✓ All gated metrics passed — CI would succeed")
    else:
        print("  ✗ Metrics below threshold — CI would fail")
    print("=" * 55 + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    golden = load_golden_dataset()
    results = evaluate_all(golden)

    # Aggregate scores
    scores = {
        "faithfulness": round(
            sum(r["faithfulness"] for r in results) / len(results), 4),
        "answer_relevancy": round(
            sum(r["answer_relevancy"] for r in results) / len(results), 4),
        "context_precision": round(
            sum(r["context_precision"] for r in results) / len(results), 4),
    }

    all_pass = (
        scores["faithfulness"] >= FAITHFULNESS_THRESHOLD and
        scores["answer_relevancy"] >= RELEVANCY_THRESHOLD
    )

    print_results_table(scores, all_pass)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "prompt_version": "v2",
        "scores": scores,
        "per_question": results,
        "num_questions": len(golden),
        "passed": all_pass,
    }

    result_path = os.path.join(RESULTS_DIR, f"ragas_{timestamp}.json")
    baseline_path = os.path.join(RESULTS_DIR, "baseline.json")

    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)

    if not os.path.exists(baseline_path):
        with open(baseline_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Baseline saved to {baseline_path}")

    print(f"Results saved to {result_path}")
    return all_pass


if __name__ == "__main__":
    passed = run()
    sys.exit(0 if passed else 1)