"""
Compares the latest evaluation run against the saved baseline.
Fails with exit code 1 if any metric drops more than 5% from baseline.
This is what prevents a bad prompt change from getting merged.
"""
import json
import os
import sys
import glob

RESULTS_DIR = "data/eval_results"
REGRESSION_THRESHOLD = 0.05  # 5% drop triggers failure


def load_latest_result() -> dict:
    pattern = os.path.join(RESULTS_DIR, "ragas_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No eval results found.")
        sys.exit(1)
    with open(files[-1]) as f:
        return json.load(f)


def load_baseline() -> dict | None:
    path = os.path.join(RESULTS_DIR, "baseline.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare(baseline: dict, latest: dict) -> bool:
    print("\n" + "=" * 60)
    print("  REGRESSION CHECK vs BASELINE")
    print("=" * 60)
    print(f"  Baseline:  {baseline['timestamp']}")
    print(f"  Latest:    {latest['timestamp']}")
    print("-" * 60)

    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    all_pass = True

    for metric in metrics:
        base_score = baseline["scores"].get(metric, 0)
        new_score = latest["scores"].get(metric, 0)
        delta = new_score - base_score
        pct = (delta / max(base_score, 0.001)) * 100

        if delta < -REGRESSION_THRESHOLD:
            status = f"⚠ REGRESSION ({pct:+.1f}%)"
            all_pass = False
        elif delta > 0:
            status = f"↑ improved ({pct:+.1f}%)"
        else:
            status = f"→ stable ({pct:+.1f}%)"

        print(f"  {metric:<22} {base_score:.4f} → {new_score:.4f}  {status}")

    print("=" * 60)
    if all_pass:
        print("  ✓ No regressions detected")
    else:
        print("  ✗ Regression detected — would block merge in CI")
    print("=" * 60 + "\n")
    return all_pass


def run():
    latest = load_latest_result()
    baseline = load_baseline()

    if baseline is None:
        print("No baseline found — saving current result as baseline.")
        baseline_path = os.path.join(RESULTS_DIR, "baseline.json")
        with open(baseline_path, "w") as f:
            json.dump(latest, f, indent=2)
        return True

    return compare(baseline, latest)


if __name__ == "__main__":
    passed = run()
    sys.exit(0 if passed else 1)