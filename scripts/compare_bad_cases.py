"""
Compare bad cases between baseline and optimized models.
No LLM needed — does structural analysis on the saved bad case files.
"""
import json
import os
import re
import argparse
from collections import Counter


def categorize_error(item):
    """Heuristic-based error categorization without LLM."""
    problem = item.get("problem", "")
    model_output = item.get("model_output", "")
    ground_truth = item.get("ground_truth", "")
    predicted = item.get("predicted", "")

    # 1. Empty or very short output
    if len(model_output.strip()) < 20:
        return "Incomplete"

    # 2. Output cut off (no boxed answer, no conclusion)
    has_boxed = "\\boxed" in model_output
    has_answer_phrase = "answer is" in model_output.lower() or "therefore" in model_output.lower()
    if not has_boxed and not has_answer_phrase and len(model_output) > 500:
        return "Incomplete"

    # 3. Format error — has correct number somewhere but extracted wrong
    gt_nums = re.findall(r'-?\d+\.?\d*', str(ground_truth))
    if gt_nums:
        gt_main = gt_nums[0]
        if gt_main in model_output and gt_main != str(predicted):
            return "Format Error"

    # 4. Close numeric answer — likely computation error
    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
        if abs(pred_val - gt_val) / max(abs(gt_val), 1) < 0.5:
            return "Computation Error"
    except (ValueError, TypeError):
        pass

    # 5. Very different answer — likely misunderstanding or strategy error
    if len(model_output) < 200:
        return "Reasoning Gap"

    # 6. Long output but wrong — likely reasoning error
    steps = model_output.count('\n')
    if steps >= 5:
        return "Computation Error"
    else:
        return "Reasoning Gap"


def analyze(bad_cases_path, label):
    """Analyze bad cases and return stats."""
    with open(bad_cases_path, "r", encoding="utf-8") as f:
        bad_cases = json.load(f)

    total = len(bad_cases)

    # Categorize errors
    categories = Counter()
    by_type = Counter()
    category_by_type = {}

    for item in bad_cases:
        cat = categorize_error(item)
        categories[cat] += 1
        prob_type = item.get("type", "unknown")
        by_type[prob_type] += 1
        category_by_type.setdefault(prob_type, Counter())[cat] += 1

    print(f"\n{'='*60}")
    print(f"Bad Case Analysis: {label}")
    print(f"{'='*60}")
    print(f"Total errors: {total}")

    print(f"\n--- Error Category Distribution ---")
    print(f"{'Category':<22} {'Count':>6} {'Ratio':>8}")
    print("-" * 38)
    for cat, count in categories.most_common():
        ratio = count / total * 100
        print(f"{cat:<22} {count:>6} {ratio:>7.1f}%")

    print(f"\n--- Errors by Problem Type ---")
    print(f"{'Type':<28} {'Errors':>6}")
    print("-" * 36)
    for t, count in by_type.most_common():
        print(f"{t:<28} {count:>6}")

    return {
        "total_errors": total,
        "categories": dict(categories),
        "by_type": dict(by_type),
    }


def compare(baseline_path, optimized_path):
    """Compare baseline and optimized bad cases."""
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_cases = json.load(f)
    with open(optimized_path, "r", encoding="utf-8") as f:
        optimized_cases = json.load(f)

    # Build problem sets
    baseline_problems = {item["problem"] for item in baseline_cases}
    optimized_problems = {item["problem"] for item in optimized_cases}

    fixed = baseline_problems - optimized_problems  # was wrong, now correct
    broken = optimized_problems - baseline_problems  # was correct, now wrong
    still_wrong = baseline_problems & optimized_problems  # wrong in both

    print(f"\n{'='*60}")
    print(f"Comparison: Baseline vs Optimized")
    print(f"{'='*60}")
    print(f"Baseline errors:   {len(baseline_cases)}")
    print(f"Optimized errors:  {len(optimized_cases)}")
    print(f"Net improvement:   {len(baseline_cases) - len(optimized_cases)}")
    print(f"\nFixed by optimization:    {len(fixed)}")
    print(f"Broken by optimization:   {len(broken)}")
    print(f"Still wrong in both:      {len(still_wrong)}")

    # Analyze fixed problems by type
    fixed_items = [item for item in baseline_cases if item["problem"] in fixed]
    fixed_types = Counter(item.get("type", "unknown") for item in fixed_items)
    print(f"\n--- Fixed Problems by Type ---")
    for t, count in fixed_types.most_common():
        print(f"  {t}: {count}")

    # Analyze broken problems by type
    broken_items = [item for item in optimized_cases if item["problem"] in broken]
    broken_types = Counter(item.get("type", "unknown") for item in broken_items)
    print(f"\n--- Broken Problems by Type ---")
    for t, count in broken_types.most_common():
        print(f"  {t}: {count}")

    return {
        "fixed": len(fixed),
        "broken": len(broken),
        "still_wrong": len(still_wrong),
        "fixed_by_type": dict(fixed_types),
        "broken_by_type": dict(broken_types),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="results/eval_baseline/math500_bad_cases.json")
    parser.add_argument("--optimized", type=str, default="results/eval_simple_opt/math500_bad_cases.json")
    parser.add_argument("--output", type=str, default="results/bad_case_comparison.json")
    args = parser.parse_args()

    stats_baseline = analyze(args.baseline, "Baseline")
    stats_optimized = analyze(args.optimized, "Optimized")
    comparison = compare(args.baseline, args.optimized)

    # Save all results
    all_results = {
        "baseline": stats_baseline,
        "optimized": stats_optimized,
        "comparison": comparison,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
