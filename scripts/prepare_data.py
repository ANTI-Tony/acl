"""
Step 1: Download MATH-train and sample 5K examples.
"""
import json
import random
import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SAMPLE_SIZE = 5000
SEED = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    print("Downloading MATH dataset...")
    ds = load_dataset("hendrycks/competition_math", split="train")
    print(f"Total samples: {len(ds)}")

    # Stratified sampling by difficulty level
    by_level = {}
    for item in ds:
        level = item.get("level", "unknown")
        by_level.setdefault(level, []).append(item)

    print("Distribution by level:")
    for level, items in sorted(by_level.items()):
        print(f"  {level}: {len(items)}")

    # Sample proportionally from each level
    sampled = []
    total = len(ds)
    for level, items in by_level.items():
        n = max(1, int(SAMPLE_SIZE * len(items) / total))
        random.shuffle(items)
        sampled.extend(items[:n])

    # Trim or pad to exact SAMPLE_SIZE
    random.shuffle(sampled)
    sampled = sampled[:SAMPLE_SIZE]

    # Convert to unified format: instruction, input, output
    formatted = []
    for item in sampled:
        formatted.append({
            "instruction": item["problem"],
            "input": "",
            "output": item["solution"],
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        })

    output_path = os.path.join(OUTPUT_DIR, "math_5k.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(formatted)} samples to {output_path}")

    # Print type distribution
    type_dist = {}
    for item in formatted:
        t = item["type"]
        type_dist[t] = type_dist.get(t, 0) + 1
    print("\nDistribution by type:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
