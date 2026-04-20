"""
Step 3 (advisor's plan): Bad case analysis with error categorization.
Compare model predictions before/after training, categorize errors.
"""
import json
import os
import argparse
from collections import Counter
from openai import OpenAI
from tqdm import tqdm

EXPERT_URL = "http://localhost:8000/v1"
EXPERT_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"

CATEGORIZE_PROMPT = """You are a math error analysis expert. Given a math problem, the correct answer, and the model's wrong answer, categorize the error.

### Error Categories:
1. **Computation Error**: Correct reasoning direction but arithmetic/calculation mistakes
2. **Reasoning Gap**: Missing critical intermediate steps leading to wrong conclusion
3. **Misunderstanding**: Misinterpreted the problem statement or requirements
4. **Strategy Error**: Chose an inappropriate solving method
5. **Format Error**: Correct answer but wrong output format (cannot be extracted)
6. **Incomplete**: Solution is cut off or unfinished

### Input
**Problem**: {problem}
**Correct Answer**: {correct_answer}
**Model Output**: {model_output}

### Response Format
```json
{{
  "error_category": 1-6,
  "category_name": "name",
  "explanation": "brief explanation of what went wrong"
}}
```
"""


def call_expert(client, prompt):
    try:
        resp = client.chat.completions.create(
            model=EXPERT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return None


def parse_json(text):
    if text is None:
        return {}
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}


def analyze_bad_cases(client, bad_cases):
    """Categorize each bad case."""
    categorized = []
    for case in tqdm(bad_cases, desc="Categorizing errors"):
        prompt = (CATEGORIZE_PROMPT
            .replace("{problem}", case["problem"])
            .replace("{correct_answer}", case["correct_answer"])
            .replace("{model_output}", case["model_output"]))
        result = parse_json(call_expert(client, prompt))
        categorized.append({
            **case,
            "error_category": result.get("error_category", 0),
            "category_name": result.get("category_name", "unknown"),
            "explanation": result.get("explanation", ""),
        })
    return categorized


def print_report(categorized, label=""):
    """Print error distribution report."""
    print(f"\n{'='*60}")
    print(f"Bad Case Analysis Report {label}")
    print(f"{'='*60}")
    print(f"Total errors: {len(categorized)}")

    category_names = {
        1: "Computation Error",
        2: "Reasoning Gap",
        3: "Misunderstanding",
        4: "Strategy Error",
        5: "Format Error",
        6: "Incomplete",
    }

    counter = Counter(c["error_category"] for c in categorized)
    print(f"\n{'Category':<25} {'Count':>6} {'Ratio':>8}")
    print("-" * 42)
    for cat_id in sorted(counter.keys()):
        name = category_names.get(cat_id, "Unknown")
        count = counter[cat_id]
        ratio = count / len(categorized) * 100
        print(f"{name:<25} {count:>6} {ratio:>7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bad_cases", type=str, required=True,
                        help="Path to bad cases JSON (list of {problem, correct_answer, model_output})")
    parser.add_argument("--output", type=str, default="analysis/bad_case_report.json")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    client = OpenAI(base_url=EXPERT_URL, api_key="not-needed")

    with open(args.bad_cases, "r", encoding="utf-8") as f:
        bad_cases = json.load(f)

    categorized = analyze_bad_cases(client, bad_cases)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(categorized, f, ensure_ascii=False, indent=2)

    print_report(categorized, label=args.label)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
