"""
Custom math evaluation script.
Evaluates LoRA fine-tuned models on math benchmarks.
Supports: MATH500, GSM8K, and custom benchmark datasets.
"""
import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm


def extract_answer(text):
    """Extract the final answer from model output.
    Looks for \\boxed{...}, 'The answer is ...', or last number."""
    # Try \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()

    # Try "The answer is ..."
    answer_match = re.search(r'[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)', text)
    if answer_match:
        return answer_match.group(1).strip()

    # Try "#### ..." (GSM8K format)
    hash_match = re.search(r'####\s*(.+)', text)
    if hash_match:
        return hash_match.group(1).strip()

    # Fallback: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return text.strip().split('\n')[-1].strip()


def normalize_answer(answer):
    """Normalize answer for comparison."""
    answer = str(answer).strip()
    # Remove $, \text{}, spaces
    answer = answer.replace('$', '').replace('\\text{', '').replace('}', '')
    answer = answer.replace(' ', '').replace(',', '')
    # Try to convert to float for numeric comparison
    try:
        return str(float(answer))
    except ValueError:
        return answer.lower()


def check_answer(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    # Try numeric comparison with tolerance
    try:
        pred_val = float(pred_norm)
        gt_val = float(gt_norm)
        return abs(pred_val - gt_val) < 1e-6
    except ValueError:
        pass

    # Check if one contains the other
    if pred_norm in gt_norm or gt_norm in pred_norm:
        return True

    return False


def load_model(base_model, adapter_path=None):
    """Load base model with optional LoRA adapter."""
    print(f"Loading model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, problem, max_new_tokens=1024):
    """Generate a response for a math problem."""
    messages = [{"role": "user", "content": f"Solve the following math problem step by step.\n\n{problem}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def load_math500():
    """Load MATH test set (500 samples)."""
    print("Loading MATH-500...")
    subsets = ['algebra', 'counting_and_probability', 'geometry',
               'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    all_items = []
    for subset in subsets:
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
        for item in ds:
            all_items.append({
                "problem": item["problem"],
                "answer": extract_answer(item["solution"]),
                "type": subset,
            })

    # MATH-500: sample 500 from test set (deterministic)
    import random
    random.seed(42)
    random.shuffle(all_items)
    samples = all_items[:500]
    print(f"  Loaded {len(samples)} samples")
    return samples


def load_gsm8k():
    """Load GSM8K test set."""
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = []
    for item in ds:
        answer = item["answer"].split("####")[-1].strip()
        samples.append({
            "problem": item["question"],
            "answer": answer,
            "type": "gsm8k",
        })
    print(f"  Loaded {len(samples)} samples")
    return samples


BENCHMARK_LOADERS = {
    "math500": load_math500,
    "gsm8k": load_gsm8k,
}


def evaluate_benchmark(model, tokenizer, samples, benchmark_name, output_dir):
    """Evaluate model on a benchmark."""
    results = []
    correct = 0

    for item in tqdm(samples, desc=f"Eval {benchmark_name}"):
        response = generate_response(model, tokenizer, item["problem"])
        predicted = extract_answer(response)
        is_correct = check_answer(predicted, item["answer"])

        if is_correct:
            correct += 1

        results.append({
            "problem": item["problem"],
            "ground_truth": item["answer"],
            "model_output": response,
            "predicted": predicted,
            "correct": is_correct,
            "type": item.get("type", ""),
        })

    accuracy = correct / len(samples) * 100
    print(f"\n{benchmark_name}: {correct}/{len(samples)} = {accuracy:.1f}%")

    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{benchmark_name}_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save bad cases
    bad_cases = [r for r in results if not r["correct"]]
    with open(os.path.join(output_dir, f"{benchmark_name}_bad_cases.json"), "w") as f:
        json.dump(bad_cases, f, ensure_ascii=False, indent=2)

    # Per-type breakdown
    type_stats = {}
    for r in results:
        t = r.get("type", "all")
        if t not in type_stats:
            type_stats[t] = {"correct": 0, "total": 0}
        type_stats[t]["total"] += 1
        if r["correct"]:
            type_stats[t]["correct"] += 1

    print(f"\n  Per-type breakdown:")
    for t, s in sorted(type_stats.items()):
        acc = s["correct"] / s["total"] * 100
        print(f"    {t}: {s['correct']}/{s['total']} = {acc:.1f}%")

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "type_breakdown": {t: s["correct"]/s["total"]*100 for t, s in type_stats.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (omit for base model eval)")
    parser.add_argument("--benchmarks", type=str, default="math500,gsm8k",
                        help="Comma-separated list of benchmarks")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)
    benchmarks = args.benchmarks.split(",")

    all_results = {}
    for bench_name in benchmarks:
        bench_name = bench_name.strip()
        if bench_name not in BENCHMARK_LOADERS:
            print(f"Unknown benchmark: {bench_name}, skipping")
            continue

        samples = BENCHMARK_LOADERS[bench_name]()
        result = evaluate_benchmark(model, tokenizer, samples, bench_name, args.output_dir)
        all_results[bench_name] = result

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    label = args.adapter if args.adapter else args.base_model
    print(f"Model: {label}")
    for name, r in all_results.items():
        print(f"  {name}: {r['accuracy']:.1f}%")

    # Save summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
