"""
Simple single-pass optimization (for Step 2 baseline comparison).
Uses Qwen72B to directly optimize each sample without the full IQD-FIR-OA pipeline.
This is what the advisor asked for as the initial baseline experiment.

Supports concurrent requests to speed up processing (~4x faster with 8 workers).
"""
import json
import os
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ---- Config ----
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "math_5k.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "simple_optimized.json")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "simple_optimize.txt")
EXPERT_URL = "http://localhost:8000/v1"
EXPERT_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"


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


async def call_expert(client, prompt, semaphore, max_retries=3):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=EXPERT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    print(f"API error after {max_retries} retries: {e}")
                    return None


async def process_item(client, item, template, semaphore, index):
    prompt = template.format(
        instruction=item["instruction"],
        output=item["output"],
    )
    result = await call_expert(client, prompt, semaphore)
    parsed = parse_json(result)
    return index, {
        "instruction": parsed.get("optimized_instruction", item["instruction"]),
        "input": item.get("input", ""),
        "output": parsed.get("optimized_output", item["output"]),
        "changes": parsed.get("changes", "none"),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    client = AsyncOpenAI(base_url=EXPERT_URL, api_key="not-needed")
    semaphore = asyncio.Semaphore(args.workers)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    print(f"Processing {len(data)} samples with {args.workers} concurrent workers...")

    # Create all tasks
    tasks = [
        process_item(client, item, template, semaphore, i)
        for i, item in enumerate(data)
    ]

    # Run with progress bar
    results = [None] * len(data)
    done_count = 0
    save_interval = 100

    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Optimizing"):
        index, result = await coro
        results[index] = result
        done_count += 1

        # Save checkpoint periodically
        if done_count % save_interval == 0:
            completed = [r for r in results if r is not None]
            with open(args.output + ".partial", "w", encoding="utf-8") as f:
                json.dump(completed, f, ensure_ascii=False, indent=2)

    # Final save (ordered by original index)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Cleanup partial file
    partial = args.output + ".partial"
    if os.path.exists(partial):
        os.remove(partial)

    changed = sum(1 for r in results if r and r["changes"] != "none")
    print(f"\nDone: {changed}/{len(results)} samples modified")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
