"""
Step 3c: Output Alignment
Align outputs with refined instruction-input pairs.
Then merge with high-quality data to form the final optimized dataset.

Supports concurrent requests.
"""
import json
import os
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ---- Config ----
HIGH_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "iqd_high.json")
REFINED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "fir_refined.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "final_optimized.json")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "oa_align.txt")
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


async def align_item(client, item, template, semaphore, index):
    ins_new = item.get("instruction_refined", item["instruction"])
    inp_new = item.get("input_refined", item.get("input", ""))
    output = item["output"]

    # Only align if instruction was actually changed
    if ins_new != item["instruction"]:
        prompt = (template
            .replace("{instruction_new}", ins_new)
            .replace("{input_new}", inp_new)
            .replace("{output}", output))
        result = await call_expert(client, prompt, semaphore)
        parsed = parse_json(result)
        aligned_output = parsed.get("aligned_output", output)
    else:
        aligned_output = output

    return index, {
        "instruction": ins_new,
        "input": inp_new,
        "output": aligned_output,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=EXPERT_URL, api_key="not-needed")
    semaphore = asyncio.Semaphore(args.workers)

    with open(HIGH_DATA_PATH, "r", encoding="utf-8") as f:
        high_data = json.load(f)
    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        refined_data = json.load(f)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    print(f"Aligning {len(refined_data)} samples with {args.workers} concurrent workers...")

    tasks = [
        align_item(client, item, template, semaphore, i)
        for i, item in enumerate(refined_data)
    ]

    results = [None] * len(refined_data)
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Aligning"):
        index, result = await coro
        results[index] = result

    # High-quality data: keep as-is
    high_formatted = [
        {"instruction": item["instruction"], "input": item.get("input", ""), "output": item["output"]}
        for item in high_data
    ]

    # Merge
    final = high_formatted + results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\nFinal dataset: {len(high_formatted)} high + {len(results)} refined = {len(final)} total")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
