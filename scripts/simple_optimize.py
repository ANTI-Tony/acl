"""
Simple single-pass optimization (for Step 2 baseline comparison).
Uses Qwen72B to directly optimize each sample without the full IQD-FIR-OA pipeline.
This is what the advisor asked for as the initial baseline experiment.
"""
import json
import os
from openai import OpenAI
from tqdm import tqdm

# ---- Config ----
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "math_5k.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "simple_optimized.json")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "simple_optimize.txt")
EXPERT_URL = "http://localhost:8000/v1"
EXPERT_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"


def call_expert(client, prompt):
    try:
        resp = client.chat.completions.create(
            model=EXPERT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
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


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    client = OpenAI(base_url=EXPERT_URL, api_key="not-needed")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    optimized = []
    for item in tqdm(data, desc="Optimizing"):
        prompt = template.format(
            instruction=item["instruction"],
            output=item["output"],
        )
        result = call_expert(client, prompt)
        parsed = parse_json(result)

        optimized.append({
            "instruction": parsed.get("optimized_instruction", item["instruction"]),
            "input": item.get("input", ""),
            "output": parsed.get("optimized_output", item["output"]),
            "changes": parsed.get("changes", "none"),
        })

        # Save periodically
        if len(optimized) % 100 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(optimized, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(optimized, f, ensure_ascii=False, indent=2)

    changed = sum(1 for o in optimized if o["changes"] != "none")
    print(f"\nDone: {changed}/{len(optimized)} samples modified")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
