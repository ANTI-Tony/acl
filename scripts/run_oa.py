"""
Step 3c: Output Alignment
Align outputs with refined instruction-input pairs.
Then merge with high-quality data to form the final optimized dataset.
"""
import json
import os
from openai import OpenAI
from tqdm import tqdm

# ---- Config ----
HIGH_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "iqd_high.json")
REFINED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "fir_refined.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "final_optimized.json")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "oa_align.txt")
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


def align_output(client, ins_new, inp_new, output, template):
    prompt = template.format(
        instruction_new=ins_new,
        input_new=inp_new,
        output=output,
    )
    result = call_expert(client, prompt)
    parsed = parse_json(result)
    return parsed.get("aligned_output", output)


def main():
    client = OpenAI(base_url=EXPERT_URL, api_key="not-needed")

    with open(HIGH_DATA_PATH, "r", encoding="utf-8") as f:
        high_data = json.load(f)
    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        refined_data = json.load(f)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    # Align outputs for refined data
    aligned = []
    for item in tqdm(refined_data, desc="Aligning outputs"):
        ins_new = item.get("instruction_refined", item["instruction"])
        inp_new = item.get("input_refined", item.get("input", ""))
        output = item["output"]

        # Only align if instruction was actually changed
        if ins_new != item["instruction"]:
            aligned_output = align_output(client, ins_new, inp_new, output, template)
        else:
            aligned_output = output

        aligned.append({
            "instruction": ins_new,
            "input": inp_new,
            "output": aligned_output,
        })

    # High-quality data: keep as-is
    high_formatted = []
    for item in high_data:
        high_formatted.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"],
        })

    # Merge
    final = high_formatted + aligned
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\nFinal dataset: {len(high_formatted)} high + {len(aligned)} refined = {len(final)} total")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
