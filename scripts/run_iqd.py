"""
Step 3a: Instruction Quality Differentiation
- Expert-guided quality labeling (math 5-dim)
- K-means clustering for diversity
- IFD scoring for difficulty

Supports concurrent requests for Stage 1 labeling.
"""
import json
import os
import asyncio
import argparse
import numpy as np
from openai import AsyncOpenAI, OpenAI
from sklearn.cluster import KMeans
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# ---- Config ----
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "math_5k.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "optimized")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "iqd_quality_label.txt")
EXPERT_URL = os.environ.get("EXPERT_URL", "http://localhost:8000/v1")
EXPERT_MODEL = os.environ.get("EXPERT_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
N_CLUSTERS = 10
TOP_K_RATIO = 0.5


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
                    max_tokens=512,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    print(f"API error after {max_retries} retries: {e}")
                    return None


async def label_item(client, item, template, semaphore, index):
    prompt = template.replace("{instruction}", item["instruction"]).replace("{output}", item["output"])
    result = await call_expert(client, prompt, semaphore)
    parsed = parse_json(result)
    return index, {
        "quality_label": parsed.get("quality_label", "low"),
        "quality_detail": parsed.get("dimensions", {}),
        "contains_reasoning": parsed.get("contains_reasoning", True),
    }


async def stage1_quality_labeling(data, template, workers=8):
    """Expert-guided quality labeling with math-specific dimensions (concurrent)."""
    print(f"\n=== Stage 1: Quality Labeling ({workers} workers) ===")
    client = AsyncOpenAI(base_url=EXPERT_URL, api_key="not-needed")
    semaphore = asyncio.Semaphore(workers)

    tasks = [
        label_item(client, item, template, semaphore, i)
        for i, item in enumerate(data)
    ]

    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Labeling"):
        index, result = await coro
        data[index].update(result)

    # Filter: keep only samples with reasoning
    data = [d for d in data if d.get("contains_reasoning", True)]
    high = [d for d in data if d["quality_label"] == "high"]
    low = [d for d in data if d["quality_label"] == "low"]
    print(f"After labeling: {len(high)} high, {len(low)} low, {len(data)} total")
    return data


def stage2_cluster_and_rank(data):
    """Cluster by semantic diversity, rank by IFD within each cluster."""
    print("\n=== Stage 2: Cluster & Difficulty-aware Scoring ===")

    # Get embeddings via sync client
    client = OpenAI(base_url=EXPERT_URL, api_key="not-needed")
    texts = [d["instruction"] + " " + d["output"] for d in data]

    try:
        all_embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=EXPERT_MODEL, input=batch)
            for item in resp.data:
                all_embeddings.append(item.embedding)
        embeddings = np.array(all_embeddings)
    except Exception as e:
        print(f"Embedding API not available ({e}), using random assignment")
        np.random.seed(42)
        for d in data:
            d["cluster"] = np.random.randint(0, N_CLUSTERS)
            d["ifd_score"] = np.random.random()
        high = [d for d in data if d["quality_label"] == "high"]
        low = [d for d in data if d["quality_label"] == "low"]
        return high, low

    # K-means clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    for i, d in enumerate(data):
        d["cluster"] = int(labels[i])

    # IFD scoring (simplified: use solution length ratio as proxy)
    # TODO: compute real IFD with backbone model
    for d in data:
        sol_len = len(d["output"].split())
        ins_len = len(d["instruction"].split())
        d["ifd_score"] = sol_len / max(ins_len, 1)

    # Rank within each cluster
    clusters = {}
    for d in data:
        clusters.setdefault(d["cluster"], []).append(d)

    final_high, final_low = [], []
    for cid, items in clusters.items():
        items.sort(key=lambda x: (0 if x["quality_label"] == "high" else 1, -x["ifd_score"]))
        k = max(1, int(len(items) * TOP_K_RATIO))
        final_high.extend(items[:k])
        final_low.extend(items[k:])

    print(f"After clustering: {len(final_high)} high, {len(final_low)} low")
    return final_high, final_low


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data", type=str, default=DATA_PATH)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    # Stage 1
    data = await stage1_quality_labeling(data, template, workers=args.workers)

    # Stage 2
    high, low = stage2_cluster_and_rank(data)

    # Save
    with open(os.path.join(OUTPUT_DIR, "iqd_high.json"), "w", encoding="utf-8") as f:
        json.dump(high, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "iqd_low.json"), "w", encoding="utf-8") as f:
        json.dump(low, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {len(high)} high-quality, {len(low)} low-quality")


if __name__ == "__main__":
    asyncio.run(main())
