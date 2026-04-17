"""
Step 3a: Instruction Quality Differentiation
- Expert-guided quality labeling (math 5-dim)
- K-means clustering for diversity
- IFD scoring for difficulty
"""
import json
import os
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm

# ---- Config ----
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "math_5k.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "optimized")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "iqd_quality_label.txt")
EXPERT_URL = "http://localhost:8000/v1"
EXPERT_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
N_CLUSTERS = 10
TOP_K_RATIO = 0.5  # top 50% as high quality per cluster


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_expert(client, prompt, max_retries=3):
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=EXPERT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"API error: {e}, retrying...")
    return None


def get_embeddings(client, texts, batch_size=32):
    """Get embeddings from the expert model."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=EXPERT_MODEL,
            input=batch,
        )
        for item in resp.data:
            all_embeddings.append(item.embedding)
    return np.array(all_embeddings)


def stage1_quality_labeling(data, client, prompt_template):
    """Expert-guided quality labeling with math-specific dimensions."""
    print("\n=== Stage 1: Quality Labeling ===")
    for item in tqdm(data, desc="Labeling"):
        prompt = prompt_template.format(
            instruction=item["instruction"],
            output=item["output"],
        )
        result = call_expert(client, prompt)
        try:
            parsed = json.loads(result.strip().strip("```json").strip("```"))
            item["quality_label"] = parsed.get("quality_label", "low")
            item["quality_detail"] = parsed.get("dimensions", {})
            item["contains_reasoning"] = parsed.get("contains_reasoning", False)
        except (json.JSONDecodeError, AttributeError):
            item["quality_label"] = "low"
            item["quality_detail"] = {}
            item["contains_reasoning"] = True

    # Filter: keep only samples with reasoning
    data = [d for d in data if d.get("contains_reasoning", True)]
    high = [d for d in data if d["quality_label"] == "high"]
    low = [d for d in data if d["quality_label"] == "low"]
    print(f"After labeling: {len(high)} high, {len(low)} low, {len(data)} total")
    return data, high, low


def stage2_cluster_and_rank(data, client):
    """Cluster by semantic diversity, rank by IFD within each cluster."""
    print("\n=== Stage 2: Cluster & Difficulty-aware Scoring ===")

    # Get embeddings
    texts = [d["instruction"] + " " + d["output"] for d in data]
    try:
        embeddings = get_embeddings(client, texts)
    except Exception as e:
        print(f"Embedding API not available ({e}), using random assignment")
        np.random.seed(42)
        for d in data:
            d["cluster"] = np.random.randint(0, N_CLUSTERS)
            d["ifd_score"] = np.random.random()
        return data

    # K-means clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    for i, d in enumerate(data):
        d["cluster"] = int(labels[i])

    # IFD scoring (simplified: use solution length as proxy, replace with real IFD later)
    # TODO: compute real IFD with backbone model
    for d in data:
        sol_len = len(d["output"].split())
        ins_len = len(d["instruction"].split())
        d["ifd_score"] = sol_len / max(ins_len, 1)

    # Rank within each cluster: high-quality first, then by IFD (higher = harder = more valuable)
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = OpenAI(base_url=EXPERT_URL, api_key="not-needed")

    data = load_data(DATA_PATH)
    prompt_template = load_prompt(PROMPT_PATH)

    # Stage 1: Quality labeling
    data, _, _ = stage1_quality_labeling(data, client, prompt_template)

    # Stage 2: Cluster and rank
    high, low = stage2_cluster_and_rank(data, client)

    # Save results
    with open(os.path.join(OUTPUT_DIR, "iqd_high.json"), "w", encoding="utf-8") as f:
        json.dump(high, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "iqd_low.json"), "w", encoding="utf-8") as f:
        json.dump(low, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {len(high)} high-quality, {len(low)} low-quality")


if __name__ == "__main__":
    main()
