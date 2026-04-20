"""
Step 3b: Feedback-driven Iterative Refinement
Iterative "evaluate → refine → review" loop for low-quality math data.

Supports concurrent processing of multiple samples (each sample's iterations are sequential,
but different samples run in parallel).
"""
import json
import os
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ---- Config ----
LOW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "iqd_low.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "fir_refined.json")
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
EXPERT_URL = "http://localhost:8000/v1"
EXPERT_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
MAX_ITERATIONS = 3


def load_prompt(name):
    with open(os.path.join(PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


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


async def generate_response(client, instruction, input_text, semaphore):
    prompt = f"Solve the following math problem step by step.\n\nProblem: {instruction}"
    if input_text:
        prompt += f"\n{input_text}"
    return await call_expert(client, prompt, semaphore)


async def refine_sample(client, sample, eval_tmpl, refine_tmpl, review_tmpl, semaphore, index):
    """Run the iterative refinement loop for a single sample."""
    ins = sample["instruction"]
    inp = sample.get("input", "")
    out = sample["output"]
    feedback_history = []

    for t in range(MAX_ITERATIONS):
        # 1. Generate model response
        out_p1 = await generate_response(client, ins, inp, semaphore)
        if not out_p1:
            break

        # 2. Evaluate
        eval_prompt = (eval_tmpl
            .replace("{instruction}", ins)
            .replace("{output}", out)
            .replace("{output_p1}", out_p1))
        eval_result = parse_json(await call_expert(client, eval_prompt, semaphore))
        feedback = eval_result.get("feedback", "")

        # 3. Refine
        refine_prompt = (refine_tmpl
            .replace("{instruction}", ins)
            .replace("{input}", inp)
            .replace("{output}", out)
            .replace("{output_p1}", out_p1)
            .replace("{feedback}", feedback)
            .replace("{feedback_history}", json.dumps(feedback_history, ensure_ascii=False)))
        refine_result = parse_json(await call_expert(client, refine_prompt, semaphore))
        ins_new = refine_result.get("refined_instruction", ins)
        inp_new = refine_result.get("refined_input", inp)

        # 4. Generate response for refined version
        out_p2 = await generate_response(client, ins_new, inp_new, semaphore)
        if not out_p2:
            break

        # 5. Review
        review_prompt = (review_tmpl
            .replace("{instruction_a}", ins)
            .replace("{output_a}", out_p1)
            .replace("{instruction_b}", ins_new)
            .replace("{output_b}", out_p2))
        review_result = parse_json(await call_expert(client, review_prompt, semaphore))
        winner = review_result.get("winner", "A")

        if winner == "B":
            return index, {
                **sample,
                "instruction_refined": ins_new,
                "input_refined": inp_new,
                "iterations": t + 1,
                "refinement_status": "success",
            }
        else:
            feedback_history.append({
                "iteration": t + 1,
                "feedback": feedback,
                "failure_reason": review_result.get("feedback_if_A_wins", ""),
            })

    return index, {
        **sample,
        "instruction_refined": ins,
        "input_refined": inp,
        "iterations": MAX_ITERATIONS,
        "refinement_status": "max_iter" if feedback_history else "error",
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent samples (each sample makes ~5 sequential calls per iter, "
                             "so 4 workers ≈ 20 concurrent requests at peak)")
    parser.add_argument("--data", type=str, default=LOW_DATA_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=EXPERT_URL, api_key="not-needed")
    semaphore = asyncio.Semaphore(args.workers * 2)  # limit total concurrent API calls

    with open(args.data, "r", encoding="utf-8") as f:
        low_data = json.load(f)

    eval_tmpl = load_prompt("fir_evaluate.txt")
    refine_tmpl = load_prompt("fir_refine.txt")
    review_tmpl = load_prompt("fir_review.txt")

    print(f"Refining {len(low_data)} samples with {args.workers} concurrent workers...")
    print(f"Max iterations per sample: {MAX_ITERATIONS}")

    tasks = [
        refine_sample(client, sample, eval_tmpl, refine_tmpl, review_tmpl, semaphore, i)
        for i, sample in enumerate(low_data)
    ]

    results = [None] * len(low_data)
    done_count = 0

    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Refining"):
        index, result = await coro
        results[index] = result
        done_count += 1

        if done_count % 50 == 0:
            completed = [r for r in results if r is not None]
            with open(args.output + ".partial", "w", encoding="utf-8") as f:
                json.dump(completed, f, ensure_ascii=False, indent=2)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    partial = args.output + ".partial"
    if os.path.exists(partial):
        os.remove(partial)

    success = sum(1 for r in results if r and r["refinement_status"] == "success")
    print(f"\nDone: {success}/{len(results)} successfully refined")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
