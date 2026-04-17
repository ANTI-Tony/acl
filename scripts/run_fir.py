"""
Step 3b: Feedback-driven Iterative Refinement
Iterative "evaluate → refine → review" loop for low-quality math data.
"""
import json
import os
from openai import OpenAI
from tqdm import tqdm

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


def generate_response(client, instruction, input_text=""):
    """Let the expert model generate a response for the given problem."""
    prompt = f"Solve the following math problem step by step.\n\nProblem: {instruction}"
    if input_text:
        prompt += f"\n{input_text}"
    return call_expert(client, prompt)


def evaluate(client, instruction, output, output_p1, template):
    """Evaluate and generate feedback via step-level verification."""
    prompt = template.format(
        instruction=instruction,
        output=output,
        output_p1=output_p1,
    )
    result = call_expert(client, prompt)
    return parse_json(result)


def refine(client, instruction, input_text, output, output_p1, feedback, feedback_history, template):
    """Refine instruction-input based on feedback."""
    prompt = template.format(
        instruction=instruction,
        input=input_text,
        output=output,
        output_p1=output_p1,
        feedback=feedback,
        feedback_history=feedback_history,
    )
    result = call_expert(client, prompt)
    return parse_json(result)


def review(client, ins_a, out_a, ins_b, out_b, template):
    """Review: compare original vs refined tuple."""
    prompt = template.format(
        instruction_a=ins_a,
        output_a=out_a,
        instruction_b=ins_b,
        output_b=out_b,
    )
    result = call_expert(client, prompt)
    return parse_json(result)


def refine_sample(client, sample, eval_tmpl, refine_tmpl, review_tmpl):
    """Run the iterative refinement loop for a single sample."""
    ins = sample["instruction"]
    inp = sample.get("input", "")
    out = sample["output"]
    feedback_history = []

    for t in range(MAX_ITERATIONS):
        # 1. Generate model response for current (ins, inp)
        out_p1 = generate_response(client, ins, inp)
        if not out_p1:
            break

        # 2. Evaluate: compare out vs out_p1, generate feedback
        eval_result = evaluate(client, ins, out, out_p1, eval_tmpl)
        feedback = eval_result.get("feedback", "")

        # 3. Refine instruction-input
        refine_result = refine(
            client, ins, inp, out, out_p1, feedback,
            json.dumps(feedback_history, ensure_ascii=False),
            refine_tmpl,
        )
        ins_new = refine_result.get("refined_instruction", ins)
        inp_new = refine_result.get("refined_input", inp)

        # 4. Generate response for refined (ins_new, inp_new)
        out_p2 = generate_response(client, ins_new, inp_new)
        if not out_p2:
            break

        # 5. Review: compare (ins, out_p1) vs (ins_new, out_p2)
        review_result = review(client, ins, out_p1, ins_new, out_p2, review_tmpl)
        winner = review_result.get("winner", "A")

        if winner == "B":
            # Refinement successful
            return {
                **sample,
                "instruction_refined": ins_new,
                "input_refined": inp_new,
                "iterations": t + 1,
                "refinement_status": "success",
            }
        else:
            # Refinement failed, accumulate feedback
            fail_feedback = review_result.get("feedback_if_A_wins", "")
            feedback_history.append({
                "iteration": t + 1,
                "feedback": feedback,
                "failure_reason": fail_feedback,
            })
            # Keep original ins/inp for next iteration

    # Max iterations reached or error, return original
    return {
        **sample,
        "instruction_refined": ins,
        "input_refined": inp,
        "iterations": MAX_ITERATIONS,
        "refinement_status": "max_iter" if feedback_history else "error",
    }


def main():
    client = OpenAI(base_url=EXPERT_URL, api_key="not-needed")

    with open(LOW_DATA_PATH, "r", encoding="utf-8") as f:
        low_data = json.load(f)

    eval_tmpl = load_prompt("fir_evaluate.txt")
    refine_tmpl = load_prompt("fir_refine.txt")
    review_tmpl = load_prompt("fir_review.txt")

    refined = []
    for sample in tqdm(low_data, desc="Refining"):
        result = refine_sample(client, sample, eval_tmpl, refine_tmpl, review_tmpl)
        refined.append(result)

        # Save periodically
        if len(refined) % 50 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(refined, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in refined if r["refinement_status"] == "success")
    print(f"\nDone: {success}/{len(refined)} successfully refined")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
