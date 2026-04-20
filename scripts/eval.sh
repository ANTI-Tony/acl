#!/bin/bash
# Evaluate LoRA fine-tuned model on math benchmarks
# Usage: bash scripts/eval.sh <adapter_path> <output_dir>
#   e.g. bash scripts/eval.sh results/baseline/final results/eval_baseline

ADAPTER_PATH=${1:-"results/baseline/final"}
OUTPUT_DIR=${2:-"results/eval_output"}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-3B"}

echo "=========================================="
echo "Evaluating LoRA adapter: ${ADAPTER_PATH}"
echo "Base model: ${BASE_MODEL}"
echo "=========================================="

# List of tasks to try - will skip if not found
# Run: lm_eval --tasks list 2>&1 | grep -i math  to find available tasks
TASKS=(
    "hendrycks_math"
    "minerva_math"
    "gsm8k"
)

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "[*] Evaluating ${TASK}..."
    lm_eval --model hf \
        --model_args "pretrained=${BASE_MODEL},peft=${ADAPTER_PATH},trust_remote_code=True,dtype=bfloat16" \
        --tasks ${TASK} \
        --batch_size 8 \
        --output_path ${OUTPUT_DIR}/${TASK} \
        --num_fewshot 0 \
        2>&1 | tee ${OUTPUT_DIR}/${TASK}.log

    if [ $? -ne 0 ]; then
        echo "  -> ${TASK} failed or not found, skipping"
    fi
done

echo ""
echo "=========================================="
echo "All evaluations complete. Results in ${OUTPUT_DIR}/"
echo "=========================================="
