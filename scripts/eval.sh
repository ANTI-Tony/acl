#!/bin/bash
# Evaluate fine-tuned model on 6 math benchmarks
# Usage: bash scripts/eval.sh <model_path>
#   e.g. bash scripts/eval.sh results/baseline/final
#   e.g. bash scripts/eval.sh results/simple_opt/final

MODEL_PATH=${1:-"Qwen/Qwen2.5-3B"}
OUTPUT_DIR=${2:-"results/eval_output"}

echo "=========================================="
echo "Evaluating: ${MODEL_PATH}"
echo "=========================================="

# Using lm-evaluation-harness for standardized evaluation
# Install: pip install lm-eval

BENCHMARKS=(
    "minerva_math"
    "amc23"
)

# MATH500
echo "[1/6] Evaluating MATH500..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16 \
    --tasks math_500 \
    --batch_size 8 \
    --output_path ${OUTPUT_DIR}/math500 \
    --num_fewshot 0

# Minerva Math
echo "[2/6] Evaluating Minerva Math..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16 \
    --tasks minerva_math \
    --batch_size 8 \
    --output_path ${OUTPUT_DIR}/minerva_math \
    --num_fewshot 0

# OlympiadBench
echo "[3/6] Evaluating OlympiadBench..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16 \
    --tasks olympiad_bench \
    --batch_size 8 \
    --output_path ${OUTPUT_DIR}/olympiad \
    --num_fewshot 0

# College Math
echo "[4/6] Evaluating College Math..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16 \
    --tasks college_math \
    --batch_size 8 \
    --output_path ${OUTPUT_DIR}/college_math \
    --num_fewshot 0

# AMC23
echo "[5/6] Evaluating AMC23..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16 \
    --tasks amc23 \
    --batch_size 8 \
    --output_path ${OUTPUT_DIR}/amc23 \
    --num_fewshot 0

# AIME25
echo "[6/6] Evaluating AIME25..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16 \
    --tasks aime25 \
    --batch_size 8 \
    --output_path ${OUTPUT_DIR}/aime25 \
    --num_fewshot 0

echo "=========================================="
echo "All evaluations complete. Results in ${OUTPUT_DIR}/"
echo "=========================================="
