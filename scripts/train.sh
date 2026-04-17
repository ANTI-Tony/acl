#!/bin/bash
# Fine-tune Qwen2.5-3B with LoRA
# Usage: bash scripts/train.sh <data_path> <output_dir>
#   e.g. bash scripts/train.sh data/raw/math_5k.json results/baseline
#   e.g. bash scripts/train.sh data/optimized/simple_optimized.json results/simple_opt
#   e.g. bash scripts/train.sh data/optimized/final_optimized.json results/full_opt

DATA_PATH=${1:-"data/raw/math_5k.json"}
OUTPUT_DIR=${2:-"results/baseline"}
MODEL="Qwen/Qwen2.5-3B"

echo "=========================================="
echo "Fine-tuning Qwen2.5-3B"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python scripts/train_lora.py \
    --model_name ${MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_length 2048
