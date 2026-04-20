#!/bin/bash
# Deploy expert model with vLLM
# Default: Qwen2.5-32B-Instruct-AWQ on single A100
# Override: EXPERT_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ TP=2 bash scripts/serve_expert.sh

MODEL=${EXPERT_MODEL:-"Qwen/Qwen2.5-32B-Instruct-AWQ"}
PORT=${PORT:-8000}
TP=${TP:-1}

echo "Starting vLLM server..."
echo "  Model: ${MODEL}"
echo "  Tensor Parallel: ${TP}"
echo "  Port: ${PORT}"

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tensor-parallel-size ${TP} \
    --port ${PORT} \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --dtype auto
