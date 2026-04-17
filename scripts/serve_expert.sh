#!/bin/bash
# Deploy Qwen2.5-72B-Instruct with vLLM on 2x A100
# Uses AWQ 4-bit quantization to fit in ~80GB

MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
PORT=8000

echo "Starting vLLM server for ${MODEL} on port ${PORT}..."
echo "Using 2x A100 with tensor parallelism"

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tensor-parallel-size 2 \
    --port ${PORT} \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --trust-remote-code

# After server is up, test with:
# curl http://localhost:8000/v1/models
