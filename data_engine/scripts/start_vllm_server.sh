#!/bin/bash
# Script to start the vLLM server for inference

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

echo "Starting vLLM server..."
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --dtype auto \
    --api-key "EMPTY"

echo "vLLM server stopped."
