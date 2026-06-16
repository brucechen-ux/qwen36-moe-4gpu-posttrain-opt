#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_6_35b_a3b}"

export CUDA_VISIBLE_DEVICES

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[vLLM] ERROR: set MODEL_PATH to a local model snapshot path before launch" >&2
  exit 1
fi

echo "[vLLM] model=${MODEL_PATH} tp=4 max_model_len=32768 host=${HOST} port=${PORT}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code
