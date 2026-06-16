#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_6_35b_a3b}"

export CUDA_VISIBLE_DEVICES

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[SGLang] ERROR: set MODEL_PATH to a local model snapshot path before launch" >&2
  exit 1
fi

echo "[SGLang] model=${MODEL_PATH} tp=4 context_length=32768 host=${HOST} port=${PORT}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tp-size 4 \
  --context-length 32768 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code
