# 4x H800 Remote Setup Checklist

This document describes the remote environment checks for the Qwen3.6-35B-A3B 4x H800 post-training skeleton under `qwen3_6_35b_a3b_4h800_posttrain/`.

It is a runbook, not an experiment report. Do not paste benchmark numbers or training results into this file.

## Safety Rules

- Run GPU, serving, and benchmark commands only on the remote 4x H800 node.
- Do not download model weights from this runbook. `MODEL_PATH` must point to an existing local model snapshot on the remote machine.
- Do not commit model weights, checkpoints, logs, benchmark jsonl, generated CSV files, or result files.
- Keep generated artifacts under ignored or external paths such as `logs/`, `outputs/`, `results/`, or a scratch directory outside the repo.
- Do not run full training from this checklist. It only covers environment checks and a small serving smoke test.

## 1. Clone the Repository

Run on the remote H800 machine:

```bash
git clone <REPO_URL> triton_teleai
cd triton_teleai
```

If the repository is already present, update it according to your normal workflow:

```bash
git status --short
git pull --ff-only
```

Do not use force reset unless you are sure there are no local changes to keep.

## 2. Create a Python Environment

Use the Python version required by the remote stack. This example keeps the environment local to the checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch with the CUDA build required by the machine image. Do this from the official PyTorch instructions or the cluster image documentation, not from this generic `requirements.txt`.

Install vLLM, SGLang, and LLaMA-Factory separately according to the remote environment policy. This repository does not pin those packages because serving and training framework compatibility is version-sensitive.

## 3. Check GPU Visibility

Run:

```bash
nvidia-smi
nvidia-smi -L
```

Confirm manually:

- Four H800 GPUs are visible.
- Driver and CUDA runtime are the expected versions for the environment.
- No unrelated process is occupying most of the memory.
- GPU health looks normal before launching serving.

Do not record command output in this repository if it includes machine-specific logs or operational details.

## 4. Check Python, PyTorch, and CUDA

Run:

```bash
python - <<'PY'
import sys

print("python:", sys.version.replace("\n", " "))

try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")

print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())

for idx in range(torch.cuda.device_count()):
    print(f"cuda_device_{idx}:", torch.cuda.get_device_name(idx))
PY
```

Confirm manually:

- `torch` imports successfully.
- CUDA is available.
- `torch.cuda.device_count()` reports 4.
- Device names match the expected H800 fleet.

This check should not allocate large tensors or run a workload.

## 5. Check Shell Script Syntax

From the repository root:

```bash
cd qwen3_6_35b_a3b_4h800_posttrain

for script in configs/serving/*.sh scripts/*.sh; do
  echo "checking ${script}"
  bash -n "${script}"
done
```

This only validates shell syntax. It does not launch serving or collect GPU stats.

## 6. Set `MODEL_PATH`

Set `MODEL_PATH` to a local model snapshot that already exists on the remote machine:

```bash
cd qwen3_6_35b_a3b_4h800_posttrain
export MODEL_PATH=/path/to/local/models/Qwen3.6-35B-A3B
test -d "${MODEL_PATH}"
```

The serving scripts fail fast if `MODEL_PATH` is empty. Do not set it to a HuggingFace Hub id for this smoke test.

Optional sanity checks:

```bash
test -f "${MODEL_PATH}/config.json"
test -f "${MODEL_PATH}/tokenizer_config.json" || test -f "${MODEL_PATH}/tokenizer.json"
```

## 7. Start vLLM TP=4 8K Serving Smoke Test

Run on the 4x H800 node:

```bash
cd qwen3_6_35b_a3b_4h800_posttrain
export MODEL_PATH=/path/to/local/models/Qwen3.6-35B-A3B
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PORT=8000
export SERVED_MODEL_NAME=qwen3_6_35b_a3b

bash configs/serving/vllm_tp4_8k.sh
```

Keep this process running in a terminal, `tmux`, or your scheduler session. Use a separate terminal for the curl and benchmark checks below.

If port `8000` is already in use, choose a different `PORT` and use the same value in the following commands.

## 8. Check `/v1/models`

From another terminal on the same remote node:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Confirm manually that the endpoint responds and exposes the served model name. Do not paste the response into tracked files.

## 9. Check `/v1/chat/completions`

Run one minimal request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3_6_35b_a3b",
    "messages": [
      {
        "role": "user",
        "content": "Write a short Python function that adds two integers."
      }
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Confirm manually that the response is valid JSON and contains a completion. Do not treat this as a quality evaluation.

## 10. Prepare 5 Prompt Smoke File Outside Tracked Results

Create a tiny prompt file under `logs/` or another scratch directory. Do not commit it if it contains private prompts.

```bash
mkdir -p logs
cat > logs/smoke_prompts_5.jsonl <<'EOF'
{"prompt":"Write a Python function that reverses a list."}
{"prompt":"Explain gradient accumulation in two sentences."}
{"prompt":"Write a unit test for an add(a, b) function."}
{"prompt":"What is sample packing in SFT?"}
{"prompt":"Give one reason to profile MoE router expert usage."}
EOF
```

## 11. Run `bench_serving.py` on 5 Prompts

Run:

```bash
python scripts/bench_serving.py \
  --input logs/smoke_prompts_5.jsonl \
  --output logs/bench_vllm_tp4_8k_smoke.jsonl \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen3_6_35b_a3b \
  --max-tokens 64 \
  --temperature 0 \
  --limit 5
```

Review the console summary and jsonl output locally on the remote machine. Do not commit `logs/bench_vllm_tp4_8k_smoke.jsonl`.

## 12. Collect GPU Stats During Smoke Test

In a separate terminal while serving or benchmarking is active:

```bash
cd qwen3_6_35b_a3b_4h800_posttrain
bash scripts/collect_gpu_stats.sh \
  --interval 2 \
  --duration 60 \
  --output logs/gpu_stats_vllm_tp4_8k_smoke.csv
```

Review the CSV on the remote machine. Do not commit `logs/gpu_stats_vllm_tp4_8k_smoke.csv`.

## 13. Stop the Serving Process

Stop the vLLM server from the terminal or session where it is running. After shutdown, confirm the endpoint is no longer active if needed:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

An unreachable endpoint after shutdown is expected.

## 14. Before Committing

Check that no generated artifacts are staged:

```bash
git status --short
```

Do not commit:

- Model weights or tokenizer snapshots.
- LoRA adapters or checkpoints.
- Training logs.
- Serving logs.
- Benchmark jsonl files.
- GPU stats CSV files.
- `outputs/`, `logs/`, `results/`, or similar generated directories.

Only commit source files, configs, docs, and intentionally curated templates.

