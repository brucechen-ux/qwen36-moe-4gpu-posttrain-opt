# Qwen3.6-35B-A3B 4x H800 Post-Training Skeleton

This project is a low-resource post-training workspace for `Qwen/Qwen3.6-35B-A3B` on a 4x H800 node. It is designed to keep the first experiments reproducible and inspectable before attempting larger training or RL pipelines.

The project intentionally avoids full-parameter training, 262K context training, complex asynchronous RL, and any automatic model download in local setup scripts.

## Goals

- Run QLoRA/LoRA SFT experiments on 4x H800.
- Measure the effect of sample packing at 2K, 4K, and 8K sequence length.
- Use activation checkpointing and gradient accumulation to fit low-resource training.
- Test an 8K Liger Kernel SFT variant.
- Compare serving behavior at 8K, 16K, and 32K context with vLLM and SGLang using TP=4.
- Run small coding reward RL experiments on HumanEval/MBPP style tasks.
- Profile MoE router and expert hit distribution during inference or training probes.
- Preserve logs, configs, and summaries so each experiment can be reproduced.

## Non-Goals

- No full-parameter training.
- No 262K context experiments.
- No large-scale async RL system.
- No automatic model download on the current machine.
- No fake benchmark or training results.

## Directory Layout

```text
.
├── configs
│   ├── serving
│   │   ├── sglang_tp4_8k.sh
│   │   ├── sglang_tp4_16k.sh
│   │   ├── sglang_tp4_32k.sh
│   │   ├── vllm_tp4_8k.sh
│   │   ├── vllm_tp4_16k.sh
│   │   └── vllm_tp4_32k.sh
│   └── sft
│       ├── deepspeed_zero2_4gpu.json
│       ├── llamafactory_qlora_sft_2k_packing.yaml
│       ├── llamafactory_qlora_sft_4k_packing.yaml
│       ├── llamafactory_qlora_sft_8k_liger.yaml
│       └── llamafactory_qlora_sft_8k_packing.yaml
├── reports
│   └── final_report_template.md
└── scripts
    ├── analyze_padding_ratio.py
    ├── analyze_rl_step_time.py
    ├── bench_serving.py
    ├── collect_gpu_stats.sh
    ├── hook_moe_router.py
    └── summarize_training_log.py
```

## What Requires a Real 4x H800 Node

Run these only on the target 4x H800 environment:

- vLLM and SGLang serving scripts under `configs/serving/`.
- QLoRA/LoRA SFT runs using configs under `configs/sft/`.
- GPU telemetry collection with `scripts/collect_gpu_stats.sh`.
- Any benchmark using a real model endpoint with `scripts/bench_serving.py`.
- MoE router profiling with `scripts/hook_moe_router.py` when loading the full model.
- HumanEval/MBPP reward RL experiments and RL log analysis after the pipeline has produced logs.

Safe local-only operations:

- Reading and editing configs.
- Running Python `--help` for the utility scripts.
- Running log analyzers against small sample logs or synthetic jsonl files.

## Model Path Policy

The serving scripts require `MODEL_PATH` and fail fast if it is not set. For reproducible H800 runs, set it to a local snapshot path before launching:

```bash
export MODEL_PATH=./models/Qwen3.6-35B-A3B
```

The SFT templates use `./models/Qwen3.6-35B-A3B` as a relative local placeholder. The Python MoE hook script uses `local_files_only=True` by default and will not download the model unless `--allow-download` is explicitly passed.

## Stage Plan

### Stage 0: Environment and Sanity

- Confirm CUDA, driver, NCCL, PyTorch, FlashAttention, vLLM, SGLang, and training framework versions.
- Confirm that `MODEL_PATH` points to a local Qwen3.6-35B-A3B snapshot.
- Launch one 8K TP=4 serving endpoint.
- Run a tiny prompt benchmark and record logs.

### Stage 1: Serving Baseline

- Launch vLLM TP=4 at 8K, 16K, and 32K.
- Launch SGLang TP=4 at 8K, 16K, and 32K.
- Use the same prompt jsonl and generation parameters for each run.
- Record TTFT, total latency, output tokens/s, success rate, GPU utilization, power, and peak memory.

### Stage 2: QLoRA SFT Baseline

- Start with 2K packed SFT.
- Scale to 4K and 8K packed SFT.
- Keep global batch size comparable through gradient accumulation.
- Use activation checkpointing in all baseline configs.
- Save training logs and summarize them with `scripts/summarize_training_log.py`.

### Stage 3: 8K Liger Variant

- Run the 8K Liger Kernel config after the baseline 8K config is stable.
- Compare step time, tokens/s, peak memory, loss curve, and any numerical instability.
- Check the LLaMA-Factory field name for enabling Liger Kernel before launch; it is marked with a TODO in the config.

### Stage 4: Length and Packing Analysis

- Run `scripts/analyze_padding_ratio.py` on tokenized lengths or raw jsonl data.
- Compare padding ratio with and without packing.
- Use the results to pick sequence lengths and packing settings for later SFT.

### Stage 5: Small Coding Reward RL

- Use small HumanEval/MBPP style prompts only.
- Keep rollout count, max tokens, and batch size small enough for 4x H800.
- Avoid complex async rollout/reward architecture in the first pass.
- Parse RL logs with `scripts/analyze_rl_step_time.py` to understand rollout, reward, ref logprob, and actor update costs.

### Stage 6: MoE Router Profiling

- Use `scripts/hook_moe_router.py` with a configurable module-name regex.
- Inspect the actual model structure first because Qwen MoE router module names may differ by implementation.
- Start with a single prompt or a tiny batch before instrumenting training.

## Experiment Metrics

### Serving

- Context length: 8K, 16K, 32K.
- Backend: vLLM or SGLang.
- Tensor parallel size: 4.
- TTFT.
- Total latency.
- Output tokens/s.
- Success rate.
- Peak GPU memory.
- Average GPU utilization.
- Average power draw.
- Error rate and error messages.

### SFT

- Sequence length.
- Packing enabled or disabled.
- QLoRA/LoRA settings.
- Gradient accumulation steps.
- Effective global batch size.
- Loss.
- Learning rate.
- Step time.
- Tokens/s.
- Peak GPU memory.
- Checkpoint size.
- Resume success.

### RL

- Dataset and prompt count.
- Rollout count per prompt.
- Reward model or reward function.
- Rollout time.
- Reward time.
- Reference logprob time.
- Actor update time.
- Total RL step time.
- Pass@1 or task-level reward, if evaluated.

### MoE Router

- Router module regex.
- Number of matched modules.
- Top-k routing setting used by the hook.
- Per-layer expert hit counts.
- Expert load entropy or imbalance ratio.
- Token count observed.

## Serving Usage

From the project root on the H800 node:

```bash
export MODEL_PATH=./models/Qwen3.6-35B-A3B
bash configs/serving/vllm_tp4_8k.sh
```

For SGLang:

```bash
export MODEL_PATH=./models/Qwen3.6-35B-A3B
bash configs/serving/sglang_tp4_8k.sh
```

Use a different port if multiple endpoints are active:

```bash
PORT=8001 bash configs/serving/vllm_tp4_16k.sh
```

## Serving Benchmark Usage

Prompt file format:

```json
{"prompt": "Write a Python function that reverses a linked list."}
{"prompt": "Explain why gradient accumulation is useful for low-resource training."}
```

Run against an OpenAI-compatible chat completions endpoint:

```bash
python scripts/bench_serving.py \
  --input data/prompts.jsonl \
  --output logs/bench_vllm_8k.jsonl \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen3_6_35b_a3b \
  --max-tokens 256
```

## GPU Stats Usage

Run this beside a serving or training job on the H800 node:

```bash
bash scripts/collect_gpu_stats.sh --interval 2 --output logs/gpu_stats.csv
```

For a bounded collection window:

```bash
bash scripts/collect_gpu_stats.sh --interval 2 --duration 600 --output logs/gpu_stats_10min.csv
```

## SFT Config Usage

The SFT templates use LLaMA-Factory style fields. Some fields are version-sensitive and marked with TODO comments. Before running, verify:

- Dataset names in `dataset`.
- Dataset registry entries expected by LLaMA-Factory.
- Model template name for this Qwen release.
- Deepspeed config path.
- Liger Kernel enable flag for the installed LLaMA-Factory version.
- Quantization backend availability.

Example launch pattern on the H800 node:

```bash
llamafactory-cli train configs/sft/llamafactory_qlora_sft_2k_packing.yaml
```

## Training Log Summary Usage

```bash
python scripts/summarize_training_log.py \
  --input logs/train_8k.log \
  --output reports/train_8k_summary.csv \
  --format csv
```

## Padding Ratio Usage

For tokenized length jsonl:

```bash
python scripts/analyze_padding_ratio.py \
  --input data/tokenized_lengths.jsonl \
  --length-field length \
  --seq-len 8192 \
  --batch-size 1 \
  --packing \
  --output reports/padding_8k.json
```

For raw jsonl using a rough character estimator:

```bash
python scripts/analyze_padding_ratio.py \
  --input data/prompts.jsonl \
  --text-field prompt \
  --seq-len 8192 \
  --estimate-tokens chars \
  --output reports/padding_raw_estimate.json
```

## RL Step-Time Analysis Usage

```bash
python scripts/analyze_rl_step_time.py \
  --input logs/rl_pipeline.log \
  --output reports/rl_step_time.json
```

## MoE Router Hook Usage

Inspect model module names first:

```bash
python scripts/hook_moe_router.py \
  --model-path ./models/Qwen3.6-35B-A3B \
  --router-regex "router|gate" \
  --dry-run
```

Run one prompt probe:

```bash
python scripts/hook_moe_router.py \
  --model-path ./models/Qwen3.6-35B-A3B \
  --router-regex "router|gate" \
  --prompt "Write a Python function for binary search." \
  --output reports/moe_router_probe.json
```

The `--router-regex` value must be adjusted after inspecting the actual HuggingFace module tree. The hook assumes router modules return logits shaped like `[tokens, num_experts]` or `[batch, tokens, num_experts]`; if the model implementation returns already-selected expert ids or a custom object, adapt `MoERouterProfiler._extract_router_tensor`.
