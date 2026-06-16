# Final Report: Qwen3.6-35B-A3B 4x H800 Post-Training

## Run Metadata

| Field | Value |
|---|---|
| Date |  |
| Machine |  |
| GPU |  |
| Driver / CUDA |  |
| PyTorch |  |
| Transformers |  |
| vLLM |  |
| SGLang |  |
| Training framework |  |
| Model snapshot |  |
| Dataset snapshot |  |
| Commit / config version |  |

## Serving Results

| Backend | Context | TP | Prompt set | Requests | Success rate | Mean TTFT (s) | Mean latency (s) | Output tokens/s | Peak memory (GB) | Avg GPU util (%) | Avg power (W) | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM | 8K | 4 |  |  |  |  |  |  |  |  |  |  |
| vLLM | 16K | 4 |  |  |  |  |  |  |  |  |  |  |
| vLLM | 32K | 4 |  |  |  |  |  |  |  |  |  |  |
| SGLang | 8K | 4 |  |  |  |  |  |  |  |  |  |  |
| SGLang | 16K | 4 |  |  |  |  |  |  |  |  |  |  |
| SGLang | 32K | 4 |  |  |  |  |  |  |  |  |  |  |

## SFT Results

| Run | Seq len | Packing | Liger | LoRA rank | Grad accum | Effective batch | Steps | Final loss | Mean step time (s) | Tokens/s | Peak memory (GB) | Checkpoint path | Notes |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| QLoRA 2K packing | 2048 | yes | no | 64 |  |  |  |  |  |  |  |  |  |
| QLoRA 4K packing | 4096 | yes | no | 64 |  |  |  |  |  |  |  |  |  |
| QLoRA 8K packing | 8192 | yes | no | 64 |  |  |  |  |  |  |  |  |  |
| QLoRA 8K Liger | 8192 | yes | yes | 64 |  |  |  |  |  |  |  |  |  |

## Length Distribution and Padding

| Dataset | Records | Mean length | P50 | P90 | P99 | Max | Seq len | Packing | Padding ratio | Truncated records | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
|  |  |  |  |  |  |  | 2048 |  |  |  |  |
|  |  |  |  |  |  |  | 4096 |  |  |  |  |
|  |  |  |  |  |  |  | 8192 |  |  |  |  |

## RL Coding Reward Results

| Run | Dataset | Prompts | Rollouts/prompt | Max tokens | Reward type | Pass@1 / reward | Total step time (s) | Rollout % | Reward % | Ref logprob % | Actor update % | Notes |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
|  | HumanEval |  |  |  |  |  |  |  |  |  |  |  |
|  | MBPP |  |  |  |  |  |  |  |  |  |  |  |

## MoE Router and Expert Profile

| Probe | Router regex | Matched layers | Top-k | Tokens observed | Max expert load % | Min expert load % | Imbalance ratio | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |  |

## Observations

- Serving:
- SFT:
- Packing:
- RL:
- MoE routing:

## Reproducibility Checklist

| Item | Path / Value |
|---|---|
| Serving configs |  |
| SFT configs |  |
| Prompt jsonl |  |
| Training logs |  |
| GPU stats CSV |  |
| Benchmark jsonl |  |
| Padding summary |  |
| RL timing summary |  |
| Router profile JSON |  |

## Risks and Follow-Ups

- 
