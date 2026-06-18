# Magicoder 1K Length-Aware Packing Benchmark

## Setup

- Model: `/workspace/models/Qwen3.6-35B-A3B`
- Dataset: `data/magicoder_oss_instruct_1k.jsonl`
- Workload: LoRA SFT on Magicoder 1K
- GPUs: 4 × NVIDIA H800 80GB
- Max length: 1024
- Per-device batch size: 1
- Gradient accumulation steps: 4
- Global batch size per optimizer update: 16
- Theoretical max tokens per update: 16,384
- LoRA rank: 4
- Gradient checkpointing: enabled
- Distributed training: 4-GPU DDP

## Results

| Mode | Avg step time | Avg tokens/sec | Token utilization | Avg tokens/update | Avg peak memory |
|---|---:|---:|---:|---:|---:|
| No packing | ~15.01 s | ~574 | ~52.6% | ~8,615 | ~67.1 GiB |
| Greedy packing | ~14.81 s | ~812 | ~70.7% | ~11,590 | ~67.8 GiB |
| Length-aware packing | 10.26 s | 1,658 | 96.1% | 15,739 | 68.1 GiB |

## Improvement

Length-aware packing improves token utilization by packing samples according to length using a best-fit decreasing strategy.

Compared with greedy packing:

- Token utilization: 70.7% -> 96.1%, about 1.36x.
- Effective token throughput: 812 -> 1,658 tokens/s, about 2.04x.
- Total 50-step runtime: ~740 s -> 513 s, about 30.7% lower.
- Peak memory remains around 68 GiB.

Compared with no-packing:

- Token utilization: 52.6% -> 96.1%, about 1.83x.
- Effective token throughput: 574 -> 1,658 tokens/s, about 2.89x.

## Notes

This length-aware packing mode is mainly intended for system benchmark and throughput analysis. For large-scale training quality, a shuffled bucketed packing strategy would be preferable to avoid overly deterministic length-sorted training order.
