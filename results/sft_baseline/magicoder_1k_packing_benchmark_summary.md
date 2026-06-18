# Magicoder 1K Packing Benchmark

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

## NaN Fix

During the initial packing benchmark, some packed sequences had all labels set to `-100`.
This caused `CrossEntropyLoss` to receive no valid supervised tokens and occasionally produce `NaN` loss.

The dataset construction was updated to skip examples where truncation or packing removes all supervised answer tokens.

Affected paths:

- `TinySFTDataset`
- `JsonlSFTDataset`
- `PackedJsonlSFTDataset`

After the fix, the packed dataset contains no all-ignore-label examples.

## Results

| Mode | Avg step time | Avg tokens/sec | Token utilization | Avg tokens/update | Avg peak memory |
|---|---:|---:|---:|---:|---:|
| No packing | 15.017 s | 573.82 | 52.58% | 8,615.32 | 67.10 GiB |
| Packing, stable window step 0-45 | 15.368 s | 750.36 | 70.39% | 11,532.22 | 67.77 GiB |
| Packing, all 50 steps | 14.805 s | 812.25 | 70.74% | 11,590.24 | 67.78 GiB |

## Improvement

Using the stable-window packing result:

- Token utilization improves from 52.58% to 70.39%, about 1.34×.
- Effective token throughput improves from 573.82 tokens/s to 750.36 tokens/s, about 1.31×.
- Average peak memory increases from 67.10 GiB to 67.77 GiB, about +0.67 GiB.

Using all 50 packing steps:

- Effective token throughput reaches 812.25 tokens/s, about 1.42× over no packing.

## Analysis

Packing improves throughput because the original Magicoder examples are shorter than the fixed 1024-token training window.
Without packing, each update processes only about 52.6% useful tokens on average.
Packing combines multiple shorter examples into longer sequences, increasing useful tokens per optimizer update.

The improvement is smaller than the toy Agentic Coding dataset because Magicoder samples are already medium-length.
Many examples are around 500-800 tokens, so two examples often cannot both fit into one 1024-token sequence.
Therefore, packing improves utilization substantially but does not reach near-100% utilization.

## Conclusion

Packing provides a clear training-system optimization on the real Magicoder 1K SFT workload.
It increases effective token throughput by about 1.31× under a conservative stable-window estimate, with less than 1 GiB additional peak memory.
