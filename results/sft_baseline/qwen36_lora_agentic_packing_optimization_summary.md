# Qwen3.6-35B-A3B Agentic Coding LoRA SFT Packing Optimization

## Summary

- Status: passed
- Script: scripts/train_lora_sft.py
- Dataset: data/agentic_coding_sft_sample.jsonl
- Model: /workspace/models/Qwen3.6-35B-A3B
- GPUs: 4 x NVIDIA H800 80GB
- world_size: 4
- max_length: 1024
- per_device_batch_size: 1
- grad_accum_steps: 4
- global_batch_size_per_update: 16
- theoretical_max_tokens_per_update: 16384
- max_steps: 5
- LoRA rank: 4
- gradient_checkpointing: enabled

## Comparison

| config | global_tokens_per_step | token_utilization | best_tokens_per_second | peak_rank0_memory_gib | result |
| --- | ---: | ---: | ---: | ---: | --- |
| no_packing | about 1258 | about 7.7% | 262.52 | 65.00 | passed |
| packing | 16156 | 98.61% | 2098.86 | 68.10 | passed |

## Improvement

- Token utilization improved from about 7.7% to 98.61%.
- Effective token utilization improved by about 12.8x.
- Best observed tokens/sec improved from 262.52 to 2098.86.
- Effective token throughput improved by about 8.0x.
- Peak rank0 memory increased from about 65.00 GiB to 68.10 GiB.
- Peak memory increased by about 3.10 GiB.

## Notes

The no-packing baseline wastes most of the sequence capacity because the examples are short. Although max_length is 1024, the actual global tokens per update are only about 1258 out of 16384 possible tokens.

Packing combines short examples into nearly full 1024-token training sequences. This increases useful token utilization to 98.61%, significantly improving effective training throughput.

This packing implementation is intended as a systems benchmark. It concatenates samples into causal sequences with EOS separators, but it does not yet implement block-diagonal attention masks or per-sample position reset.

## Conclusion

Packing is the first high-impact training optimization in this project. On the Agentic Coding SFT smoke workload, packing improves token utilization by about 12.8x and effective token throughput by about 8.0x, at the cost of about 3.1 GiB additional peak memory per rank.
