# Qwen3.6-35B-A3B Agentic Coding LoRA SFT Packing 50-Step Baseline

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
- max_steps: 50
- LoRA rank: 4
- gradient_checkpointing: enabled
- packing: enabled

## Main Results

- total_time_s: 358.31
- total_effective_tokens: 807800
- overall_effective_tokens_per_second: 2254.50
- avg_step_time_all_steps: 7.165 s
- avg_step_time_excluding_step0: 7.074 s
- avg_tokens_per_second_excluding_step0: 2283.89
- step_time_range_excluding_step0: 6.924-7.169 s
- tokens_per_second_range_excluding_step0: 2253.45-2333.31
- global_tokens_per_step: 16156
- token_utilization: 98.61%
- peak_rank0_memory: 68.10 GiB
- adapter saved: yes
- no OOM, Traceback, or NCCL error observed

## Loss

- first_loss: 0.8846
- final_loss: 0.00021

The loss decreases quickly because this is a small 64-record smoke dataset repeatedly cycled for systems benchmarking. The loss curve only indicates training stability, not real coding generalization.

## Analysis

This run confirms that packing remains stable beyond a short smoke test. Across 50 optimizer updates, token utilization stays at 98.61%, global_tokens_per_step stays at 16156 out of 16384 theoretical tokens, and peak rank0 memory stays around 68.10 GiB.

Compared with the earlier no-packing baseline, packing removes most padding waste and turns short SFT samples into nearly full 1024-token training sequences. This significantly improves effective token throughput.

## Conclusion

Milestone 3A is complete: packing is a high-impact SFT training optimization for this Agentic Coding workload. It improves token utilization from about 7.7% to 98.61% and provides a stable 4-GPU LoRA SFT throughput baseline of about 2.28k effective tokens/s after warmup.
