# Qwen3.6-35B-A3B LoRA SFT Gradient Checkpointing Comparison

## Summary

- Status: passed
- Model: /workspace/models/Qwen3.6-35B-A3B
- Python env: /workspace/sft_env
- GPUs: 4 x NVIDIA H800 80GB
- Launcher: torchrun
- Backend: NCCL
- Torch: 2.11.0+cu130
- Transformers: 5.12.1
- PEFT: 0.19.1
- max_length: 512
- max_steps: 20
- batch_size per rank: 1
- LoRA rank: 4
- loss label: avg_loss_across_ranks

## Comparison

| config | total_time_s | avg_step_time_s | peak_rank0_memory_gib | result |
| --- | ---: | ---: | ---: | --- |
| gradient_checkpointing_on | 26.06 | 1.303 | 64.81 | passed |
| gradient_checkpointing_off | 17.46 | 0.873 | 65.78 | passed |

## Observations

- Disabling gradient checkpointing did not OOM for this short max_length=512 LoRA run.
- Peak rank0 memory increased from about 64.81 GiB to 65.78 GiB.
- Peak memory increased by about 0.97 GiB.
- Total 20-step time decreased from 26.06s to 17.46s.
- The no-GC run was about 33% faster in this tiny benchmark.
- This is expected because gradient checkpointing saves activation memory by recomputing activations during backward, which adds compute overhead.

## No-GC Run Results

- total_time_s: 17.46
- average step time: 0.873 s
- first step time: 2.642 s
- peak rank0 memory: 65.78 GiB
- adapter saved: yes
- no OOM, Traceback, or NCCL error observed

## Avg Loss Across Ranks, No-GC

1.2802, 1.2296, 1.1032, 0.9612, 0.8296, 0.7319, 0.6203, 0.5183, 0.4398, 0.3772, 0.3124, 0.2364, 0.1713, 0.1141, 0.0665, 0.0385, 0.0176, 0.0087, 0.0049, 0.0031

## Conclusion

For this 4-GPU DDP LoRA SFT max_length=512 baseline, disabling gradient checkpointing improves throughput at the cost of roughly 1 GiB extra peak memory per rank. Since H800 80GB still has enough headroom in this configuration, no-GC is faster for this tiny baseline. Longer sequence lengths should be benchmarked separately because activation memory grows with sequence length.
