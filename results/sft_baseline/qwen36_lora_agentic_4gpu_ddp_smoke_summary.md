# Qwen3.6-35B-A3B Agentic Coding LoRA SFT 4-GPU DDP Smoke Baseline

## Summary

- Status: passed
- Script: scripts/train_lora_sft.py
- Dataset: data/agentic_coding_sft_sample.jsonl
- Dataset records: 64
- Model: /workspace/models/Qwen3.6-35B-A3B
- GPUs: 4 x NVIDIA H800 80GB
- world_size: 4
- max_length: 1024
- per_device_batch_size: 1
- grad_accum_steps: 4
- global_batch_size_per_update: 16
- max_steps: 5
- LoRA rank: 4
- gradient_checkpointing: enabled
- loss label: avg_loss_across_ranks

## Results

- total_time_s: 35.23
- post-warmup step time range: 4.79-6.98 s
- best observed tokens_per_second: 262.52
- global_tokens_per_step: about 1248-1258
- theoretical max tokens/update: 16384
- token utilization: about 7.7%
- global_samples_per_step: 16
- peak rank0 memory: 65.00 GiB
- adapter saved: yes
- no OOM, Traceback, or NCCL error observed

## Conclusion

The formal Agentic Coding LoRA SFT 4-GPU DDP smoke baseline passed. However, token utilization is only about 7.7%, so the next optimization target is packing.
