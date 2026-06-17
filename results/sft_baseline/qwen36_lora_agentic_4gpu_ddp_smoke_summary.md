# Qwen3.6-35B-A3B Agentic Coding LoRA SFT 4-GPU DDP Smoke Baseline

## Summary

- Status: passed
- Script: scripts/train_lora_sft.py
- Dataset: data/agentic_coding_sft_sample.jsonl
- Dataset records: 64
- Python env: /workspace/sft_env
- Model: /workspace/models/Qwen3.6-35B-A3B
- GPUs: 4 x NVIDIA H800 80GB
- Torch: 2.11.0+cu130
- CUDA runtime: 13.0
- Transformers: 5.12.1
- PEFT: 0.19.1
- Launcher: torchrun
- Backend: NCCL
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
- warmup step time: 13.65 s
- post-warmup step time range: 4.79-6.98 s
- post-warmup tokens_per_second range: 180.29-262.52
- best observed tokens_per_second: 262.52
- global_tokens_per_step: about 1248-1258
- global_samples_per_step: 16
- peak rank0 memory: 65.00 GiB
- adapter saved: yes
- no OOM, Traceback, or NCCL error observed

## Losses

1.8954, 1.8550, 1.7732, 1.6526, 1.5263

## Notes

This run is the first formal Agentic Coding LoRA SFT smoke baseline. It validates JSONL data loading, chat-template formatting, DDP, gradient accumulation, avg-loss all-reduce, tokens/sec logging, samples/sec logging, memory logging, and adapter saving.

Although max_length is 1024, the actual samples are short. The measured global_tokens_per_step is only about 1250 tokens for a global batch of 16, or roughly 79 tokens per sample. Therefore this run is not yet a long-sequence training benchmark.

## Conclusion

The formal Agentic Coding LoRA SFT 4-GPU DDP smoke baseline passed. The next step should build a realistic long-context code SFT benchmark or add packing so that max_length and token utilization become meaningful.
