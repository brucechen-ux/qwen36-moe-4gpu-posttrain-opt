# Qwen3.6-35B-A3B LoRA SFT 4-GPU DDP 20-Step Baseline

## Summary

- Status: passed
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
- max_length: 512
- max_steps: 20
- batch_size per rank: 1
- LoRA rank: 4
- LoRA alpha: 8
- loss label: avg_loss_across_ranks

## Results

- total_time_s: 26.06
- average step time: 1.303 s
- average step time after warmup: about 1.15 s
- first step time: 3.751 s
- peak rank0 training memory: 64.81 GiB
- base model memory per rank: 64.56 GiB
- adapter saved: yes
- no OOM, Traceback, or NCCL error observed

## Avg Loss Across Ranks

1.2802, 1.2311, 1.1247, 0.9659, 0.8509, 0.7371, 0.6188, 0.5239, 0.4581, 0.4034, 0.3356, 0.2574, 0.1878, 0.1256, 0.0859, 0.0485, 0.0253, 0.0119, 0.0055, 0.0035

## Adapter Output

The LoRA adapter was saved to outputs/sft/qwen36_lora_tiny_4gpu_ddp_20step/adapter.

## Notes

This run validates the 4-GPU DDP training path for Qwen3.6-35B-A3B LoRA SFT. Each rank loads one full bf16 copy of the model, so memory is not reduced compared with single-GPU LoRA. The value of this run is distributed training correctness and benchmark instrumentation.

The loss curve is not a quality signal because the dataset only has 4 tiny samples and is repeatedly cycled. The sharp loss decrease reflects tiny-sample overfitting.

## Conclusion

The 4-GPU DDP 20-step LoRA SFT baseline passed. This provides a distributed systems baseline for later experiments with larger datasets, longer sequence length, packing, gradient checkpointing comparisons, and DeepSpeed/ZeRO.
