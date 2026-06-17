# Qwen3.6-35B-A3B LoRA SFT 4-GPU DDP Smoke Baseline

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
- max_steps: 2
- batch_size per rank: 1
- LoRA rank: 4
- LoRA alpha: 8
- target modules: q_proj, k_proj, v_proj, o_proj, in_proj_qkv, out_proj
- trainable params: 2,826,240
- all params: 34,663,436,928
- trainable ratio: 0.0082%

## Results

- 4 ranks launched successfully
- each rank loaded the bf16 base model successfully
- DDP forward, loss, backward, and optimizer step completed
- rank0 saved tiny_lora_sft_summary.json
- rank0 saved the LoRA adapter
- no OOM, Traceback, or NCCL error observed
- rank0 total_time_s: 5.13
- rank0 losses: 0.5119, 0.4641
- peak memory per rank: about 64.8 GiB

## Adapter Output

The LoRA adapter was saved to outputs/sft/qwen36_lora_tiny_4gpu_ddp_smoke/adapter.

## Notes

This experiment validates the distributed training path. It does not reduce memory because each DDP rank loads one full bf16 copy of the 35B model.

The rank0 summary records rank0 local losses only. A future formal DDP benchmark should all-reduce losses across ranks.

## Conclusion

The 4-GPU DDP LoRA SFT smoke baseline passed. This confirms that the project can run Qwen3.6-35B-A3B LoRA training with torchrun/DDP across 4 H800 GPUs.
