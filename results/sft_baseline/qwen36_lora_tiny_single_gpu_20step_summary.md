# Qwen3.6-35B-A3B LoRA SFT 20-Step Single-GPU Baseline

## Summary

- Status: passed
- Python env: /workspace/sft_env
- Model: /workspace/models/Qwen3.6-35B-A3B
- GPU: 1 x NVIDIA H800 80GB
- Torch: 2.11.0+cu130
- CUDA runtime: 13.0
- Transformers: 5.12.1
- PEFT: 0.19.1
- dtype: bf16
- LoRA rank: 4
- LoRA alpha: 8
- target modules: q_proj, k_proj, v_proj, o_proj, in_proj_qkv, out_proj
- trainable params: 2,826,240
- all params: 34,663,436,928
- trainable ratio: 0.0082%
- max_length: 512
- max_steps: 20
- batch_size: 1
- optimizer: AdamW
- learning rate: 2e-4
- gradient checkpointing: enabled
- dataset: 4 tiny hand-written SFT samples

## Results

- total_time_s: 28.93
- average step time: 1.446 s
- average step time after warmup, step 4-19: about 1.018 s
- min step time: 0.944 s
- max step time: 3.466 s
- base model memory: 64.56 GiB
- peak training memory: 64.84 GiB

## Losses

0.5119, 0.4044, 2.0354, 2.1216, 0.1034, 0.0667, 1.6986, 1.7576, 0.0184, 0.0244, 1.2068, 1.3183, 0.0094, 0.0140, 0.8286, 0.9786, 0.0062, 0.0103, 0.4624, 0.5735

## Adapter Output

The LoRA adapter was saved to outputs/sft/qwen36_lora_tiny_single_gpu_20step/adapter.

Saved files include README.md, adapter_model.safetensors, adapter_config.json, chat_template.jinja, tokenizer_config.json, and tokenizer.json.

## Notes

The loss curve is not a quality signal because the dataset only has 4 tiny samples and is repeatedly cycled. The purpose of this run is to validate the training system path: model loading, LoRA injection, forward, loss, backward, optimizer step, memory logging, and adapter saving.

## Conclusion

The 20-step single-GPU LoRA SFT baseline passed. This run provides a systems baseline for later experiments with longer sequence length, larger data, 4-GPU distributed training, packing, and gradient-checkpointing comparisons.
