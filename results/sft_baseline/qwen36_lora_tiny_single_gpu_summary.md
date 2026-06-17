# Qwen3.6-35B-A3B LoRA SFT Tiny Single-GPU Baseline

## Goal

This run verifies the minimal supervised fine-tuning training loop for Qwen3.6-35B-A3B:

- model loading
- LoRA adapter injection
- forward pass
- causal LM loss
- backward pass
- optimizer step
- GPU memory logging

This is a sanity baseline, not a quality training run.

## Environment

- Python env: `/workspace/sft_env`
- Model path: `/workspace/models/Qwen3.6-35B-A3B`
- GPU: 1 × NVIDIA H800 80GB
- Torch: 2.11.0+cu130
- CUDA runtime: 13.0
- Transformers: 5.12.1
- PEFT: 0.19.1

## Model Config

- model_type: qwen3_5_moe
- text model_type: qwen3_5_moe_text
- hidden_size: 2048
- num_hidden_layers: 40
- num_attention_heads: 16
- num_key_value_heads: 2
- num_experts: 256
- num_experts_per_tok: 8
- moe_intermediate_size: 512
- vocab_size: 248320

## LoRA Config

- finetuning_type: LoRA
- dtype: bf16
- lora_rank: 4
- lora_alpha: 8
- target modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - in_proj_qkv
  - out_proj

## Trainable Parameters

- trainable params: 2,826,240
- all params: 34,663,436,928
- trainable ratio: 0.0082%

## Training Setup

- max_length: 512
- max_steps: 2
- batch_size: 1
- optimizer: AdamW
- lr: 2e-4
- gradient checkpointing: enabled
- dataset: 4 tiny hand-written SFT samples

## Results

| step | loss | step_time_s | peak_memory_gib |
| ---: | ---: | ----------: | --------------: |
| 0 | 0.5119276643 | 3.537 | 64.74 |
| 1 | 0.4050170183 | 3.127 | 64.80 |

## Memory

- after base model load: 64.56 GiB allocated
- before training: 64.57 GiB allocated
- peak during training: 64.80 GiB

## Conclusion

The tiny single-GPU LoRA SFT sanity baseline passed. The run confirms that Qwen3.6-35B-A3B can be loaded in bf16 on one H800, LoRA adapters can be injected into attention and linear-attention modules, and the full forward/loss/backward/optimizer-step loop works.
