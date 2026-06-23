# FSDP Sequence Capacity Sweep

Model: Qwen3.6-35B-A3B  
Hardware: 4 × H800 80GB  
Trainer: PyTorch FSDP FULL_SHARD + LoRA SFT  
Dataset: Magicoder OSS Instruct 1K  
Packing: length-aware packing  
per_device_batch_size: 1  
grad_accum_steps: 4  
global_batch_size_per_update: 16  

| max_length | GC | status | peak_memory_gib | note |
|---:|---:|---:|---:|---|
| 1024 | ON | success | ~40.69 | 50-step benchmark |
| 1024 | OFF | success | ~63.17 | 50-step benchmark |
| 1536 | OFF | success | ~74.79 | 5-step smoke; close to 80GB boundary |
| 2048 | OFF | OOM | ~79GB | OOM during first forward |
| 2048 | ON | success | ~42.30 | 50-step benchmark |

## Conclusion

FSDP FULL_SHARD reduces replicated model-state memory and makes GC OFF feasible at max_length=1024 and 1536. However, at max_length=2048, GC OFF hits an activation memory wall and OOMs during the first forward pass. Enabling gradient checkpointing trades recomputation for memory reduction and successfully restores 2048-token training capacity.


## 2048 GC ON 50-step Benchmark

FSDP + GC ON successfully completed a 50-step benchmark at max_length=2048. This confirms that gradient checkpointing restores long-sequence training capacity after GC OFF hits an activation-memory OOM at 2048.
