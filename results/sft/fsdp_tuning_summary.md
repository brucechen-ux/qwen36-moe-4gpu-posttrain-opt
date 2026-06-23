# FSDP Communication Strategy Tuning

Model: Qwen3.6-35B-A3B  
Hardware: 4 × H800 80GB  
Trainer: PyTorch FSDP FULL_SHARD + LoRA SFT  
Dataset: Magicoder OSS Instruct 1K  
Packing: length-aware packing  
max_length: 2048  
gradient_checkpointing: ON  
per_device_batch_size: 1  
grad_accum_steps: 4  
global_batch_size_per_update: 16  

## 5-step Smoke Matrix

| config | backward_prefetch | limit_all_gathers | status | peak_memory_gib | note |
|---|---|---:|---|---:|---|
| baseline | pre | true | success | ~42.30 | default after patch |
| A | post | true | success | ~40.74 | lower memory |
| B | none | true | success | ~40.74 | best smoke throughput |
| C | pre | false | success | ~42.30 | higher reserved memory |

## 50-step Benchmark

| config | backward_prefetch | limit_all_gathers | total_time_s | avg_step_time_s | avg_tokens_per_second | total_tokens_per_second | peak_memory_gib |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | pre | true | 709.30 | 14.19 | 2341.98 | 2257.93 | 42.30 |
| tuned | none | true | 687.61 | 13.75 | 2416.94 | 2329.15 | 40.74 |

## Conclusion

On the 2048-token FSDP + GC ON workload, disabling backward prefetch while keeping limit_all_gathers enabled reduced peak memory from about 42.30 GiB to 40.74 GiB and improved average tokens/sec by about 3.2%. This suggests that aggressive backward prefetching is not beneficial for this workload; the more conservative configuration reduces parameter all-gather pressure while maintaining stable throughput.
