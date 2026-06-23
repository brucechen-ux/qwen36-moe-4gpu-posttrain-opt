# FSDP Strategy Tuning Summary

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

## 1. Communication Strategy Tuning

The first sweep tuned FSDP communication behavior while keeping transformer-layer auto-wrap.

| config | backward_prefetch | limit_all_gathers | status | peak_memory_gib | note |
|---|---|---:|---|---:|---|
| baseline | pre | true | success | ~42.30 | default after patch |
| A | post | true | success | ~40.74 | lower memory |
| B | none | true | success | ~40.74 | best smoke throughput |
| C | pre | false | success | ~42.30 | higher reserved memory |

### 50-step Communication Benchmark

| config | auto_wrap | backward_prefetch | limit_all_gathers | total_time_s | avg_step_time_s | avg_tokens_per_second | total_tokens_per_second | peak_memory_gib |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | transformer | pre | true | 709.30 | 14.19 | 2341.98 | 2257.93 | 42.30 |
| tuned | transformer | none | true | 687.61 | 13.75 | 2416.94 | 2329.15 | 40.74 |

Communication tuning shows that disabling backward prefetch while keeping limit_all_gathers enabled improves throughput and reduces peak memory for this workload.

## 2. Auto-wrap Granularity Tuning

The second sweep fixed the best communication setting and tuned FSDP auto-wrap granularity.

| config | auto_wrap_policy | min_num_params | status | total_time_s | avg_step_time_s | avg_tokens_per_second | peak_memory_gib |
|---|---|---:|---|---:|---:|---:|---:|
| transformer tuned smoke | transformer | - | success | 81.93 | 16.39 | 2074.00 | 40.74 |
| size-50M smoke | size | 50000000 | success | 78.13 | 15.63 | 2132.79 | 26.81 |
| size-100M smoke | size | 100000000 | success | 77.74 | 15.55 | 2139.66 | 26.81 |
| size-200M smoke | size | 200000000 | success | 78.16 | 15.63 | 2128.08 | 26.81 |

### 50-step Auto-wrap Benchmark

| config | auto_wrap_policy | min_num_params | backward_prefetch | limit_all_gathers | total_time_s | avg_step_time_s | avg_tokens_per_second | total_tokens_per_second | peak_memory_gib |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| throughput-oriented | transformer | - | none | true | 687.61 | 13.75 | 2416.94 | 2329.15 | 40.74 |
| memory-oriented | size | 100000000 | none | true | 716.97 | 14.34 | 2312.88 | 2233.78 | 26.81 |

## 3. Interpretation

FSDP tuning produced two useful configurations instead of a single global optimum.

### Throughput-oriented configuration

- auto_wrap_policy: transformer
- backward_prefetch: none
- limit_all_gathers: true

This configuration gives the best 50-step throughput in the tested search space.

### Memory-oriented configuration

- auto_wrap_policy: size
- fsdp_min_num_params: 100M
- backward_prefetch: none
- limit_all_gathers: true

This configuration reduces peak memory from about 40.74 GiB to about 26.81 GiB, a reduction of about 34.2%, at the cost of about 4.3% lower average tokens/sec.

The size-based 50M, 100M, and 200M smoke tests all reached approximately the same peak memory, suggesting that lowering the threshold below 100M did not provide additional memory savings for this workload.

## 4. Conclusion

This should be described as a limited FSDP strategy sweep and a best-known configuration search under the tested workload, not as a proof of global FSDP optimality.

The final recommendation is:

- Use transformer-layer auto-wrap when throughput is the priority.
- Use size-based 100M auto-wrap when memory headroom is the priority.
