# vLLM TP=4 8K Serving Baseline

## Environment

- Backend: vLLM 0.23.0
- Model: qwen3_6_35b_a3b
- Model path: /workspace/models/Qwen3.6-35B-A3B
- Hardware: 4 x NVIDIA H800 80GB
- Tensor parallel size: 4
- Max model length: 8192
- GDN prefill backend: triton
- Endpoint: http://127.0.0.1:8000/v1/chat/completions

## Launch Notes

Default FlashInfer GDN prefill JIT failed with ninja / nvcc killed code 137, so serving was launched with:

```bash
--gdn-prefill-backend triton
```

The model successfully served `/v1/models` and `/v1/chat/completions`.

For Qwen-style chat requests, `chat_template_kwargs.enable_thinking=false` is required to avoid reasoning-style output.

## 5-Prompt Streaming Benchmark: Default Thinking

| Metric | Value |
|---|---:|
| total_requests | 5 |
| successes | 5 |
| failures | 0 |
| success_rate | 1.0 |
| mean_ttft_s | 0.04595 |
| mean_total_latency_s | 0.94219 |
| mean_output_tokens_per_s | 285.64 |

## 5-Prompt Streaming Benchmark: No Thinking

| Metric | Value |
|---|---:|
| total_requests | 5 |
| successes | 5 |
| failures | 0 |
| success_rate | 1.0 |
| mean_ttft_s | 0.04195 |
| mean_total_latency_s | 0.82973 |
| mean_output_tokens_per_s | 285.36 |

## Per-request No-thinking Results

| request_id | success | TTFT (s) | latency (s) | completion_tokens | output tok/s |
|---|---:|---:|---:|---:|---:|
| p001 | true | 0.0588 | 0.6901 | 180 | 285.16 |
| p002 | true | 0.0373 | 0.9340 | 256 | 285.49 |
| p003 | true | 0.0396 | 0.6559 | 176 | 285.58 |
| p004 | true | 0.0361 | 0.9326 | 256 | 285.55 |
| p005 | true | 0.0379 | 0.9360 | 256 | 285.02 |

## Notes

- `scripts/bench_serving.py` was updated with `--disable-thinking`.
- `--disable-thinking` injects:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

- No-thinking output was verified with:

```bash
grep -n "Thinking Process\|Here's a thinking process" \
  results/serving_baseline/vllm_tp4_8k_5prompts_stream_no_thinking_v2.jsonl \
  || echo "no thinking text found"
```

- Result: `no thinking text found`.
- No-thinking reduces total latency mainly because some outputs finish before hitting `max_tokens=256`.
