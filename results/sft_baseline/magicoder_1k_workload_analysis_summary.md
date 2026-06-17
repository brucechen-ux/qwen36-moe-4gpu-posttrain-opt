# Magicoder 1K Real Code SFT Workload Analysis

## Status

- Real code SFT dataset conversion: completed
- Dataset source: ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response
- Converted sample file: data/magicoder_oss_instruct_1k.jsonl
- Number of records: 1000
- Length stats file: results/sft_baseline/magicoder_1k_length_stats.json

## Token Length Distribution

- total_tokens min: 200
- total_tokens p50: 518
- total_tokens p90: 741
- total_tokens p95: 815
- total_tokens p99: 1021
- total_tokens max: 1629
- total_tokens mean: 538.21

## Length Buckets

- <=512: 482
- 513-1024: 508
- 1025-2048: 10
- 2049-4096: 0
- >4096: 0

## Analysis

Magicoder 1K is a medium-short code SFT workload rather than a long-context workload. Since p99 total length is 1021 tokens, max_length=1024 is a reasonable primary benchmark setting. Only 10 out of 1000 samples exceed 1024 tokens, and no samples exceed 2048 tokens.

The expected no-packing token utilization is around mean_total_tokens / max_length = 538.21 / 1024 = about 52.6%. This matches the first no-packing run where token_utilization was mostly around 0.5.

## Known Issue

A first no-packing 50-step run completed but produced NaN loss at step 18 and step 46. The likely cause is SFT truncation: some samples have prompts longer than max_length, so the answer labels can be fully truncated, leaving all labels as -100. This should be fixed by skipping examples with no supervised answer tokens after truncation.

## Next Steps

1. Patch train_lora_sft.py to skip samples whose labels are all -100 after truncation.
2. Re-run Magicoder 1K max_length=1024 no-packing 50-step clean baseline.
3. Run Magicoder 1K max_length=1024 packing 50-step baseline.
4. Compare token_utilization, tokens/sec, step time, memory, and loss stability.
