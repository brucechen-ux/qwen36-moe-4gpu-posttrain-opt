import json
import statistics
from pathlib import Path
from transformers import AutoTokenizer

model_path = "/workspace/models/Qwen3.6-35B-A3B"
data_path = Path("data/magicoder_oss_instruct_1k.jsonl")
out_path = Path("results/sft_baseline/magicoder_1k_length_stats.json")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def build_prompt_text(instruction, input_text):
    user_text = instruction if not input_text else instruction + "\n\n" + input_text
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def pct(xs, p):
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * p / 100))
    return xs[idx]

rows = []
with data_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

prompt_lens = []
answer_lens = []
total_lens = []

for r in rows:
    prompt_text = build_prompt_text(r["instruction"], r.get("input", ""))
    answer_text = r["output"] + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    prompt_lens.append(len(prompt_ids))
    answer_lens.append(len(answer_ids))
    total_lens.append(len(prompt_ids) + len(answer_ids))

stats = {
    "num_records": len(rows),
    "prompt_tokens": {
        "min": min(prompt_lens),
        "p50": pct(prompt_lens, 50),
        "p90": pct(prompt_lens, 90),
        "p95": pct(prompt_lens, 95),
        "p99": pct(prompt_lens, 99),
        "max": max(prompt_lens),
        "mean": round(statistics.mean(prompt_lens), 2),
    },
    "answer_tokens": {
        "min": min(answer_lens),
        "p50": pct(answer_lens, 50),
        "p90": pct(answer_lens, 90),
        "p95": pct(answer_lens, 95),
        "p99": pct(answer_lens, 99),
        "max": max(answer_lens),
        "mean": round(statistics.mean(answer_lens), 2),
    },
    "total_tokens": {
        "min": min(total_lens),
        "p50": pct(total_lens, 50),
        "p90": pct(total_lens, 90),
        "p95": pct(total_lens, 95),
        "p99": pct(total_lens, 99),
        "max": max(total_lens),
        "mean": round(statistics.mean(total_lens), 2),
    },
    "length_buckets_total_tokens": {
        "<=512": sum(x <= 512 for x in total_lens),
        "513-1024": sum(512 < x <= 1024 for x in total_lens),
        "1025-2048": sum(1024 < x <= 2048 for x in total_lens),
        "2049-4096": sum(2048 < x <= 4096 for x in total_lens),
        ">4096": sum(x > 4096 for x in total_lens),
    },
}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(stats, indent=2, ensure_ascii=False))
print("saved:", out_path)
