import json
from pathlib import Path
from datasets import load_dataset

dataset_name = "ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response"
out_path = Path("data/magicoder_oss_instruct_1k.jsonl")
max_records = 1000

print("loading dataset:", dataset_name)
ds = load_dataset(dataset_name, split="train")
print("num_rows:", len(ds))
print("columns:", ds.column_names)
print("first row:", ds[0])

def pick(row, names):
    for name in names:
        if name in row and row[name] is not None:
            val = row[name]
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""

written = 0
with out_path.open("w", encoding="utf-8") as f:
    for row in ds:
        instruction = pick(row, ["instruction", "problem", "prompt", "query", "question"])
        output = pick(row, ["response", "output", "solution", "answer", "completion"])
        code = pick(row, ["code", "canonical_solution"])
        lang = pick(row, ["lang", "language"])

        if not output and code:
            output = code

        if not instruction or not output:
            continue

        record = {
            "instruction": instruction,
            "input": f"Language: {lang}" if lang else "",
            "output": output,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

        if written >= max_records:
            break

print("wrote:", out_path)
print("records:", written)
if written == 0:
    raise RuntimeError("No records written. Check dataset columns above.")
