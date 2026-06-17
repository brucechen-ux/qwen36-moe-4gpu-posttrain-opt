import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


SAMPLES = [
    {
        "instruction": "Write a Python function add(a, b) that returns a + b.",
        "output": "def add(a, b):\n    return a + b",
    },
    {
        "instruction": "Write a Python function is_even(n) that returns whether n is even.",
        "output": "def is_even(n):\n    return n % 2 == 0",
    },
    {
        "instruction": "Explain the difference between prefill and decode in LLM inference.",
        "output": "Prefill processes the prompt in parallel and builds the KV cache. Decode generates new tokens one by one while reusing the KV cache.",
    },
    {
        "instruction": "Explain what gradient checkpointing does.",
        "output": "Gradient checkpointing reduces activation memory by recomputing some intermediate activations during backward. It trades extra compute for lower GPU memory usage.",
    },
]


def normalize_token_ids(x) -> List[int]:
    """Convert list / tensor / BatchEncoding / nested input_ids to a flat list[int]."""
    if hasattr(x, "input_ids"):
        x = x.input_ids
    elif isinstance(x, dict) and "input_ids" in x:
        x = x["input_ids"]

    if hasattr(x, "tolist"):
        x = x.tolist()

    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        x = x[0]

    return list(x)


def build_prompt_ids(tokenizer, instruction: str) -> List[int]:
    messages = [{"role": "user", "content": instruction}]
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
    except TypeError:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    return normalize_token_ids(out)


class TinySFTDataset(Dataset):
    def __init__(self, tokenizer, max_length: int):
        self.examples = []
        eos = tokenizer.eos_token or ""
        for sample in SAMPLES:
            prompt_ids = build_prompt_ids(tokenizer, sample["instruction"])
            answer_ids = tokenizer(
                sample["output"] + eos,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = [1] * len(input_ids)

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class Collator:
    tokenizer: object

    def __call__(self, features: List[Dict]):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for x in features:
            pad_len = max_len - len(x["input_ids"])
            batch["input_ids"].append(x["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(x["attention_mask"] + [0] * pad_len)
            batch["labels"].append(x["labels"] + [-100] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def print_gpu_memory(prefix: str):
    if not torch.cuda.is_available():
        print(f"{prefix}: cuda not available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"{prefix}: allocated={allocated:.2f} GiB, "
        f"reserved={reserved:.2f} GiB, max_allocated={max_allocated:.2f} GiB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/Qwen3.6-35B-A3B")
    parser.add_argument("--output-dir", default="outputs/sft/qwen36_lora_tiny_smoke")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-adapter", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    distributed = args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        args.device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    if distributed:
        dist.barrier()

    print("torch:", torch.__version__)
    print("rank:", rank, "local_rank:", local_rank, "world_size:", world_size)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        print("using device:", args.device, torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": args.device} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    if args.no_gradient_checkpointing:
        if is_main:
            print("gradient checkpointing: disabled")
    else:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if is_main:
            print("gradient checkpointing: enabled")

    print_gpu_memory("after base model load")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "out_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    model.train()

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    dataset = TinySFTDataset(tokenizer, max_length=args.max_length)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        collate_fn=Collator(tokenizer),
    )

    print("num samples:", len(dataset))
    print("max_length:", args.max_length)
    print("max_steps:", args.max_steps)
    print_gpu_memory("before training")

    losses = []
    step = 0
    start = time.time()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            batch = {k: v.to(args.device) for k, v in batch.items()}

            torch.cuda.reset_peak_memory_stats()
            step_start = time.time()

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            step_time = time.time() - step_start

            log_loss = loss.detach()
            if distributed:
                log_loss = log_loss.clone()
                dist.all_reduce(log_loss, op=dist.ReduceOp.SUM)
                log_loss = log_loss / world_size

            loss_value = float(log_loss.cpu())

            if is_main:
                losses.append(loss_value)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "avg_loss" if distributed else "loss": loss_value,
                            "rank0_step_time_s": round(step_time, 4),
                            "rank0_max_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                            "world_size": world_size,
                        },
                        ensure_ascii=False,
                    )
                )
                print_gpu_memory(f"after step {step}")

            step += 1

    total_time = time.time() - start
    summary = {
        "status": "ok",
        "model_path": args.model_path,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "lora_rank": args.lora_rank,
        "distributed": distributed,
        "world_size": world_size,
        "loss_label": "avg_loss_across_ranks" if distributed else "loss",
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "losses": losses,
        "total_time_s": total_time,
    }

    if distributed:
        dist.barrier()

    if is_main:
        summary_path = os.path.join(args.output_dir, "tiny_lora_sft_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("saved summary:", summary_path)
        print(json.dumps(summary, indent=2))

        if args.save_adapter:
            adapter_dir = os.path.join(args.output_dir, "adapter")
            save_model = model.module if distributed else model
            save_model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            print("saved adapter:", adapter_dir)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
