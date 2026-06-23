import argparse
import json
import os
import time
from functools import partial

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora_sft import (
    Collator,
    JsonlSFTDataset,
    LengthAwarePackedJsonlSFTDataset,
    PackedJsonlSFTDataset,
    TinySFTDataset,
    print_gpu_memory,
)


SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
}


def setup_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def resolve_transformer_layer_classes(model, class_names):
    requested = {name.strip() for name in class_names.split(",") if name.strip()}
    matched = set()
    for module in model.modules():
        if module.__class__.__name__ in requested:
            matched.add(module.__class__)
    return matched


def build_auto_wrap_policy(model, args, is_main):
    if args.fsdp_auto_wrap_policy == "none":
        return None

    if args.fsdp_auto_wrap_policy == "size":
        return partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)

    transformer_classes = resolve_transformer_layer_classes(model, args.fsdp_transformer_layer_classes)
    if not transformer_classes:
        if is_main:
            print(
                "warning: no transformer layer class matched "
                f"{args.fsdp_transformer_layer_classes!r}; falling back to size-based auto wrap"
            )
        return partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)

    if is_main:
        names = sorted(cls.__name__ for cls in transformer_classes)
        print("fsdp transformer auto wrap classes:", names)
    return partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_classes)


def build_dataset(args, tokenizer):
    if args.dataset_mode == "tiny":
        return TinySFTDataset(tokenizer, max_length=args.max_length)

    if args.packing and args.packing_strategy == "greedy":
        return PackedJsonlSFTDataset(tokenizer, args.data_path, max_length=args.max_length)
    if args.packing and args.packing_strategy == "length_aware":
        return LengthAwarePackedJsonlSFTDataset(tokenizer, args.data_path, max_length=args.max_length)
    return JsonlSFTDataset(tokenizer, args.data_path, max_length=args.max_length)


def reduce_mean(value, world_size):
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / world_size


def save_fsdp_adapter(model, tokenizer, adapter_dir, is_main):
    save_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_config):
        full_state_dict = model.state_dict()
        if is_main:
            adapter_state_dict = get_peft_model_state_dict(model.module, state_dict=full_state_dict)
            model.module.save_pretrained(adapter_dir, state_dict=adapter_state_dict)
            tokenizer.save_pretrained(adapter_dir)
            print("saved adapter:", adapter_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/Qwen3.6-35B-A3B")
    parser.add_argument("--output-dir", default="outputs/sft/qwen36_lora_fsdp_smoke")
    parser.add_argument("--data-path", default="data/agentic_coding_sft_sample.jsonl")
    parser.add_argument("--dataset-mode", choices=["tiny", "jsonl"], default="jsonl")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--packing-strategy", choices=["greedy", "length_aware"], default="greedy")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--save-adapter", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--fsdp-sharding-strategy", choices=sorted(SHARDING_STRATEGIES), default="full_shard")
    parser.add_argument("--fsdp-auto-wrap-policy", choices=["transformer", "size", "none"], default="transformer")
    parser.add_argument("--fsdp-min-num-params", type=int, default=100_000_000)
    parser.add_argument(
        "--fsdp-transformer-layer-classes",
        default="Qwen3MoeDecoderLayer,Qwen3DecoderLayer,Qwen2MoeDecoderLayer,Qwen2DecoderLayer,DecoderLayer",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()

    print("torch:", torch.__version__)
    print("rank:", rank, "local_rank:", local_rank, "world_size:", world_size)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        print("using device:", device, torch.cuda.get_device_name(local_rank))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model on CPU before FSDP wrapping...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
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

    auto_wrap_policy = build_auto_wrap_policy(model, args, is_main)
    mixed_precision = None
    if torch.cuda.is_available():
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # use_orig_params=True supports PEFT's mix of frozen base weights and trainable LoRA weights.
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=SHARDING_STRATEGIES[args.fsdp_sharding_strategy],
        device_id=device if torch.cuda.is_available() else None,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    model.train()

    print_gpu_memory("after fsdp model wrap")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    dataset = build_dataset(args, tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=Collator(tokenizer),
    )

    print("num samples:", len(dataset))
    print("max_length:", args.max_length)
    print("max_steps:", args.max_steps)
    print_gpu_memory("before training")

    losses = []
    metrics = []
    step = 0
    start = time.time()
    data_iter = iter(loader)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_start = time.time()

        accum_loss = 0.0
        accum_tokens = 0
        accum_samples = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = {key: value.to(device) for key, value in batch.items()}

            local_tokens = batch["attention_mask"].sum().detach()
            local_samples = torch.tensor(batch["input_ids"].shape[0], device=device, dtype=torch.long)

            global_tokens = local_tokens.clone()
            global_samples = local_samples.clone()
            dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_samples, op=dist.ReduceOp.SUM)

            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / args.grad_accum_steps
            loss.backward()

            log_loss = reduce_mean(raw_loss, world_size)
            accum_loss += float(log_loss.cpu())
            accum_tokens += int(global_tokens.cpu())
            accum_samples += int(global_samples.cpu())

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_memory = 0.0

        step_time = time.time() - step_start
        loss_value = accum_loss / args.grad_accum_steps
        tokens_per_second = accum_tokens / step_time
        samples_per_second = accum_samples / step_time
        theoretical_max_tokens = args.max_length * args.per_device_batch_size * args.grad_accum_steps * world_size
        token_utilization = accum_tokens / theoretical_max_tokens

        if is_main:
            losses.append(loss_value)
            row = {
                "step": step,
                "avg_loss": loss_value,
                "rank0_step_time_s": round(step_time, 4),
                "global_tokens_per_step": accum_tokens,
                "theoretical_max_tokens_per_update": theoretical_max_tokens,
                "token_utilization": round(token_utilization, 4),
                "global_samples_per_step": accum_samples,
                "tokens_per_second": round(tokens_per_second, 2),
                "samples_per_second": round(samples_per_second, 4),
                "rank0_max_memory_gib": round(peak_memory, 2),
                "world_size": world_size,
                "grad_accum_steps": args.grad_accum_steps,
            }
            metrics.append(row)
            print(json.dumps(row, ensure_ascii=False))
            print_gpu_memory(f"after step {step}")

        step += 1

    total_time = time.time() - start
    summary = {
        "status": "ok",
        "trainer": "fsdp",
        "model_path": args.model_path,
        "max_length": args.max_length,
        "dataset_mode": args.dataset_mode,
        "data_path": args.data_path,
        "packing": args.packing,
        "packing_strategy": args.packing_strategy,
        "max_steps": args.max_steps,
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "global_batch_size_per_update": args.per_device_batch_size * args.grad_accum_steps * world_size,
        "theoretical_max_tokens_per_update": args.max_length * args.per_device_batch_size * args.grad_accum_steps * world_size,
        "lora_rank": args.lora_rank,
        "distributed": True,
        "world_size": world_size,
        "loss_label": "avg_loss_across_ranks",
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "fsdp_sharding_strategy": args.fsdp_sharding_strategy,
        "fsdp_auto_wrap_policy": args.fsdp_auto_wrap_policy,
        "fsdp_transformer_layer_classes": args.fsdp_transformer_layer_classes,
        "losses": losses,
        "metrics": metrics,
        "total_time_s": total_time,
    }

    dist.barrier()
    if is_main:
        summary_path = os.path.join(args.output_dir, "tiny_lora_sft_fsdp_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("saved summary:", summary_path)
        print(json.dumps(summary, indent=2))

        if args.save_adapter:
            adapter_dir = os.path.join(args.output_dir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
    dist.barrier()

    if args.save_adapter:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        save_fsdp_adapter(model, tokenizer, adapter_dir, is_main)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
