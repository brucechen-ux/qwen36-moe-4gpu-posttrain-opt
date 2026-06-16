#!/usr/bin/env python3
"""Reference HuggingFace MoE router hook for expert-hit profiling."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


LOGGER = logging.getLogger("hook_moe_router")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach forward hooks to router/gate modules and record expert hit counts. "
            "Module names are model-implementation dependent; adjust --router-regex after inspection."
        )
    )
    parser.add_argument("--model-path", required=True, help="Local HuggingFace model path or model id.")
    parser.add_argument("--router-regex", default="router|gate", help="Regex matched against module names.")
    parser.add_argument("--prompt", default=None, help="Optional prompt for a one-pass probe.")
    parser.add_argument("--output", default="reports/moe_router_profile.json", help="Output JSON path.")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k experts to count from router logits.")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="New tokens for generate mode.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument("--allow-download", action="store_true", help="Allow HuggingFace downloads. Default is local only.")
    parser.add_argument("--dry-run", action="store_true", help="Only list matching modules; do not run a prompt.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class MoERouterProfiler:
    """Collect expert hit counts from matched router modules."""

    def __init__(self, model: Any, router_regex: str, top_k: int) -> None:
        self.model = model
        self.router_pattern = re.compile(router_regex)
        self.top_k = top_k
        self.handles: List[Any] = []
        self.counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.tokens: Dict[str, int] = defaultdict(int)
        self.matched_modules: List[str] = []

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if not name:
                continue
            if self.router_pattern.search(name):
                self.matched_modules.append(name)
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        LOGGER.info("Matched %s router candidate modules", len(self.matched_modules))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            try:
                router_tensor = self._extract_router_tensor(output)
                if router_tensor is None:
                    return
                self._record_tensor(name, router_tensor)
            except Exception as exc:  # noqa: BLE001 - hooks should not crash model execution.
                LOGGER.debug("Router hook failed for %s: %s", name, exc)

        return hook

    def _extract_router_tensor(self, output: Any) -> Optional[Any]:
        """Return a tensor shaped [..., num_experts] from common router outputs.

        Adapt this method if the target model returns a custom object or already-selected expert ids.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - only relevant without torch installed.
            raise RuntimeError("torch is required for router profiling") from exc

        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value):
                    return value
        if isinstance(output, dict):
            for key in ("router_logits", "logits", "scores"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        for attr in ("router_logits", "logits", "scores"):
            value = getattr(output, attr, None)
            if torch.is_tensor(value):
                return value
        return None

    def _record_tensor(self, name: str, tensor: Any) -> None:
        import torch

        if tensor.ndim < 2:
            return
        expert_dim = tensor.shape[-1]
        if expert_dim <= 1:
            return
        flat = tensor.reshape(-1, expert_dim).detach()
        top_k = min(self.top_k, expert_dim)
        indices = torch.topk(flat, k=top_k, dim=-1).indices.reshape(-1).to("cpu")
        self.tokens[name] += int(flat.shape[0])
        unique, counts = torch.unique(indices, return_counts=True)
        for expert_id, count in zip(unique.tolist(), counts.tolist()):
            self.counts[name][int(expert_id)] += int(count)

    def summary(self) -> Dict[str, Any]:
        layers = {}
        for name in self.matched_modules:
            counts = dict(sorted(self.counts.get(name, {}).items()))
            total_hits = sum(counts.values())
            layers[name] = {
                "tokens_observed": self.tokens.get(name, 0),
                "expert_hits": {str(k): v for k, v in counts.items()},
                "total_hits": total_hits,
                "top_k": self.top_k,
            }
        return {
            "router_regex": self.router_pattern.pattern,
            "matched_modules": self.matched_modules,
            "layers": layers,
        }


def load_model_and_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required; install it in the H800 environment") from exc

    local_files_only = not args.allow_download
    LOGGER.info("Loading tokenizer from %s local_files_only=%s", args.model_path, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    LOGGER.info("Loading model from %s local_files_only=%s", args.model_path, local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
        device_map=args.device_map,
    )
    model.eval()
    return model, tokenizer


def run_prompt(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> None:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    first_param = next(model.parameters(), None)
    if first_param is not None:
        inputs = {key: value.to(first_param.device) for key, value in inputs.items()}

    with torch.no_grad():
        if max_new_tokens > 0:
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            model(**inputs)


def write_summary(summary: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote router profile to %s", output)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        model, tokenizer = load_model_and_tokenizer(args)
        profiler = MoERouterProfiler(model, args.router_regex, args.top_k)
        profiler.attach()

        if args.dry_run:
            LOGGER.info("Dry run complete. Matched modules: %s", profiler.matched_modules)
        else:
            if not args.prompt:
                raise ValueError("--prompt is required unless --dry-run is set")
            run_prompt(model, tokenizer, args.prompt, args.max_new_tokens)

        write_summary(profiler.summary(), Path(args.output))
        profiler.close()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("MoE router profiling failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
