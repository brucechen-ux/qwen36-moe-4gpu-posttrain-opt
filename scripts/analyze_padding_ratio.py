#!/usr/bin/env python3
"""Analyze sequence length distribution and padding ratio from jsonl data."""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOGGER = logging.getLogger("analyze_padding_ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute length percentiles and padding ratio.")
    parser.add_argument("--input", required=True, help="Input jsonl file.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--length-field", default="length", help="Field with tokenized length.")
    parser.add_argument("--text-field", default="text", help="Field with raw text if length is missing.")
    parser.add_argument("--seq-len", type=int, required=True, help="Target sequence length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for non-packing padding estimate.")
    parser.add_argument("--packing", action="store_true", help="Estimate packed fixed-block padding ratio.")
    parser.add_argument("--sort-by-length", action="store_true", help="Sort by length before batching.")
    parser.add_argument(
        "--estimate-tokens",
        choices=("chars", "whitespace"),
        default="chars",
        help="Fallback raw-text token estimator.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Limit records. 0 means all.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_nested(item: Dict[str, Any], field: str) -> Any:
    current: Any = item
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def estimate_text_tokens(text: str, method: str) -> int:
    if method == "whitespace":
        return max(1, len(text.split())) if text else 0
    return max(1, math.ceil(len(text) / 4)) if text else 0


def read_lengths(args: argparse.Namespace) -> List[int]:
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    lengths: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if args.max_records and len(lengths) >= args.max_records:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")

            length_value = get_nested(item, args.length_field)
            if length_value is not None:
                try:
                    length = int(length_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_no}: invalid length value {length_value!r}") from exc
            else:
                text_value = get_nested(item, args.text_field)
                if not isinstance(text_value, str):
                    raise ValueError(
                        f"{path}:{line_no}: missing numeric '{args.length_field}' and string '{args.text_field}'"
                    )
                length = estimate_text_tokens(text_value, args.estimate_tokens)

            if length < 0:
                raise ValueError(f"{path}:{line_no}: length must be non-negative")
            lengths.append(length)

    if not lengths:
        raise ValueError("no lengths found")
    return lengths


def percentile(sorted_values: List[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def compute_non_packing_padding(lengths: List[int], seq_len: int, batch_size: int, sort_by_length: bool) -> Dict[str, Any]:
    ordered = sorted(lengths) if sort_by_length else list(lengths)
    actual_tokens = 0
    padded_tokens = 0
    truncated_records = 0

    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        clipped = [min(length, seq_len) for length in batch]
        truncated_records += sum(1 for length in batch if length > seq_len)
        batch_max = max(clipped) if clipped else 0
        actual_tokens += sum(clipped)
        padded_tokens += batch_max * len(clipped)

    padding_tokens = padded_tokens - actual_tokens
    return {
        "mode": "batch_padding",
        "actual_tokens_after_truncation": actual_tokens,
        "padded_tokens": padded_tokens,
        "padding_tokens": padding_tokens,
        "padding_ratio": padding_tokens / padded_tokens if padded_tokens else 0.0,
        "truncated_records": truncated_records,
    }


def compute_packing_padding(lengths: List[int], seq_len: int) -> Dict[str, Any]:
    clipped = [min(length, seq_len) for length in lengths]
    actual_tokens = sum(clipped)
    blocks = math.ceil(actual_tokens / seq_len) if actual_tokens else 0
    padded_tokens = blocks * seq_len
    padding_tokens = padded_tokens - actual_tokens
    return {
        "mode": "packing",
        "actual_tokens_after_truncation": actual_tokens,
        "packed_blocks": blocks,
        "padded_tokens": padded_tokens,
        "padding_tokens": padding_tokens,
        "padding_ratio": padding_tokens / padded_tokens if padded_tokens else 0.0,
        "truncated_records": sum(1 for length in lengths if length > seq_len),
    }


def histogram(lengths: List[int], seq_len: int) -> Dict[str, int]:
    buckets = {
        "<=512": 0,
        "513-1024": 0,
        "1025-2048": 0,
        "2049-4096": 0,
        "4097-8192": 0,
        "8193-16384": 0,
        "16385-32768": 0,
        ">32768": 0,
        f">{seq_len}": 0,
    }
    for length in lengths:
        if length <= 512:
            buckets["<=512"] += 1
        elif length <= 1024:
            buckets["513-1024"] += 1
        elif length <= 2048:
            buckets["1025-2048"] += 1
        elif length <= 4096:
            buckets["2049-4096"] += 1
        elif length <= 8192:
            buckets["4097-8192"] += 1
        elif length <= 16384:
            buckets["8193-16384"] += 1
        elif length <= 32768:
            buckets["16385-32768"] += 1
        else:
            buckets[">32768"] += 1
        if length > seq_len:
            buckets[f">{seq_len}"] += 1
    return buckets


def build_summary(lengths: List[int], args: argparse.Namespace) -> Dict[str, Any]:
    sorted_lengths = sorted(lengths)
    padding = (
        compute_packing_padding(lengths, args.seq_len)
        if args.packing
        else compute_non_packing_padding(lengths, args.seq_len, args.batch_size, args.sort_by_length)
    )
    return {
        "input": args.input,
        "records": len(lengths),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "packing": args.packing,
        "sort_by_length": args.sort_by_length,
        "mean": statistics.fmean(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "p50": percentile(sorted_lengths, 50),
        "p90": percentile(sorted_lengths, 90),
        "p99": percentile(sorted_lengths, 99),
        "histogram": histogram(lengths, args.seq_len),
        "padding": padding,
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        if args.seq_len <= 0:
            raise ValueError("--seq-len must be positive")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        lengths = read_lengths(args)
        summary = build_summary(lengths, args)
        rendered = json.dumps(summary, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
            LOGGER.info("Wrote summary to %s", output_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to analyze padding ratio: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
