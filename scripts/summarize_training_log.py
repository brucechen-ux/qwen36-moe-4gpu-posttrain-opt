#!/usr/bin/env python3
"""Extract common training metrics from text logs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOGGER = logging.getLogger("summarize_training_log")

PATTERNS = {
    "step": [
        re.compile(r"(?:^|[\s,{])(?:global_)?step(?:\s*[:=]\s*|\s+)(?P<value>\d+)", re.IGNORECASE),
        re.compile(r"(?:^|\s)(?P<value>\d+)\s*/\s*\d+(?:\s|\[|$)", re.IGNORECASE),
    ],
    "loss": [
        re.compile(r"(?:^|[\s,{'\"])(?:train_)?loss['\"]?(?:\s*[:=]\s*)(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE),
    ],
    "learning_rate": [
        re.compile(r"(?:learning_rate|learn_rate|lr)['\"]?(?:\s*[:=]\s*)(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE),
    ],
    "step_time_s": [
        re.compile(r"(?:step[_\s-]?time|time/step|seconds/step)['\"]?(?:\s*[:=]\s*)(?P<value>\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", re.IGNORECASE),
        re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?:s/it|sec/it)", re.IGNORECASE),
    ],
    "tokens_per_s": [
        re.compile(r"(?:tokens/s|tok/s|tokens_per_s|train_tokens_per_second)['\"]?(?:\s*[:=]\s*)(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE),
    ],
    "peak_memory_mb": [
        re.compile(r"(?:peak[_\s-]?)?(?:gpu[_\s-]?)?memory(?:_allocated|_reserved)?(?:_mb)?['\"]?(?:\s*[:=]\s*)(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mb|mib|gb|gib)?", re.IGNORECASE),
        re.compile(r"(?:max|peak).{0,24}(?:memory|mem).{0,12}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mb|mib|gb|gib)", re.IGNORECASE),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse training logs into tabular metrics.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more training log files.")
    parser.add_argument("--output", required=True, help="Output path.")
    parser.add_argument("--format", choices=("csv", "jsonl", "json"), default="csv", help="Output format.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def to_float(value: str, unit: Optional[str] = None) -> float:
    result = float(value)
    if unit and unit.lower() in {"gb", "gib"}:
        result *= 1024.0
    return result


def extract_metric(line: str, metric: str) -> Optional[float]:
    for pattern in PATTERNS[metric]:
        match = pattern.search(line)
        if match:
            value = match.group("value")
            unit = match.groupdict().get("unit")
            if metric == "step":
                return float(int(value))
            return to_float(value, unit)
    return None


def parse_log(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    latest_by_step: Dict[int, Dict[str, Any]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            extracted: Dict[str, Any] = {"source": str(path), "line_no": line_no}
            for metric in PATTERNS:
                value = extract_metric(line, metric)
                if value is not None:
                    if metric == "step":
                        extracted[metric] = int(value)
                    else:
                        extracted[metric] = value

            metric_keys = set(extracted) - {"source", "line_no"}
            if not metric_keys:
                continue

            step = extracted.get("step")
            if isinstance(step, int):
                row = latest_by_step.setdefault(step, {"source": str(path), "step": step})
                row.update({k: v for k, v in extracted.items() if k not in {"line_no"}})
                row["last_line_no"] = line_no
            else:
                rows.append(extracted)

    rows.extend(latest_by_step[step] for step in sorted(latest_by_step))
    return rows


def write_rows(rows: List[Dict[str, Any]], output: Path, output_format: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "step", "loss", "learning_rate", "step_time_s", "tokens_per_s", "peak_memory_mb", "line_no", "last_line_no"]

    if output_format == "csv":
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    elif output_format == "jsonl":
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with output.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    all_rows: List[Dict[str, Any]] = []
    try:
        for input_name in args.input:
            path = Path(input_name)
            if not path.exists():
                raise FileNotFoundError(f"input log not found: {path}")
            rows = parse_log(path)
            LOGGER.info("Parsed %s rows from %s", len(rows), path)
            all_rows.extend(rows)
        write_rows(all_rows, Path(args.output), args.format)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to summarize logs: %s", exc)
        return 1

    LOGGER.info("Wrote %s rows to %s", len(all_rows), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
