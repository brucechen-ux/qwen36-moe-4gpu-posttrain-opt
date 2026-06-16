#!/usr/bin/env python3
"""Analyze RL pipeline component timings from text logs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOGGER = logging.getLogger("analyze_rl_step_time")

COMPONENT_ALIASES = {
    "rollout": ("rollout", "generate", "generation"),
    "reward": ("reward", "scoring", "score"),
    "ref_logprob": ("ref_logprob", "ref logprob", "reference_logprob", "reference logprob"),
    "actor_update": ("actor_update", "actor update", "policy_update", "train_actor"),
}

DURATION_RE = re.compile(
    r"(?:(?:time|duration|elapsed|latency|cost)(?:_s|_ms)?\s*[:=]\s*)?"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|millisecond(?:s)?|s|sec|second(?:s)?|m|min|minute(?:s)?)",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"(?:^|[\s,{])step(?:\s*[:=]\s*|\s+)(?P<value>\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RL rollout/reward/ref/actor timing proportions.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more RL pipeline log files.")
    parser.add_argument("--output", required=True, help="Output path.")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="Output format.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def duration_to_seconds(value: str, unit: str) -> float:
    seconds = float(value)
    normalized = unit.lower()
    if normalized.startswith("ms") or normalized.startswith("millisecond"):
        return seconds / 1000.0
    if normalized in {"m", "min"} or normalized.startswith("minute"):
        return seconds * 60.0
    return seconds


def detect_component(line: str) -> Optional[str]:
    lowered = line.lower()
    for component, aliases in COMPONENT_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return component
    return None


def extract_duration(line: str) -> Optional[float]:
    matches = list(DURATION_RE.finditer(line))
    if not matches:
        return None
    match = matches[-1]
    return duration_to_seconds(match.group("value"), match.group("unit"))


def extract_step(line: str) -> Optional[int]:
    match = STEP_RE.search(line)
    if not match:
        return None
    return int(match.group("value"))


def parse_logs(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                component = detect_component(line)
                if not component:
                    continue
                duration_s = extract_duration(line)
                if duration_s is None:
                    LOGGER.debug("No duration found for component line %s:%s", path, line_no)
                    continue
                events.append(
                    {
                        "source": str(path),
                        "line_no": line_no,
                        "step": extract_step(line),
                        "component": component,
                        "duration_s": duration_s,
                    }
                )
    return events


def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for event in events:
        component = event["component"]
        totals[component] += event["duration_s"]
        counts[component] += 1

    total_duration = sum(totals.values())
    by_component = []
    for component in COMPONENT_ALIASES:
        total_s = totals.get(component, 0.0)
        count = counts.get(component, 0)
        by_component.append(
            {
                "component": component,
                "count": count,
                "total_s": total_s,
                "mean_s": total_s / count if count else None,
                "percent": (100.0 * total_s / total_duration) if total_duration else 0.0,
            }
        )

    return {
        "events": len(events),
        "total_profiled_time_s": total_duration,
        "by_component": by_component,
    }


def write_output(summary: Dict[str, Any], output: Path, output_format: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "count", "total_s", "mean_s", "percent"])
        writer.writeheader()
        for row in summary["by_component"]:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        paths = [Path(name) for name in args.input]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing input log(s): {', '.join(missing)}")
        events = parse_logs(paths)
        summary = summarize(events)
        write_output(summary, Path(args.output), args.format)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to analyze RL step time: %s", exc)
        return 1

    LOGGER.info("Parsed %s timing events and wrote %s", summary["events"], args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
