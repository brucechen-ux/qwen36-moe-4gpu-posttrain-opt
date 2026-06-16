#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible chat completions endpoint from jsonl prompts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOGGER = logging.getLogger("bench_serving")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send jsonl prompts to an OpenAI-compatible endpoint and record latency metrics."
    )
    parser.add_argument("--input", required=True, help="Input jsonl file with prompts.")
    parser.add_argument("--output", required=True, help="Output jsonl result file.")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument("--model", required=True, help="Model name served by the endpoint.")
    parser.add_argument("--prompt-field", default="prompt", help="Json field containing the prompt.")
    parser.add_argument("--system-field", default=None, help="Optional json field containing a system message.")
    parser.add_argument("--id-field", default=None, help="Optional json field used as request id.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum prompts to send. 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming. TTFT will equal total latency.")
    parser.add_argument(
        "--estimate-output-tokens",
        choices=("none", "whitespace", "chars"),
        default="chars",
        help="Fallback token estimator if response usage is missing.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, item


def build_messages(item: Dict[str, Any], prompt_field: str, system_field: Optional[str]) -> List[Dict[str, str]]:
    if prompt_field not in item:
        raise KeyError(f"missing prompt field '{prompt_field}'")
    prompt = item[prompt_field]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"field '{prompt_field}' must be a non-empty string")

    messages: List[Dict[str, str]] = []
    if system_field and item.get(system_field):
        system = item[system_field]
        if not isinstance(system, str):
            raise ValueError(f"field '{system_field}' must be a string when present")
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def make_payload(args: argparse.Namespace, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": not args.no_stream,
    }
    if not args.no_stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def estimate_tokens(text: str, method: str) -> Optional[int]:
    if method == "none":
        return None
    if method == "whitespace":
        return max(1, len(text.split())) if text else 0
    if method == "chars":
        return max(1, math.ceil(len(text) / 4)) if text else 0
    raise ValueError(f"unknown estimation method: {method}")


def request_once(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: float,
    estimate_method: str,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_at: Optional[float] = None
    text_parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if payload.get("stream"):
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        LOGGER.debug("Skipping non-JSON stream chunk: %s", data)
                        continue
                    usage = chunk.get("usage") or usage
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            text_parts.append(content)
            else:
                response_body = response.read().decode("utf-8")
                data = json.loads(response_body)
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    text_parts.append(message.get("content") or "")
                usage = data.get("usage")
                first_token_at = time.perf_counter()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc

    end = time.perf_counter()
    generated_text = "".join(text_parts)
    completion_tokens = None
    prompt_tokens = None
    total_tokens = None
    if usage:
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
    if completion_tokens is None:
        completion_tokens = estimate_tokens(generated_text, estimate_method)

    total_latency = end - start
    ttft = (first_token_at - start) if first_token_at is not None else total_latency
    decode_latency = max(total_latency - ttft, 1e-9)
    output_tokens_per_s = None
    if completion_tokens is not None:
        output_tokens_per_s = completion_tokens / decode_latency

    return {
        "success": True,
        "ttft_s": ttft,
        "total_latency_s": total_latency,
        "decode_latency_s": decode_latency,
        "output_tokens_per_s": output_tokens_per_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "response_text": generated_text,
        "usage": usage,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    successes = [row for row in results if row.get("success")]
    success_count = len(successes)

    def mean_field(name: str) -> Optional[float]:
        values = [row[name] for row in successes if isinstance(row.get(name), (int, float))]
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "total_requests": total,
        "successes": success_count,
        "failures": total - success_count,
        "success_rate": success_count / total if total else 0.0,
        "mean_ttft_s": mean_field("ttft_s"),
        "mean_total_latency_s": mean_field("total_latency_s"),
        "mean_output_tokens_per_s": mean_field("output_tokens_per_s"),
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        LOGGER.error("Input file does not exist: %s", input_path)
        return 2

    results: List[Dict[str, Any]] = []
    LOGGER.info("Benchmarking endpoint=%s model=%s input=%s", args.endpoint, args.model, input_path)

    try:
        with output_path.open("w", encoding="utf-8") as out:
            for idx, (line_no, item) in enumerate(iter_jsonl(input_path), start=1):
                if args.limit and idx > args.limit:
                    break

                request_id = item.get(args.id_field) if args.id_field else idx
                row: Dict[str, Any] = {"request_id": request_id, "line_no": line_no}

                try:
                    messages = build_messages(item, args.prompt_field, args.system_field)
                    payload = make_payload(args, messages)
                    metrics = request_once(
                        args.endpoint,
                        payload,
                        timeout=args.timeout,
                        estimate_method=args.estimate_output_tokens,
                    )
                    row.update(metrics)
                    LOGGER.info(
                        "request_id=%s success ttft=%.4fs latency=%.4fs tok/s=%s",
                        request_id,
                        row["ttft_s"],
                        row["total_latency_s"],
                        f"{row['output_tokens_per_s']:.2f}" if row.get("output_tokens_per_s") else "n/a",
                    )
                except Exception as exc:  # noqa: BLE001 - benchmark should continue per prompt.
                    row.update({"success": False, "error": str(exc)})
                    LOGGER.warning("request_id=%s failed: %s", request_id, exc)

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                results.append(row)
                if args.sleep > 0:
                    time.sleep(args.sleep)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Benchmark aborted: %s", exc)
        return 1

    summary = summarize(results)
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Wrote jsonl results to %s", output_path)
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
