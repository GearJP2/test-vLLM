#!/usr/bin/env python3
"""Run a reproducible closed-loop streaming benchmark against one vLLM endpoint.

The runner uses only the Python standard library. It writes raw request records,
server metric snapshots, and an aggregate summary. It does not start a server.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def post_stream(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    output_events = 0
    usage: dict[str, Any] | None = None
    error: str | None = None
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        now = time.perf_counter()
                        first_token_at = first_token_at or now
                        last_token_at = now
                        output_events += 1
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        error = str(exc)
    ended = time.perf_counter()
    output_tokens = usage.get("completion_tokens") if usage else None
    ttft = (first_token_at - started) if first_token_at else None
    e2e = ended - started
    tpot = None
    if first_token_at and last_token_at and isinstance(output_tokens, int) and output_tokens > 1:
        tpot = (last_token_at - first_token_at) / (output_tokens - 1)
    return {
        "started_monotonic": started,
        "ended_monotonic": ended,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "e2e_seconds": e2e,
        "output_events": output_events,
        "usage": usage,
        "error": error,
    }


def messages(row: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if "system_prompt" in row:
        result.append({"role": "system", "content": row["system_prompt"]})
    result.append({"role": "user", "content": row["prompt"]})
    return result


def run_one(index: int, row: dict[str, Any], api_url: str, model: str, generation: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages(row),
        "temperature": generation["temperature"],
        "max_tokens": row.get("target_output_tokens", generation["max_tokens"]),
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": f"thai-benchmark-{index}-{uuid.uuid4().hex}",
    }
    result = post_stream(api_url, payload, generation["timeout_seconds"])
    result.update({"request_index": index, "dataset_id": row["id"], "workload": row["workload"]})
    return result


def run_batch(rows: list[dict[str, Any]], count: int, concurrency: int, api_url: str, model: str, generation: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_one, index, rows[index % len(rows)], api_url, model, generation) for index in range(count)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results, time.perf_counter() - started


def summary(records: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    succeeded = [record for record in records if not record["error"]]
    output_tokens = sum((record.get("usage") or {}).get("completion_tokens", 0) for record in succeeded)
    def field(name: str) -> dict[str, float | None]:
        values = [record[name] for record in succeeded if record[name] is not None]
        return {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "mean": statistics.fmean(values) if values else None}
    return {
        "attempted_requests": len(records),
        "successful_requests": len(succeeded),
        "failed_requests": len(records) - len(succeeded),
        "wall_seconds": wall_seconds,
        "request_throughput_per_second": len(succeeded) / wall_seconds if wall_seconds else None,
        "output_token_throughput_per_second": output_tokens / wall_seconds if wall_seconds else None,
        "ttft_seconds": field("ttft_seconds"),
        "tpot_seconds": field("tpot_seconds"),
        "e2e_seconds": field("e2e_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "benchmark_matrix.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "thai_workloads.jsonl")
    parser.add_argument("--workload", choices=("short", "long_context", "shared_prefix"), required=True)
    parser.add_argument("--variant", choices=("A", "B", "C", "D", "E"), required=True, help="Feature state configured on the server")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, help="Override measured_requests for a smoke test")
    parser.add_argument("--warmup-requests", type=int, help="Override warmup_requests")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = load_json(args.config)
    rows = [row for row in load_jsonl(args.dataset) if row["workload"] == args.workload]
    if not rows:
        raise SystemExit(f"No rows found for workload {args.workload}")
    protocol, server, generation = config["protocol"], config["server"], config["generation"]
    measured = args.requests or protocol["measured_requests"]
    warmups = args.warmup_requests if args.warmup_requests is not None else protocol["warmup_requests"]
    if args.concurrency < 1 or measured < 1 or warmups < 0:
        raise SystemExit("concurrency and requests must be positive; warmup-requests cannot be negative")
    api_url = f"http://{server['host']}:{server['port']}{server['api_path']}"
    model = config["model"].get("served_model_name", config["model"]["id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{args.workload}-c{args.concurrency}"
    if warmups:
        warmup_records, _ = run_batch(rows, warmups, args.concurrency, api_url, model, generation)
        if any(record["error"] for record in warmup_records):
            raise SystemExit("Warm-up failed; inspect server health and endpoint before measuring")
    try:
        before = metrics(f"http://{server['host']}:{server['port']}/metrics", generation["timeout_seconds"])
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Cannot scrape /metrics: {exc}") from exc
    records, wall_seconds = run_batch(rows, measured, args.concurrency, api_url, model, generation)
    after = metrics(f"http://{server['host']}:{server['port']}/metrics", generation["timeout_seconds"])
    (args.output_dir / f"{run_id}.before.prom").write_text(before, encoding="utf-8")
    (args.output_dir / f"{run_id}.after.prom").write_text(after, encoding="utf-8")
    with (args.output_dir / f"{run_id}.requests.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_summary = summary(records, wall_seconds)
    output_summary.update({"run_id": run_id, "variant": args.variant, "workload": args.workload, "concurrency": args.concurrency, "api_url": api_url, "model": model})
    summary_path = args.output_dir / f"{run_id}.summary.json"
    summary_path.write_text(json.dumps(output_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))
    if output_summary["failed_requests"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
