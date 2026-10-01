#!/usr/bin/env python3
"""Validate the local benchmark scaffold before a GPU run.

This script deliberately uses only Python's standard library. Token counts must
be validated later with the pinned model tokenizer after its revision is chosen.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "benchmark_matrix.json"
DATASET = ROOT / "data" / "thai_workloads.jsonl"
REQUIRED_VARIANTS = {"A", "B", "C", "D", "E"}
REQUIRED_WORKLOADS = {"short", "long_context", "shared_prefix"}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path.relative_to(ROOT)}: {error}")
    if not isinstance(value, dict):
        fail(f"{path.relative_to(ROOT)} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read {path.relative_to(ROOT)}: {error}")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid JSON at {path.relative_to(ROOT)}:{line_number}: {error}")
        if not isinstance(row, dict):
            fail(f"row {line_number} must be an object")
        rows.append(row)
    return rows


def main() -> None:
    config = read_json(CONFIG)
    dataset = read_jsonl(DATASET)

    allow_placeholders = "--allow-placeholders" in sys.argv[1:]
    unknown_args = set(sys.argv[1:]) - {"--allow-placeholders"}
    if unknown_args:
        fail(f"unknown argument(s): {', '.join(sorted(unknown_args))}")

    if config.get("model", {}).get("revision") == "PIN_BEFORE_GPU_RUN" and not allow_placeholders:
        fail("pin model.revision to an immutable Hugging Face revision before GPU testing")

    variants = config.get("variants", {})
    if set(variants) != REQUIRED_VARIANTS:
        fail("variants must be exactly A, B, C, D, and E")

    if set(config.get("protocol", {}).get("concurrency", [])) != {1, 4, 8, 16, 32}:
        fail("protocol.concurrency must contain 1, 4, 8, 16, and 32")

    identifiers = [row.get("id") for row in dataset]
    if len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
        fail("dataset IDs must be present and unique")

    workloads = {row.get("workload") for row in dataset}
    if workloads != REQUIRED_WORKLOADS:
        fail("dataset must contain short, long_context, and shared_prefix workloads")

    for row in dataset:
        for key in ("prompt", "target_input_tokens", "target_output_tokens", "source", "license"):
            if key not in row:
                fail(f"dataset row {row['id']} is missing {key}")
        if not isinstance(row["target_input_tokens"], int) or not isinstance(row["target_output_tokens"], int):
            fail(f"dataset row {row['id']} token targets must be integers")

    prefix_rows = [row for row in dataset if row["workload"] == "shared_prefix"]
    if any("system_prompt" not in row for row in prefix_rows):
        fail("each shared_prefix row must include system_prompt")

    if not allow_placeholders:
        placeholders = [row["id"] for row in dataset if row.get("source") == "placeholder"]
        if placeholders:
            fail("replace placeholder dataset rows before GPU testing: " + ", ".join(placeholders))
        prompts = {row["system_prompt"] for row in prefix_rows}
        if len(prompts) != 1:
            fail("shared_prefix rows must use byte-identical system_prompt values")

    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    print("Scaffold structure is valid.")
    print(f"Dataset SHA-256: {digest}")
    if allow_placeholders:
        print("Scaffold mode: dataset placeholders are allowed only for local structure checks.")
    else:
        print("Preflight passed for a GPU benchmark run.")


if __name__ == "__main__":
    main()
