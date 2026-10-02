#!/usr/bin/env python3
"""Validate post-template token lengths using the same tokenizer as vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if "system_prompt" in row:
        messages.append({"role": "system", "content": row["system_prompt"]})
    messages.append({"role": "user", "content": row["prompt"]})
    return messages


def token_count(tokenized: Any) -> int:
    """Normalize Transformers list, tensor, or BatchEncoding outputs to IDs."""
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    while tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return len(tokenized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "benchmark_matrix.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "thai_workloads.jsonl")
    parser.add_argument("--model", help="Overrides config.model.id")
    parser.add_argument("--revision", help="Overrides config.model.revision; must be an immutable commit SHA")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tokenizer-report.json")
    args = parser.parse_args()
    config, rows = read_json(args.config), read_jsonl(args.dataset)
    model = args.model or config["model"]["id"]
    revision = args.revision or config["model"]["revision"]
    if revision == "PIN_BEFORE_GPU_RUN":
        raise SystemExit("Set an immutable model revision before generating a tokenizer report.")
    placeholder_ids = [row["id"] for row in rows if row.get("source") == "placeholder"]
    if placeholder_ids:
        raise SystemExit("Replace placeholder dataset rows first: " + ", ".join(placeholder_ids))
    prefixes = {row["system_prompt"] for row in rows if row["workload"] == "shared_prefix"}
    if len(prefixes) != 1:
        raise SystemExit("All shared_prefix rows must have a byte-identical system_prompt.")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("transformers is required. Run inside the vLLM container or install it in a separate environment.") from error
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        text = "\n".join(message["content"] for message in render_messages(row))
        normalized = unicodedata.normalize("NFC", text)
        raw_token_ids = tokenizer(normalized, add_special_tokens=False)["input_ids"]
        rendered_token_ids = tokenizer.apply_chat_template(
            render_messages(row), tokenize=True, add_generation_prompt=True
        )
        actual = token_count(rendered_token_ids)
        target = row["target_input_tokens"]
        report_rows.append({
            "id": row["id"],
            "workload": row["workload"],
            "target_input_tokens": target,
            "actual_post_template_tokens": actual,
            "token_delta": actual - target,
            "raw_tokens": len(raw_token_ids),
            "unicode_code_points": len(normalized),
            "tokens_per_code_point": len(raw_token_ids) / len(normalized) if normalized else None,
            "system_prompt_sha256": hashlib.sha256(row.get("system_prompt", "").encode("utf-8")).hexdigest() if "system_prompt" in row else None,
        })
    report = {
        "model": model,
        "revision": revision,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "rows": report_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
