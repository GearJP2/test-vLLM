#!/usr/bin/env python3
"""Emit a deterministic, synthetic Thai workload dataset tuned to one tokenizer.

Run inside the vLLM container and redirect stdout to a host-mounted JSONL file.
The text is authored synthetic content for serving tests; it is not a quality
benchmark or a substitute for an organization's production corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "benchmark_matrix.json"

FILLER = (
    "หน่วยงานบันทึกข้อเท็จจริง ตรวจสอบแหล่งข้อมูล ระบุผู้รับผิดชอบ "
    "และติดตามผลตามระยะเวลาที่กำหนดอย่างเป็นระบบ"
)
SHORT_BASE = {
    100: "อธิบายแนวทางปกป้องข้อมูลส่วนบุคคลสำหรับพนักงานใหม่เป็นภาษาไทยแบบกระชับ พร้อมยกตัวอย่างที่ทำได้จริง",
    300: "จัดทำคำแนะนำสำหรับพนักงานใหม่เกี่ยวกับการตั้งรหัสผ่าน การยืนยันตัวตนหลายปัจจัย การระวังอีเมลหลอกลวง และขั้นตอนแจ้งเหตุผิดปกติ โดยสรุปเป็นข้อที่นำไปใช้ได้ทันที",
}
LONG_BASE = (
    "จากเอกสารการดำเนินงานต่อไปนี้ จงสรุปประเด็นสำคัญ ความเสี่ยง "
    "ผู้รับผิดชอบ และข้อเสนอแนะ โดยอ้างอิงเฉพาะข้อมูลในเอกสาร\n\nเอกสาร:\n"
)
PREFIX_BASE = (
    "คุณเป็นเจ้าหน้าที่รับเรื่องร้องเรียนของหน่วยงานภาครัฐ ตอบเป็นภาษาไทยสุภาพ "
    "สรุปตามข้อมูลที่ได้รับเท่านั้น ระบุขั้นตอน ผู้รับผิดชอบ ระยะเวลา และข้อมูลที่ต้องขอเพิ่ม "
    "หากข้อมูลไม่พอให้บอกอย่างชัดเจนว่าต้องตรวจสอบอะไรต่อไป "
)
PREFIX_QUERIES = [
    "สรุปหน้าที่ของเจ้าหน้าที่รับเรื่องร้องเรียนเป็นสามข้อ พร้อมระบุลำดับการทำงาน",
    "ถ้าข้อมูลในคำร้องไม่ครบ เจ้าหน้าที่ควรดำเนินการอย่างไร และควรแจ้งผู้ร้องเรื่องใดบ้าง",
]


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def fill(tokenizer: Any, make_messages: Any, base: str, target: int) -> str:
    """Extend text until it reaches the nearest usable token count at/above target."""
    text = base
    while token_count(tokenizer, make_messages(text)) < target:
        text += " " + FILLER
    # Trim whole filler repetitions where possible while retaining target length.
    suffix = " " + FILLER
    while text.endswith(suffix) and token_count(tokenizer, make_messages(text[: -len(suffix)])) >= target:
        text = text[: -len(suffix)]
    return text


def row(identifier: str, workload: str, prompt: str, target_input: int, output: int, **extra: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "workload": workload,
        "prompt": prompt,
        "target_input_tokens": target_input,
        "target_output_tokens": output,
        "source": "authored_synthetic",
        "license": "CC0-1.0",
        **extra,
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model, revision = config["model"]["id"], config["model"]["revision"]
    print("Loading pinned tokenizer from the local container cache...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    print("Building short workloads...", file=sys.stderr, flush=True)
    for target, base in SHORT_BASE.items():
        prompt = fill(tokenizer, lambda value: [{"role": "user", "content": value}], base, target)
        print(json.dumps(row(f"short-{target}", "short", prompt, target, 200), ensure_ascii=False), flush=True)
    print("Building long workloads; 8K token calibration can take a short while...", file=sys.stderr, flush=True)
    for target in (2000, 8000):
        prompt = fill(tokenizer, lambda value: [{"role": "user", "content": value}], LONG_BASE, target)
        print(json.dumps(row(f"long-{target}", "long_context", prompt, target, 500), ensure_ascii=False), flush=True)
    # Keep exactly the same system prefix for each shared-prefix request.
    system_prompt = fill(
        tokenizer,
        lambda value: [{"role": "system", "content": value}, {"role": "user", "content": PREFIX_QUERIES[0]}],
        PREFIX_BASE,
        2100,
    )
    for index, query in enumerate(PREFIX_QUERIES, start=1):
        target = token_count(tokenizer, [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}])
        print(json.dumps(row(f"prefix-{index:03d}", "shared_prefix", query, target, 200, system_prompt=system_prompt), ensure_ascii=False), flush=True)
    print("Thai workload generation completed.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
