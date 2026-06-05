from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from vlm.src.schema import FORBIDDEN_PROMPT_FIELDS, ValidationReport

MAX_PROMPT_CHARS = 6000


def validate_jsonl(input_path: Path) -> ValidationReport:
    n_rows = 0
    missing_images = 0
    invalid_targets = 0
    leakage_hits = 0
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        n_rows += 1
        record = json.loads(line)
        for image in record.get("images", []):
            if not Path(image).exists():
                missing_images += 1
        prompt = _prompt_text(record)
        if len(prompt) > MAX_PROMPT_CHARS:
            invalid_targets += 1
        leakage_hits += sum(1 for field in FORBIDDEN_PROMPT_FIELDS if field in prompt)
        try:
            target = json.loads(record["messages"][1]["content"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            invalid_targets += 1
            continue
        if not _target_valid(target):
            invalid_targets += 1
    valid = missing_images == 0 and invalid_targets == 0 and leakage_hits == 0
    return ValidationReport(
        valid=valid,
        n_rows=n_rows,
        missing_images=missing_images,
        invalid_targets=invalid_targets,
        leakage_hits=leakage_hits,
    )


def _prompt_text(record: dict[str, object]) -> str:
    messages = record["messages"]
    if not isinstance(messages, list):
        return ""
    user_message = messages[0]
    if not isinstance(user_message, dict):
        return ""
    content = user_message.get("content", [])
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            value = block.get("text", "")
            if isinstance(value, str):
                texts.append(value)
    return "\n".join(texts)


def _target_valid(target: object) -> bool:
    if not isinstance(target, dict):
        return False
    required = {"label_id", "diagnosis", "risk_level", "reason", "recommended_action"}
    if not required.issubset(target):
        return False
    label_id = target.get("label_id")
    return isinstance(label_id, int) and 0 <= label_id <= 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_jsonl(args.input)
    payload = {
        "valid": report.valid,
        "n_rows": report.n_rows,
        "missing_images": report.missing_images,
        "invalid_targets": report.invalid_targets,
        "leakage_hits": report.leakage_hits,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
