from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES: tuple[str, ...] = (
    "mode",
    "model_id",
    "status",
    "json_parse_success_rate",
    "schema_validity_rate",
    "label_accuracy",
    "macro_f1",
    "parse_failures",
)


def summarize_experiments(input_dir: Path, output_path: Path) -> int:
    rows = _collect_rows(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _collect_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metadata_path in sorted(input_dir.glob("*/metadata.json")):
        run_dir = metadata_path.parent
        metrics_path = run_dir / "metrics.json"
        metadata = _read_json(metadata_path)
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        rows.append(
            {
                "mode": str(metadata.get("mode", run_dir.name)),
                "model_id": str(metadata.get("model_id", "")),
                "status": str(metadata.get("status", "")),
                "json_parse_success_rate": _metric(metrics, "json_parse_success_rate"),
                "schema_validity_rate": _metric(metrics, "schema_validity_rate"),
                "label_accuracy": _metric(metrics, "label_accuracy"),
                "macro_f1": _metric(metrics, "macro_f1"),
                "parse_failures": _metric(metrics, "parse_failures"),
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return parsed


def _metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key, "")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = summarize_experiments(args.input, args.output)
    print(json.dumps({"rows_written": rows_written, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
