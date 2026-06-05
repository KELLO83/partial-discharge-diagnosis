from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any

summarize_experiments: Any = getattr(importlib.import_module("vlm.scripts.summarize_experiments"), "summarize_experiments")


def test_summarize_experiments_writes_metrics_csv(tmp_path: Path) -> None:
    input_dir = tmp_path / "experiments"
    output_path = tmp_path / "summary.csv"
    run_dir = input_dir / "image_metadata"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"mode": "image_metadata", "model_id": "Qwen/Qwen3-VL-2B-Instruct", "status": "dry_run"}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"json_parse_success_rate": 1.0, "schema_validity_rate": 1.0, "label_accuracy": 1.0}),
        encoding="utf-8",
    )

    rows_written = summarize_experiments(input_dir, output_path)

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))
    assert rows_written == 1
    assert rows[0]["mode"] == "image_metadata"
    assert rows[0]["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert rows[0]["label_accuracy"] == "1.0"
