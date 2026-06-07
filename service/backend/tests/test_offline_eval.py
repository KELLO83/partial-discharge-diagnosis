from __future__ import annotations

import csv
import json
from pathlib import Path

from service.backend.app.offline import run_offline_mock_evaluation


def test_offline_mock_evaluation_writes_jsonl_and_summary(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "predictions.jsonl"
    _write_manifest(manifest_path)

    summary = run_offline_mock_evaluation(manifest_path, output_path, sample_size=2)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary.rows == 2
    assert summary.completed == 2
    assert summary.needs_review == 0
    assert summary.ts_vlm_agreement_rate == 1.0
    assert rows[0]["status"] == "completed"


def _write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label_id"])
        writer.writeheader()
        writer.writerow({"sample_id": "sample-1", "label_id": "0"})
        writer.writerow({"sample_id": "sample-2", "label_id": "3"})
