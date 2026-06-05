from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

import numpy as np

export_ts_context: Any = getattr(importlib.import_module("vlm.scripts.export_ts_context"), "export_ts_context")


def test_export_ts_context_writes_feature_summary(tmp_path: Path) -> None:
    signal_path = tmp_path / "signal.csv"
    np.savetxt(signal_path, np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float32), delimiter=",")
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "ts_context.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "timeseries_path"])
        writer.writeheader()
        writer.writerow({"sample_id": "sample-1", "timeseries_path": str(signal_path)})

    rows_written = export_ts_context(manifest_path, output_path, sample_size=1)

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))
    assert rows_written == 1
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["ts_model_name"] == "feature_summary_untrained"
    assert float(rows[0]["rms"]) > 0
    assert float(rows[0]["spectral_energy"]) > 0
