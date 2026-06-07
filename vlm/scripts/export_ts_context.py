from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ml.src.features.timeseries_summary import format_summary_for_csv, summarize_signal


FIELDNAMES: tuple[str, ...] = (
    "sample_id",
    "ts_model_name",
    "ts_pred_label_id",
    "ts_confidence",
    "ts_prob_0",
    "ts_prob_1",
    "ts_prob_2",
    "ts_prob_3",
    "ts_prob_4",
    "rms",
    "std",
    "abs_p99",
    "pulse_rate",
    "spectral_energy",
)


def export_ts_context(manifest_path: Path, output_path: Path, sample_size: int | None) -> int:
    rows = _read_manifest(manifest_path)
    selected_rows = rows[:sample_size] if sample_size is not None else rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(_context_row(row))
    return len(selected_rows)


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _context_row(row: dict[str, str]) -> dict[str, str]:
    signal = np.loadtxt(row["timeseries_path"], delimiter=",", dtype=np.float32)
    summary = format_summary_for_csv(summarize_signal(signal))
    return {
        "sample_id": row["sample_id"],
        "ts_model_name": "feature_summary_untrained",
        "ts_pred_label_id": "",
        "ts_confidence": "",
        "ts_prob_0": "",
        "ts_prob_1": "",
        "ts_prob_2": "",
        "ts_prob_3": "",
        "ts_prob_4": "",
        "rms": summary["rms"],
        "std": summary["std"],
        "abs_p99": summary["abs_p99"],
        "pulse_rate": summary["pulse_rate"],
        "spectral_energy": summary["spectral_energy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = export_ts_context(args.manifest, args.output, args.sample_size)
    print(json.dumps({"rows_written": rows_written, "output": str(args.output)}))


if __name__ == "__main__":
    main()
