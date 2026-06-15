from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from ml.timeseries.src.features.timeseries_summary import format_summary_for_csv, summarize_signal
from service.backend.app.application.contracts import TimeSeriesToolInput
from service.backend.app.models.checkpoint_adapters import CheckpointTimeSeriesInferenceAdapter
from service.backend.app.models.model_artifacts import ModelArtifactRegistry


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


def export_ts_context(
    manifest_path: Path,
    output_path: Path,
    sample_size: int | None,
    model_artifact_root: Path | None = None,
) -> int:
    rows = _read_manifest(manifest_path)
    selected_rows = _balanced_sample(rows, sample_size)
    adapter = _time_series_adapter(model_artifact_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(_context_row(manifest_path, row, adapter))
    return len(selected_rows)


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _balanced_sample(rows: list[dict[str, str]], sample_size: int | None) -> list[dict[str, str]]:
    if sample_size is None or sample_size >= len(rows):
        return rows
    buckets: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(int(row.get("label_id", "0")), []).append(row)
    selected: list[dict[str, str]] = []
    cursor = 0
    label_ids = sorted(buckets)
    while len(selected) < sample_size:
        added = False
        for label_id in label_ids:
            bucket = buckets[label_id]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                added = True
                if len(selected) >= sample_size:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _context_row(
    manifest_path: Path,
    row: dict[str, str],
    adapter: CheckpointTimeSeriesInferenceAdapter | None,
) -> dict[str, str]:
    timeseries_path = _resolve_manifest_file_path(manifest_path, row["timeseries_path"])
    signal = np.loadtxt(timeseries_path, delimiter=",", dtype=np.float32)
    summary = format_summary_for_csv(summarize_signal(signal))
    prediction = _predict_timeseries(adapter, timeseries_path)
    return {
        "sample_id": row["sample_id"],
        "ts_model_name": prediction.get("model_name", "feature_summary_untrained"),
        "ts_pred_label_id": prediction.get("label_id", ""),
        "ts_confidence": prediction.get("confidence", ""),
        "ts_prob_0": prediction.get("prob_0", ""),
        "ts_prob_1": prediction.get("prob_1", ""),
        "ts_prob_2": prediction.get("prob_2", ""),
        "ts_prob_3": prediction.get("prob_3", ""),
        "ts_prob_4": prediction.get("prob_4", ""),
        "rms": summary["rms"],
        "std": summary["std"],
        "abs_p99": summary["abs_p99"],
        "pulse_rate": summary["pulse_rate"],
        "spectral_energy": summary["spectral_energy"],
    }


def _time_series_adapter(model_artifact_root: Path | None) -> CheckpointTimeSeriesInferenceAdapter | None:
    if model_artifact_root is None:
        return None
    record = ModelArtifactRegistry(model_artifact_root).get("time_series")
    if not record.ready:
        raise RuntimeError(f"time_series model artifact is not ready: {record.error}")
    return CheckpointTimeSeriesInferenceAdapter(record)


def _predict_timeseries(adapter: CheckpointTimeSeriesInferenceAdapter | None, path: Path) -> dict[str, object]:
    if adapter is None:
        return {}
    result = adapter.run(TimeSeriesToolInput(csv_path=path, csv_sha256="context-export"))
    probabilities = result.probabilities or {}
    return {
        "model_name": result.model_name,
        "label_id": result.label_id,
        "confidence": result.confidence,
        **{f"prob_{label_id}": probabilities.get(str(label_id), "") for label_id in range(5)},
    }


def _resolve_manifest_file_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    normalized = raw_path.replace("\\", "/")
    relative_without_train = normalized.removeprefix("Train/")
    manifest_dir = manifest_path.parent
    candidates = (
        manifest_dir / normalized,
        manifest_dir / relative_without_train,
        manifest_dir.parent / normalized,
        manifest_dir.parent / relative_without_train,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model-artifact-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = export_ts_context(args.manifest, args.output, args.sample_size, args.model_artifact_root)
    print(json.dumps({"rows_written": rows_written, "output": str(args.output)}))


if __name__ == "__main__":
    main()
