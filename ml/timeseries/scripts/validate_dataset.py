"""Validate manifest rows and CSV signal quality before model training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.timeseries.src.data.loader import load_manifest
from ml.timeseries.src.schema import LABEL_ID_TO_NAME, TimeSeriesShape

LOGGER = logging.getLogger(__name__)
LEAKAGE_RISK_COLUMNS = {
    "sample_id",
    "json_path",
    "image_path",
    "timeseries_path",
    "label_name",
    "json_image_path",
    "json_timeseries_path",
    "defect_details",
    "defect_nums",
    "max_discharge_value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality_report.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/data_quality_summary.json"))
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit with a non-zero status if any invalid row is found.",
    )
    return parser.parse_args()


def path_exists(value: Any) -> bool:
    if pd.isna(value):
        return False
    return Path(str(value)).exists()


def validate_signal(path: str | Path, expected_shape: tuple[int, int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_rows": 0,
        "n_cols": 0,
        "num_nan": 0,
        "num_inf": 0,
        "min": np.nan,
        "max": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "rms": np.nan,
        "is_constant": False,
        "is_valid_shape": False,
        "error_message": "",
    }
    try:
        array = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D CSV array, got ndim={array.ndim}")
        result["n_rows"], result["n_cols"] = int(array.shape[0]), int(array.shape[1])
        result["is_valid_shape"] = tuple(array.shape) == expected_shape
        result["num_nan"] = int(np.isnan(array).sum())
        result["num_inf"] = int(np.isinf(array).sum())
        finite = array[np.isfinite(array)]
        if finite.size:
            result["min"] = float(finite.min())
            result["max"] = float(finite.max())
            result["mean"] = float(finite.mean())
            result["std"] = float(finite.std())
            result["rms"] = float(np.sqrt(np.mean(finite**2)))
            result["is_constant"] = bool(float(finite.std()) <= 1e-12)
        if not result["is_valid_shape"]:
            result["error_message"] = f"shape={tuple(array.shape)} expected={expected_shape}"
        if result["num_nan"] or result["num_inf"]:
            result["error_message"] = "; ".join(
                part
                for part in [
                    str(result["error_message"]),
                    f"nan={result['num_nan']} inf={result['num_inf']}",
                ]
                if part
            )
    except Exception as exc:  # noqa: BLE001 - report every bad sample, do not stop at first failure.
        result["error_message"] = str(exc)
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest).reset_index(drop=True)
    expected_shape = TimeSeriesShape().channel_first
    duplicated_sample_ids = manifest["sample_id"].duplicated(keep=False)
    report_rows = []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="validate csv"):
        row_dict = row._asdict()
        label_id = int(row_dict.get("label_id", -1))
        label_name = str(row_dict.get("label_name", ""))
        expected_label_name = LABEL_ID_TO_NAME.get(label_id, "")
        timeseries_path = row_dict.get("timeseries_path", "")

        signal_report = validate_signal(timeseries_path, expected_shape) if path_exists(timeseries_path) else {
            "n_rows": 0,
            "n_cols": 0,
            "num_nan": 0,
            "num_inf": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "rms": np.nan,
            "is_constant": False,
            "is_valid_shape": False,
            "error_message": "missing timeseries_path",
        }
        index = int(row_dict.get("Index", len(report_rows)))
        sample_id = row_dict.get("sample_id", "")
        problems = []
        if label_id not in LABEL_ID_TO_NAME:
            problems.append("invalid_label_id")
        if expected_label_name and label_name and label_name != expected_label_name:
            problems.append("label_name_mismatch")
        for path_column in ("timeseries_path", "json_path", "image_path"):
            if path_column in manifest.columns and not path_exists(row_dict.get(path_column, "")):
                problems.append(f"missing_{path_column}")
        if bool(duplicated_sample_ids.iloc[index]):
            problems.append("duplicate_sample_id")
        if not signal_report["is_valid_shape"]:
            problems.append("invalid_shape")
        if signal_report["num_nan"] or signal_report["num_inf"]:
            problems.append("non_finite_signal")
        if signal_report["is_constant"]:
            problems.append("constant_signal")
        if signal_report["error_message"] and "invalid_shape" not in problems:
            problems.append("csv_read_error")

        report_rows.append(
            {
                "sample_id": sample_id,
                "timeseries_path": timeseries_path,
                "json_path": row_dict.get("json_path", ""),
                "image_path": row_dict.get("image_path", ""),
                "label_id": label_id,
                "label_name": label_name,
                **signal_report,
                "is_valid": not problems,
                "problems": "|".join(problems),
            }
        )

    report = pd.DataFrame(report_rows)
    report.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = {
        "manifest": str(args.manifest),
        "rows": int(len(manifest)),
        "valid_rows": int(report["is_valid"].sum()),
        "invalid_rows": int((~report["is_valid"]).sum()),
        "label_distribution": {
            str(label): int(count)
            for label, count in manifest["label_id"].value_counts().sort_index().items()
        },
        "problem_counts": report.loc[report["problems"] != "", "problems"]
        .str.get_dummies(sep="|")
        .sum()
        .astype(int)
        .sort_values(ascending=False)
        .to_dict(),
        "leakage_risk_columns_present": sorted(LEAKAGE_RISK_COLUMNS.intersection(manifest.columns)),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Data quality report saved: %s", args.output)
    LOGGER.info("Data quality summary saved: %s", args.summary)
    if args.fail_on_invalid and summary["invalid_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
