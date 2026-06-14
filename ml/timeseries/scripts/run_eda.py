"""Run EDA before training partial-discharge time-series models.

The manifest-level checks use all rows.  Signal-level EDA reads a stratified
sample by default so it is safe to run before model experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.timeseries.src.data.loader import load_manifest, read_timeseries_csv

LOGGER = logging.getLogger(__name__)

LABEL_NAMES = {
    0: "normal",
    1: "noise",
    2: "surface",
    3: "corona",
    4: "void",
}
SAMPLES_PER_ROW = 7680
POWER_FREQUENCY_HZ = 60
CYCLE_LENGTH = SAMPLES_PER_ROW // POWER_FREQUENCY_HZ
PHASE_BINS = 24
LEAKAGE_COLUMNS = {
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


def configure_matplotlib_font() -> None:
    """Use a Korean-capable font on Windows when available."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class-examples", type=int, default=3)
    parser.add_argument(
        "--full-signal-eda",
        action="store_true",
        help="Read every CSV for signal-level EDA. Default reads a stratified sample only.",
    )
    return parser.parse_args()


def stratified_sample(frame: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    parts = []
    n_classes = int(frame["label_id"].nunique())
    per_class = max(1, sample_size // n_classes)
    for _, part in frame.groupby("label_id", sort=True):
        parts.append(part.sample(min(len(part), per_class), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def safe_value_counts(frame: pd.DataFrame, column: str, top_k: int = 30) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame(columns=[column, "count"])
    counts = frame[column].fillna("<NA>").astype(str).value_counts(dropna=False).head(top_k)
    return counts.rename_axis(column).reset_index(name="count")


def manifest_summary(frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    label_counts = frame["label_id"].value_counts().sort_index().rename_axis("label_id").reset_index(name="count")
    label_counts["label_name_en"] = label_counts["label_id"].map(LABEL_NAMES)
    label_counts.to_csv(output_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")

    for column in ("insulator_type", "equipment_name", "sensor_type", "recording_time_length"):
        safe_value_counts(frame, column).to_csv(output_dir / f"{column}_distribution.csv", index=False, encoding="utf-8-sig")

    missing_paths = {}
    for column in ("timeseries_path", "image_path", "json_path"):
        if column in frame.columns:
            missing_paths[column] = int((~frame[column].map(lambda value: Path(str(value)).exists())).sum())

    duplicated_sample_ids = int(frame["sample_id"].duplicated().sum()) if "sample_id" in frame.columns else 0
    leakage_present = sorted(column for column in LEAKAGE_COLUMNS if column in frame.columns)

    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "label_distribution": {str(row.label_id): int(row.count) for row in label_counts.itertuples()},
        "duplicated_sample_ids": duplicated_sample_ids,
        "missing_paths": missing_paths,
        "leakage_risk_columns_present": leakage_present,
    }


def robust_threshold(flat_abs: np.ndarray) -> float:
    median = float(np.median(flat_abs))
    mad = float(np.median(np.abs(flat_abs - median)))
    scale = max(mad, float(flat_abs.std()), 1e-6)
    return median + 5.0 * scale


def signal_stats(signal: np.ndarray) -> dict[str, float]:
    flat = signal.reshape(-1).astype(np.float32)
    abs_flat = np.abs(flat)
    threshold = robust_threshold(abs_flat)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(flat**2))),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "abs_p95": float(np.percentile(abs_flat, 95)),
        "abs_p99": float(np.percentile(abs_flat, 99)),
        "max_abs": float(abs_flat.max()),
        "pulse_rate": float((abs_flat > threshold).mean()),
        "pulse_count": float((abs_flat > threshold).sum()),
    }


def phase_counts(signal: np.ndarray) -> np.ndarray:
    abs_signal = np.abs(signal.astype(np.float32))
    threshold = robust_threshold(abs_signal.reshape(-1))
    time_index = np.arange(abs_signal.shape[1], dtype=np.int32)
    phase_bin_per_time = ((time_index % CYCLE_LENGTH) * PHASE_BINS // CYCLE_LENGTH).astype(np.int32)
    repeated_phase_bins = np.broadcast_to(phase_bin_per_time, abs_signal.shape).reshape(-1)
    event_mask = abs_signal.reshape(-1) > threshold
    counts = np.bincount(repeated_phase_bins[event_mask], minlength=PHASE_BINS).astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def load_signal_eda(
    frame: pd.DataFrame,
    output_dir: Path,
    signal_summary_name: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, list[np.ndarray]], dict[int, np.ndarray]]:
    stat_rows = []
    phase_by_label: dict[int, list[np.ndarray]] = {label: [] for label in sorted(frame["label_id"].unique())}
    examples_by_label: dict[int, list[np.ndarray]] = {label: [] for label in sorted(frame["label_id"].unique())}
    mean_abs_sum_by_label: dict[int, np.ndarray] = {}
    mean_abs_count_by_label: dict[int, int] = {label: 0 for label in sorted(frame["label_id"].unique())}

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="EDA read csv"):
        label_id = int(getattr(row, "label_id"))
        path = getattr(row, "timeseries_path")
        signal = read_timeseries_csv(path)
        stats = signal_stats(signal)
        stats.update(
            {
                "sample_id": getattr(row, "sample_id"),
                "label_id": label_id,
                "label_name_en": LABEL_NAMES.get(label_id, str(label_id)),
                "csv_shape": f"{signal.shape[0]}x{signal.shape[1]}",
            }
        )
        stat_rows.append(stats)
        phase_by_label[label_id].append(phase_counts(signal))
        mean_abs = np.abs(signal).mean(axis=0)
        if label_id not in mean_abs_sum_by_label:
            mean_abs_sum_by_label[label_id] = np.zeros_like(mean_abs, dtype=np.float64)
        mean_abs_sum_by_label[label_id] += mean_abs
        mean_abs_count_by_label[label_id] += 1
        if len(examples_by_label[label_id]) < 3:
            examples_by_label[label_id].append(signal)

    stats_frame = pd.DataFrame(stat_rows)
    stats_frame.to_csv(output_dir / signal_summary_name, index=False, encoding="utf-8-sig")

    phase_matrix = []
    for label_id in sorted(phase_by_label):
        values = phase_by_label[label_id]
        if values:
            phase_matrix.append(np.stack(values, axis=0).mean(axis=0))
        else:
            phase_matrix.append(np.zeros(PHASE_BINS, dtype=np.float64))
    mean_abs_by_label = {
        label_id: total / max(mean_abs_count_by_label.get(label_id, 0), 1)
        for label_id, total in mean_abs_sum_by_label.items()
    }
    return stats_frame, np.stack(phase_matrix, axis=0), examples_by_label, mean_abs_by_label


def plot_label_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    counts = frame["label_id"].value_counts().sort_index().reset_index()
    counts.columns = ["label_id", "count"]
    counts["label"] = counts["label_id"].map(lambda value: f"{value}:{LABEL_NAMES.get(int(value), value)}")
    plt.figure(figsize=(8, 4))
    plt.bar(counts["label"], counts["count"])
    plt.title("Label distribution")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=160)
    plt.close()


def plot_manifest_metadata(frame: pd.DataFrame, output_dir: Path) -> None:
    columns = [column for column in ("insulator_type", "sensor_type", "equipment_name") if column in frame.columns]
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 4 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, columns):
        counts = safe_value_counts(frame, column, top_k=20)
        axis.barh(counts[column].astype(str), counts["count"])
        axis.set_title(f"{column} distribution")
        axis.set_xlabel("count")
        axis.set_ylabel(column)
        axis.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "metadata_distributions.png", dpi=160)
    plt.close()


def plot_signal_stats(stats_frame: pd.DataFrame, output_dir: Path) -> None:
    if stats_frame.empty:
        return
    metrics = ["rms", "abs_p99", "max_abs", "pulse_rate"]
    long = stats_frame.melt(
        id_vars=["label_id", "label_name_en"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    long["class"] = long["label_id"].astype(str) + ":" + long["label_name_en"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.reshape(-1), metrics):
        part = long[long["metric"] == metric]
        labels = sorted(part["class"].unique())
        values = [part.loc[part["class"] == label, "value"].to_numpy() for label in labels]
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "signal_stats_by_class.png", dpi=160)
    plt.close()


def plot_phase_heatmap(phase_matrix: np.ndarray, output_dir: Path) -> None:
    labels = [f"{label}:{LABEL_NAMES[label]}" for label in sorted(LABEL_NAMES)]
    plt.figure(figsize=(12, 4))
    plt.imshow(phase_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="normalized pulse count")
    plt.xticks(
        np.arange(PHASE_BINS),
        [f"{int(i * 360 / PHASE_BINS)}" for i in range(PHASE_BINS)],
        rotation=45,
    )
    plt.yticks(np.arange(phase_matrix.shape[0]), labels[: phase_matrix.shape[0]])
    plt.title("Phase-bin pulse distribution by class")
    plt.xlabel("phase degree bin")
    plt.ylabel("class")
    plt.tight_layout()
    plt.savefig(output_dir / "phase_pulse_distribution.png", dpi=160)
    plt.close()


def plot_sample_waveforms(examples_by_label: dict[int, list[np.ndarray]], output_dir: Path, per_class_examples: int) -> None:
    labels = sorted(examples_by_label)
    if not labels:
        return
    rows = len(labels)
    cols = max(1, per_class_examples)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.4 * rows), squeeze=False)
    for row_idx, label_id in enumerate(labels):
        examples = examples_by_label[label_id][:cols]
        for col_idx in range(cols):
            axis = axes[row_idx][col_idx]
            if col_idx < len(examples):
                signal = examples[col_idx]
                axis.plot(signal[0], linewidth=0.7)
                axis.set_title(f"{label_id}:{LABEL_NAMES.get(label_id, label_id)} sample {col_idx + 1}")
                axis.set_xlim(0, signal.shape[1] - 1)
            else:
                axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_waveforms_by_class.png", dpi=160)
    plt.close()


def plot_class_mean_abs_waveform(mean_abs_by_label: dict[int, np.ndarray], output_dir: Path) -> None:
    plt.figure(figsize=(12, 5))
    for label_id, mean_waveform in sorted(mean_abs_by_label.items()):
        plt.plot(mean_waveform, linewidth=1.0, label=f"{label_id}:{LABEL_NAMES.get(label_id, label_id)}")
    plt.title("Class mean absolute waveform")
    plt.xlabel("time index")
    plt.ylabel("mean absolute amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "class_mean_abs_waveform.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    configure_matplotlib_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    manifest = manifest.reset_index(drop=True)
    signal_sample_size = None if args.full_signal_eda else args.sample_size
    signal_frame = stratified_sample(manifest, signal_sample_size, args.seed)

    LOGGER.info("Manifest rows=%s columns=%s", len(manifest), len(manifest.columns))
    LOGGER.info("Signal EDA rows=%s full_signal_eda=%s", len(signal_frame), args.full_signal_eda)

    summary = manifest_summary(manifest, args.output_dir)
    summary["signal_eda_rows"] = int(len(signal_frame))
    summary["signal_eda_full"] = bool(args.full_signal_eda)

    plot_label_distribution(manifest, args.output_dir)
    plot_manifest_metadata(manifest, args.output_dir)

    signal_summary_name = "signal_summary_full.csv" if args.full_signal_eda else "signal_summary_sample.csv"
    stats_frame, phase_matrix, examples_by_label, mean_abs_by_label = load_signal_eda(
        signal_frame,
        args.output_dir,
        signal_summary_name,
    )
    plot_signal_stats(stats_frame, args.output_dir)
    plot_phase_heatmap(phase_matrix, args.output_dir)
    plot_sample_waveforms(examples_by_label, args.output_dir, args.per_class_examples)
    plot_class_mean_abs_waveform(mean_abs_by_label, args.output_dir)

    summary["signal_stat_mean_by_label"] = (
        stats_frame.groupby("label_id")[["rms", "abs_p99", "max_abs", "pulse_rate"]].mean().round(6).to_dict()
    )
    (args.output_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("EDA outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
