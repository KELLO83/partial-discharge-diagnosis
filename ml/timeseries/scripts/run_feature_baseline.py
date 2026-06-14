"""Run feature-level tabular baselines from partial-discharge time-series CSVs."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from ml.timeseries.src.data.loader import load_manifest, make_stratified_split, read_timeseries_csv
from ml.timeseries.src.eval.metrics import classification_metrics
from ml.timeseries.src.experiments.logger import append_experiment_result

LOGGER = logging.getLogger(__name__)
SAMPLES_PER_ROW = 7680
POWER_FREQUENCY_HZ = 60
CYCLE_LENGTH = SAMPLES_PER_ROW // POWER_FREQUENCY_HZ
PHASE_BINS_MEDIUM = 24
PHASE_BINS_PRPD = 12
AMPLITUDE_BINS_PRPD = 8
SAFE_NUMERIC_METADATA = (
    "temperature",
    "humidity",
    "recording_time_length",
)
SAFE_UNIT_METADATA = (
    "equipment_rated_voltage",
    "equipment_rated_current",
    "power_supply_frequency",
    "power_supply_ramping_up_time",
    "power_supply_cutoff_current",
    "clearance_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logistic", choices=["logistic", "svm", "random_forest", "tabpfn"])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-set",
        default="small",
        choices=["small", "medium", "full"],
        help=(
            "Feature group size. small=64 CSV features, medium=128 CSV features, "
            "full=182 CSV features including compact numeric PRPD histogram."
        ),
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Append safe numeric metadata only. Path, label text, defect details, and IDs are never used as features.",
    )
    return parser.parse_args()


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value)
    chars = []
    for char in text:
        if char.isdigit() or char in {".", "-", "+"}:
            chars.append(char)
        elif chars:
            break
    try:
        return float("".join(chars)) if chars else default
    except ValueError:
        return default


def normalized_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def robust_stats(values: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(mad, float(values.std()), 1e-6)
    return median, mad, scale


def local_peak_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    if values.size < 3:
        return np.zeros_like(values, dtype=bool)
    mask = np.zeros_like(values, dtype=bool)
    mask[1:-1] = (values[1:-1] > threshold) & (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    return mask


def global_amplitude_features(flat: np.ndarray) -> np.ndarray:
    centered = flat - flat.mean()
    centered_std = max(float(centered.std()), 1e-6)
    median, mad, _ = robust_stats(flat)
    p01, p05, p25, p75, p95, p99 = np.percentile(flat, [1, 5, 25, 75, 95, 99])
    return np.asarray(
        [
            flat.mean(),
            flat.std(),
            median,
            mad,
            flat.min(),
            flat.max(),
            p01,
            p05,
            p25,
            p75,
            p95,
            p99,
            p75 - p25,
            np.sqrt(np.mean(flat**2)),
            np.mean(centered**3) / (centered_std**3),
            np.mean(centered**4) / (centered_std**4),
        ],
        dtype=np.float32,
    )


def spectral_features(flat: np.ndarray) -> np.ndarray:
    centered = flat - flat.mean()
    fft = np.fft.rfft(centered)
    power = (np.abs(fft) ** 2).astype(np.float64)
    if power.shape[0] > 1:
        power = power[1:]
    total = max(float(power.sum()), 1e-12)
    normalized = power / total
    splits = np.array_split(power, 4)
    band_ratios = [float(part.sum() / total) if part.size else 0.0 for part in splits]
    freqs = np.arange(1, power.shape[0] + 1, dtype=np.float64)
    centroid = float((freqs * normalized).sum())
    bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * normalized).sum()))
    entropy = float(-(normalized * np.log(normalized + 1e-12)).sum())
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
    dominant_idx = int(np.argmax(power)) if power.size else 0
    dominant_freq_ratio = float((dominant_idx + 1) / max(power.shape[0], 1))
    dominant_power_ratio = float(power[dominant_idx] / total) if power.size else 0.0
    return np.asarray(
        [
            *band_ratios,
            centroid / max(power.shape[0], 1),
            bandwidth / max(power.shape[0], 1),
            entropy,
            flatness,
            dominant_freq_ratio,
            dominant_power_ratio,
        ],
        dtype=np.float32,
    )


def robust_thresholds(flat_abs: np.ndarray) -> tuple[float, float, float]:
    median, mad, scale = robust_stats(flat_abs)
    return median + 3.0 * scale, median + 5.0 * scale, median + 8.0 * scale


def contiguous_true_lengths(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.asarray([], dtype=np.float32)
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return (changes[1::2] - changes[::2]).astype(np.float32)


def pulse_features(abs_signal: np.ndarray, thresholds: tuple[float, float, float]) -> np.ndarray:
    flat_abs = abs_signal.reshape(-1)
    th3, th5, th8 = thresholds
    peak_mask = local_peak_mask(flat_abs, th5)
    peak_values = flat_abs[peak_mask]
    peak_positions = np.flatnonzero(peak_mask)
    widths = contiguous_true_lengths(flat_abs > th5)
    if peak_positions.size > 1:
        peak_interval = np.diff(peak_positions).astype(np.float32)
        interval_mean = float(peak_interval.mean())
        interval_std = float(peak_interval.std())
    else:
        interval_mean = 0.0
        interval_std = 0.0
    burstiness = float(peak_values.std() / max(float(peak_values.mean()), 1e-6)) if peak_values.size else 0.0
    return np.asarray(
        [
            float((flat_abs > th3).sum()),
            float((flat_abs > th5).sum()),
            float((flat_abs > th8).sum()),
            float((flat_abs > th5).mean()),
            float(peak_values.max()) if peak_values.size else 0.0,
            float(peak_values.mean()) if peak_values.size else 0.0,
            float(np.percentile(peak_values, 95)) if peak_values.size else 0.0,
            float(widths.mean()) if widths.size else 0.0,
            float(widths.std()) if widths.size else 0.0,
            interval_mean,
            interval_std,
            burstiness,
        ],
        dtype=np.float32,
    )


def cycle_features(abs_signal: np.ndarray, threshold: float) -> np.ndarray:
    usable_length = (abs_signal.shape[1] // CYCLE_LENGTH) * CYCLE_LENGTH
    cycles = abs_signal[:, :usable_length].reshape(abs_signal.shape[0], -1, CYCLE_LENGTH)
    cycle_peak = cycles.max(axis=2).reshape(-1)
    cycle_rms = np.sqrt(np.mean(cycles**2, axis=2)).reshape(-1)
    cycle_pulse_count = (cycles > threshold).sum(axis=2).reshape(-1)
    active = cycle_pulse_count > 0
    active_lengths = contiguous_true_lengths(active)
    return np.asarray(
        [
            cycle_peak.mean(),
            cycle_peak.std(),
            np.percentile(cycle_peak, 95),
            cycle_peak.max(),
            cycle_rms.mean(),
            cycle_rms.std(),
            np.percentile(cycle_rms, 95),
            cycle_rms.max(),
            cycle_pulse_count.mean(),
            cycle_pulse_count.std(),
            np.percentile(cycle_pulse_count, 95),
            cycle_pulse_count.max(),
            active.mean(),
            1.0 - active.mean(),
            float(active_lengths.max()) if active_lengths.size else 0.0,
            float(np.mean(np.abs(np.diff(cycle_peak)))) if cycle_peak.size > 1 else 0.0,
        ],
        dtype=np.float32,
    )


def segment_summary_features(signal: np.ndarray, threshold: float, summary_count: int) -> np.ndarray:
    abs_signal = np.abs(signal.astype(np.float32))
    segment_features = np.stack(
        [
            signal.mean(axis=1),
            signal.std(axis=1),
            np.sqrt(np.mean(signal**2, axis=1)),
            abs_signal.max(axis=1),
            np.percentile(abs_signal, 95, axis=1),
            (abs_signal > threshold).sum(axis=1),
        ],
        axis=1,
    )
    summaries = []
    for column in range(segment_features.shape[1]):
        values = segment_features[:, column]
        if summary_count == 3:
            summaries.extend([values.mean(), values.std(), values.max()])
        elif summary_count == 4:
            summaries.extend([values.mean(), values.std(), values.min(), values.max()])
        else:
            raise ValueError(f"Unsupported segment summary_count: {summary_count}")
    return np.asarray(summaries, dtype=np.float32)


def phase_count_max_features(abs_signal: np.ndarray, threshold: float) -> np.ndarray:
    time_index = np.arange(abs_signal.shape[1], dtype=np.int32)
    phase_bin_per_time = ((time_index % CYCLE_LENGTH) * PHASE_BINS_MEDIUM // CYCLE_LENGTH).astype(np.int32)
    repeated_phase_bins = np.broadcast_to(phase_bin_per_time, abs_signal.shape).reshape(-1)
    flat_abs = abs_signal.reshape(-1)
    event_mask = flat_abs > threshold
    counts = np.bincount(repeated_phase_bins[event_mask], minlength=PHASE_BINS_MEDIUM).astype(np.float32)
    counts = counts / max(float(counts.sum()), 1.0)
    max_by_phase = np.zeros(PHASE_BINS_MEDIUM, dtype=np.float32)
    for phase_bin in range(PHASE_BINS_MEDIUM):
        values = flat_abs[repeated_phase_bins == phase_bin]
        max_by_phase[phase_bin] = float(values.max()) if values.size else 0.0
    return np.concatenate([counts, max_by_phase])


def half_cycle_features(abs_signal: np.ndarray, threshold: float) -> np.ndarray:
    time_index = np.arange(abs_signal.shape[1], dtype=np.int32)
    phase_position = time_index % CYCLE_LENGTH
    positive_mask_time = phase_position < (CYCLE_LENGTH // 2)
    positive = abs_signal[:, positive_mask_time].reshape(-1)
    negative = abs_signal[:, ~positive_mask_time].reshape(-1)
    pos_events = positive > threshold
    neg_events = negative > threshold
    pos_count = float(pos_events.sum())
    neg_count = float(neg_events.sum())
    phase_counts = phase_count_max_features(abs_signal, threshold)[:PHASE_BINS_MEDIUM]
    entropy = float(-(phase_counts * np.log(phase_counts + 1e-12)).sum())
    concentration = float(phase_counts.max()) if phase_counts.size else 0.0
    return np.asarray(
        [
            pos_count,
            neg_count,
            pos_count / max(neg_count, 1.0),
            float(positive.max()) if positive.size else 0.0,
            float(negative.max()) if negative.size else 0.0,
            float(positive.max()) / max(float(negative.max()), 1e-6) if negative.size else 0.0,
            entropy,
            concentration,
        ],
        dtype=np.float32,
    )


def numeric_prpd_histogram(abs_signal: np.ndarray, threshold: float) -> np.ndarray:
    time_index = np.arange(abs_signal.shape[1], dtype=np.int32)
    phase_bin_per_time = ((time_index % CYCLE_LENGTH) * PHASE_BINS_PRPD // CYCLE_LENGTH).astype(np.int32)
    repeated_phase_bins = np.broadcast_to(phase_bin_per_time, abs_signal.shape).reshape(-1)
    flat_abs = abs_signal.reshape(-1)
    event_mask = flat_abs > threshold
    event_phase = repeated_phase_bins[event_mask]
    event_amp = flat_abs[event_mask]
    if event_amp.size:
        amp_hi = max(float(np.percentile(event_amp, 99.5)), 1e-6)
        prpd_hist, _, _ = np.histogram2d(
            event_phase,
            np.clip(event_amp, 0.0, amp_hi),
            bins=(PHASE_BINS_PRPD, AMPLITUDE_BINS_PRPD),
            range=((0, PHASE_BINS_PRPD), (0.0, amp_hi)),
        )
        prpd_hist = prpd_hist.astype(np.float32).reshape(-1)
        prpd_hist = prpd_hist / max(float(prpd_hist.sum()), 1.0)
    else:
        prpd_hist = np.zeros(PHASE_BINS_PRPD * AMPLITUDE_BINS_PRPD, dtype=np.float32)
    return prpd_hist


def extract_features(signal: np.ndarray, feature_set: str = "small") -> np.ndarray:
    """Extract compact tabular feature sets from one (20, 7680) signal."""
    if feature_set not in {"small", "medium", "full"}:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    flat = signal.reshape(-1).astype(np.float32)
    abs_signal = np.abs(signal.astype(np.float32))
    flat_abs = abs_signal.reshape(-1)
    thresholds = robust_thresholds(flat_abs)
    threshold = thresholds[1]
    global_features = global_amplitude_features(flat)
    spectral = spectral_features(flat)
    pulse = pulse_features(abs_signal, thresholds)
    cycle = cycle_features(abs_signal, threshold)
    half = half_cycle_features(abs_signal, threshold)

    if feature_set == "small":
        parts = [
            global_features[:14],
            spectral[:8],
            normalized_histogram(
                flat_abs,
                bins=12,
                value_range=(0.0, max(float(np.percentile(flat_abs, 99.5)), 1e-6)),
            ),
            pulse[:10],
            cycle[:12],
            half,
        ]
    elif feature_set == "medium":
        parts = [
            global_features,
            spectral,
            pulse,
            cycle,
            segment_summary_features(signal, threshold, summary_count=3),
            phase_count_max_features(abs_signal, threshold),
            half,
        ]
    else:
        parts = [
            global_features,
            spectral,
            pulse,
            cycle,
            segment_summary_features(signal, threshold, summary_count=4),
            half,
            numeric_prpd_histogram(abs_signal, threshold),
        ]
    feature_vector = np.concatenate(parts)
    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def metadata_features(row) -> np.ndarray:
    values = []
    for column in SAFE_NUMERIC_METADATA:
        values.append(safe_float(row.get(column, 0.0)))
    for column in SAFE_UNIT_METADATA:
        values.append(safe_float(row.get(column, 0.0)))
    return np.asarray(values, dtype=np.float32)


def load_features(frame, include_metadata: bool, feature_set: str) -> np.ndarray:
    rows = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="extract features", leave=False):
        features = extract_features(read_timeseries_csv(row["timeseries_path"]), feature_set=feature_set)
        if include_metadata:
            features = np.concatenate([features, metadata_features(row)])
        rows.append(features)
    return np.stack(rows, axis=0)


def create_model(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
    if name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError("TabPFN baseline requires tabpfn. Install it before running --model tabpfn.") from exc
        return TabPFNClassifier()
    raise ValueError(f"Unsupported feature baseline: {name}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    manifest = load_manifest(args.manifest)
    split = make_stratified_split(manifest, valid_ratio=args.valid_ratio, seed=args.seed, sample_size=args.sample_size)

    LOGGER.info("Loading feature baseline data: train=%s valid=%s model=%s", len(split.train), len(split.valid), args.model)
    x_train = load_features(split.train, include_metadata=args.include_metadata, feature_set=args.feature_set)
    y_train = split.train["label_id"].to_numpy(dtype=int)
    x_valid = load_features(split.valid, include_metadata=args.include_metadata, feature_set=args.feature_set)
    y_valid = split.valid["label_id"].to_numpy(dtype=int)

    model = create_model(args.model, args.seed)
    start_train = time.perf_counter()
    LOGGER.info(
        "Training feature baseline: %s feature_set=%s features=%s include_metadata=%s",
        args.model,
        args.feature_set,
        x_train.shape[1],
        args.include_metadata,
    )
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    pred = model.predict(x_valid)
    predict_time = time.perf_counter() - start_predict
    metrics = classification_metrics(y_valid, pred)
    LOGGER.info("Validation: accuracy=%.6f macro_f1=%.6f", metrics.accuracy, metrics.macro_f1)

    append_experiment_result(
        args.output,
        {
            "experiment_id": f"feature_{args.model}_{args.sample_size}_seed{args.seed}",
            "model_name": f"feature_{args.model}",
            "model_family": "feature_baseline",
            "training_mode": "feature_tabular",
            "pretrained": args.model == "tabpfn",
            "device": "cpu",
            "manifest_path": str(args.manifest),
            "split_type": split.split_type,
            "sample_size": args.sample_size,
            "train_rows": len(split.train),
            "valid_rows": len(split.valid),
            "valid_ratio": args.valid_ratio,
            "split_seed": args.seed,
            "n_features": x_train.shape[1],
            "feature_set": args.feature_set,
            "include_metadata": args.include_metadata,
            "train_time_sec": round(train_time, 6),
            "predict_time_sec": round(predict_time, 6),
            "valid_accuracy": metrics.accuracy,
            "valid_macro_f1": metrics.macro_f1,
            "valid_weighted_f1": metrics.weighted_f1,
            "valid_balanced_accuracy": metrics.balanced_accuracy,
            "valid_per_class_f1": metrics.per_class_f1,
            "valid_per_class_precision": metrics.per_class_precision,
            "valid_per_class_recall": metrics.per_class_recall,
            "valid_pd_to_normal_error_count": metrics.pd_to_normal_error_count,
            "valid_normal_recall": metrics.per_class_recall[0],
            "valid_noise_recall": metrics.per_class_recall[1],
            "valid_surface_recall": metrics.per_class_recall[2],
            "valid_corona_recall": metrics.per_class_recall[3],
            "valid_void_recall": metrics.per_class_recall[4],
            "valid_confusion_matrix": metrics.confusion_matrix,
        },
    )


if __name__ == "__main__":
    main()
