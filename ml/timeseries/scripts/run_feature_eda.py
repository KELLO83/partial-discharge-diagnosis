"""Analyze extracted tabular features before partial-discharge feature baselines."""

from __future__ import annotations

import argparse
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from tqdm.auto import tqdm

from ml.timeseries.scripts.run_feature_baseline import extract_features, metadata_features
from ml.timeseries.src.data.loader import load_manifest, read_timeseries_csv

LOGGER = logging.getLogger(__name__)

LABEL_NAMES = {
    0: "normal",
    1: "noise",
    2: "surface",
    3: "corona",
    4: "void",
}


def configure_matplotlib_font() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/feature_eda_small"))
    parser.add_argument("--feature-set", default="small", choices=["small", "medium", "full"])
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified sample size. Default uses all rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--top-k-plot", type=int, default=24)
    return parser.parse_args()


def stratified_sample(frame: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    parts = []
    per_class = max(1, sample_size // int(frame["label_id"].nunique()))
    for _, part in frame.groupby("label_id", sort=True):
        parts.append(part.sample(min(len(part), per_class), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def feature_names(feature_set: str, include_metadata: bool) -> list[str]:
    global_names = [
        "global_mean",
        "global_std",
        "global_median",
        "global_mad",
        "global_min",
        "global_max",
        "global_p01",
        "global_p05",
        "global_p25",
        "global_p75",
        "global_p95",
        "global_p99",
        "global_iqr",
        "global_rms",
        "global_skew",
        "global_kurtosis",
    ]
    spectral_names = [
        "fft_band_ratio_0",
        "fft_band_ratio_1",
        "fft_band_ratio_2",
        "fft_band_ratio_3",
        "fft_centroid",
        "fft_bandwidth",
        "fft_entropy",
        "fft_flatness",
        "fft_dominant_freq_ratio",
        "fft_dominant_power_ratio",
    ]
    pulse_names = [
        "pulse_count_th3",
        "pulse_count_th5",
        "pulse_count_th8",
        "pulse_rate_th5",
        "pulse_peak_max",
        "pulse_peak_mean",
        "pulse_peak_p95",
        "pulse_width_mean",
        "pulse_width_std",
        "pulse_interval_mean",
        "pulse_interval_std",
        "pulse_burstiness",
    ]
    cycle_names = [
        "cycle_peak_mean",
        "cycle_peak_std",
        "cycle_peak_p95",
        "cycle_peak_max",
        "cycle_rms_mean",
        "cycle_rms_std",
        "cycle_rms_p95",
        "cycle_rms_max",
        "cycle_pulse_count_mean",
        "cycle_pulse_count_std",
        "cycle_pulse_count_p95",
        "cycle_pulse_count_max",
        "cycle_active_ratio",
        "cycle_empty_ratio",
        "cycle_active_run_max",
        "cycle_peak_variability",
    ]
    half_names = [
        "half_positive_pulse_count",
        "half_negative_pulse_count",
        "half_pos_neg_count_ratio",
        "half_positive_max",
        "half_negative_max",
        "half_pos_neg_max_ratio",
        "phase_entropy",
        "phase_concentration",
    ]
    if feature_set == "small":
        names = (
            global_names[:14]
            + spectral_names[:8]
            + [f"amp_hist_{idx:02d}" for idx in range(12)]
            + pulse_names[:10]
            + cycle_names[:12]
            + half_names
        )
    elif feature_set == "medium":
        names = (
            global_names
            + spectral_names
            + pulse_names
            + cycle_names
            + [f"segment_summary_{idx:02d}" for idx in range(18)]
            + [f"phase_count_{idx:02d}" for idx in range(24)]
            + [f"phase_max_{idx:02d}" for idx in range(24)]
            + half_names
        )
    else:
        names = (
            global_names
            + spectral_names
            + pulse_names
            + cycle_names
            + [f"segment_summary_{idx:02d}" for idx in range(24)]
            + half_names
            + [f"numeric_prpd_{idx:03d}" for idx in range(96)]
        )
    if include_metadata:
        names += [
            "metadata_temperature",
            "metadata_humidity",
            "metadata_recording_time_length",
            "metadata_equipment_rated_voltage",
            "metadata_equipment_rated_current",
            "metadata_power_supply_frequency",
            "metadata_power_supply_ramping_up_time",
            "metadata_power_supply_cutoff_current",
            "metadata_clearance_distance",
        ]
    return names


def feature_group(name: str) -> str:
    if name.startswith("global_"):
        return "global"
    if name.startswith("fft_"):
        return "spectral"
    if name.startswith("amp_hist_"):
        return "amplitude_histogram"
    if name.startswith("pulse_"):
        return "pulse"
    if name.startswith("cycle_"):
        return "cycle"
    if name.startswith("half_") or name.startswith("phase_"):
        return "phase_half_cycle"
    if name.startswith("segment_"):
        return "segment"
    if name.startswith("numeric_prpd_"):
        return "numeric_prpd"
    if name.startswith("metadata_"):
        return "metadata"
    return "other"


def load_feature_frame(frame: pd.DataFrame, feature_set: str, include_metadata: bool) -> pd.DataFrame:
    rows = []
    names = feature_names(feature_set, include_metadata)
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"extract {feature_set} features"):
        features = extract_features(read_timeseries_csv(row["timeseries_path"]), feature_set=feature_set)
        if include_metadata:
            features = np.concatenate([features, metadata_features(row)])
        rows.append(features)
    features = np.stack(rows, axis=0)
    if features.shape[1] != len(names):
        raise ValueError(f"Feature name count mismatch: X has {features.shape[1]}, names has {len(names)}")
    feature_frame = pd.DataFrame(features, columns=names)
    feature_frame.insert(0, "label_id", frame["label_id"].to_numpy(dtype=np.int64))
    feature_frame.insert(1, "label_name", feature_frame["label_id"].map(LABEL_NAMES))
    return feature_frame


def save_correlation(feature_frame: pd.DataFrame, feature_columns: list[str], output_dir: Path) -> None:
    corr = feature_frame[feature_columns].corr(method="pearson").fillna(0.0)
    corr.to_csv(output_dir / "feature_correlation.csv", encoding="utf-8-sig")

    high_pairs = []
    values = corr.to_numpy()
    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):
            value = float(values[i, j])
            if abs(value) >= 0.95:
                high_pairs.append(
                    {
                        "feature_a": feature_columns[i],
                        "feature_b": feature_columns[j],
                        "correlation": value,
                    }
                )
    pd.DataFrame(high_pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False).to_csv(
        output_dir / "high_correlation_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="pearson r")
    tick_step = max(1, len(feature_columns) // 16)
    ticks = list(range(0, len(feature_columns), tick_step))
    plt.xticks(ticks, [feature_columns[idx] for idx in ticks], rotation=70, ha="right", fontsize=7)
    plt.yticks(ticks, [feature_columns[idx] for idx in ticks], fontsize=7)
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation_heatmap.png", dpi=160)
    plt.close()


def save_separability(feature_frame: pd.DataFrame, feature_columns: list[str], output_dir: Path, seed: int, rf_trees: int) -> pd.DataFrame:
    x = feature_frame[feature_columns].to_numpy(dtype=np.float32)
    y = feature_frame["label_id"].to_numpy(dtype=np.int64)
    variance = x.var(axis=0)
    f_scores, p_values = f_classif(x, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    mutual_info = mutual_info_classif(x, y, random_state=seed)

    forest = RandomForestClassifier(
        n_estimators=rf_trees,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    forest.fit(x, y)
    rf_importance = forest.feature_importances_

    ranking = pd.DataFrame(
        {
            "feature": feature_columns,
            "group": [feature_group(name) for name in feature_columns],
            "variance": variance,
            "anova_f": f_scores,
            "anova_p": p_values,
            "mutual_info": mutual_info,
            "rf_importance": rf_importance,
        }
    )
    ranking["rank_anova"] = ranking["anova_f"].rank(ascending=False, method="min")
    ranking["rank_mutual_info"] = ranking["mutual_info"].rank(ascending=False, method="min")
    ranking["rank_rf_importance"] = ranking["rf_importance"].rank(ascending=False, method="min")
    ranking["rank_mean"] = ranking[["rank_anova", "rank_mutual_info", "rank_rf_importance"]].mean(axis=1)
    ranking = ranking.sort_values("rank_mean")
    ranking.to_csv(output_dir / "feature_separability.csv", index=False, encoding="utf-8-sig")
    return ranking


def plot_top_features(feature_frame: pd.DataFrame, ranking: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    top_features = ranking.head(top_k)["feature"].tolist()
    if not top_features:
        return
    plot_rows = []
    for feature in top_features:
        for label_id, part in feature_frame.groupby("label_id", sort=True):
            values = part[feature].to_numpy(dtype=np.float32)
            plot_rows.append(
                {
                    "feature": feature,
                    "label_id": int(label_id),
                    "label_name": LABEL_NAMES.get(int(label_id), str(label_id)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std()),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                }
            )
    pd.DataFrame(plot_rows).to_csv(output_dir / "top_feature_stats_by_class.csv", index=False, encoding="utf-8-sig")

    group_importance = ranking.groupby("group", as_index=False)["rf_importance"].sum().sort_values("rf_importance", ascending=False)
    plt.figure(figsize=(9, 4))
    plt.bar(group_importance["group"], group_importance["rf_importance"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("sum rf_importance")
    plt.title("Feature-group importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_group_importance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, max(5, min(12, len(top_features) * 0.35))))
    plt.barh(ranking.head(top_k)["feature"][::-1], ranking.head(top_k)["rank_mean"][::-1])
    plt.xlabel("mean rank, lower is better")
    plt.title("Top feature separability ranking")
    plt.tight_layout()
    plt.savefig(output_dir / "top_feature_ranking.png", dpi=160)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    configure_matplotlib_font()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_manifest(args.manifest)
    frame = stratified_sample(frame, args.sample_size, args.seed)
    LOGGER.info("Feature EDA rows=%s feature_set=%s include_metadata=%s", len(frame), args.feature_set, args.include_metadata)

    feature_frame = load_feature_frame(frame, args.feature_set, args.include_metadata)
    feature_path = args.output_dir / f"features_{args.feature_set}.csv"
    feature_frame.to_csv(feature_path, index=False, encoding="utf-8-sig")

    feature_columns = [column for column in feature_frame.columns if column not in {"label_id", "label_name"}]
    low_variance = pd.DataFrame(
        {
            "feature": feature_columns,
            "variance": feature_frame[feature_columns].var(axis=0).to_numpy(),
        }
    ).sort_values("variance")
    low_variance.to_csv(args.output_dir / "feature_variance.csv", index=False, encoding="utf-8-sig")

    save_correlation(feature_frame, feature_columns, args.output_dir)
    ranking = save_separability(feature_frame, feature_columns, args.output_dir, args.seed, args.rf_trees)
    plot_top_features(feature_frame, ranking, args.output_dir, args.top_k_plot)

    summary = {
        "rows": int(len(feature_frame)),
        "feature_set": args.feature_set,
        "include_metadata": bool(args.include_metadata),
        "feature_count": int(len(feature_columns)),
        "top_features": ranking.head(10)["feature"].tolist(),
    }
    pd.Series(summary).to_json(args.output_dir / "feature_eda_summary.json", force_ascii=False, indent=2)
    LOGGER.info("Saved feature EDA outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
