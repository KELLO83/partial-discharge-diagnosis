"""Create label-by-metadata EDA tables for partial-discharge manifests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from ml.src.data.loader import load_manifest

LOGGER = logging.getLogger(__name__)

LABEL_NAMES = {
    0: "normal",
    1: "noise",
    2: "surface",
    3: "corona",
    4: "void",
}
CROSSTAB_COLUMNS = (
    "sensor_type",
    "insulator_type",
    "equipment_name",
    "equipment_rated_voltage",
    "clearance_distance",
    "recording_date",
)


def configure_matplotlib_font() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("Train/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/metadata_crosstab"))
    parser.add_argument("--top-k", type=int, default=30, help="Limit wide categorical plots to top-K values.")
    return parser.parse_args()


def normalize_label_index(table: pd.DataFrame) -> pd.DataFrame:
    table = table.reindex(sorted(LABEL_NAMES), fill_value=0)
    table.index.name = "label_id"
    table.insert(0, "label_name", [LABEL_NAMES[index] for index in table.index])
    return table


def save_heatmap(table: pd.DataFrame, column: str, output_dir: Path, top_k: int) -> None:
    plot_table = table.drop(columns=["label_name"], errors="ignore")
    column_totals = plot_table.sum(axis=0).sort_values(ascending=False)
    selected_columns = list(column_totals.head(top_k).index)
    plot_table = plot_table[selected_columns]
    if plot_table.empty:
        return

    row_sums = plot_table.sum(axis=1).replace(0, 1)
    normalized = plot_table.div(row_sums, axis=0)

    fig_width = max(8, min(24, len(selected_columns) * 0.8))
    plt.figure(figsize=(fig_width, 4.5))
    plt.imshow(normalized.to_numpy(), aspect="auto", cmap="Blues")
    plt.colorbar(label="row ratio")
    plt.xticks(range(len(selected_columns)), selected_columns, rotation=60, ha="right")
    plt.yticks(range(len(normalized.index)), [f"{idx}:{LABEL_NAMES[idx]}" for idx in normalized.index])
    plt.title(f"label_id x {column}")
    plt.tight_layout()
    plt.savefig(output_dir / f"label_by_{column}.png", dpi=160)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    configure_matplotlib_font()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_manifest(args.manifest)
    LOGGER.info("Loaded manifest rows=%s", len(frame))

    summary_rows = []
    for column in CROSSTAB_COLUMNS:
        if column not in frame.columns:
            LOGGER.info("Skip missing metadata column: %s", column)
            continue
        values = frame[column].fillna("<NA>").astype(str)
        table = pd.crosstab(frame["label_id"], values)
        table = normalize_label_index(table)
        table.to_csv(args.output_dir / f"label_by_{column}.csv", encoding="utf-8-sig")
        save_heatmap(table, column, args.output_dir, top_k=args.top_k)

        count_table = table.drop(columns=["label_name"], errors="ignore")
        row_sums = count_table.sum(axis=1).replace(0, 1)
        dominant_ratio = count_table.max(axis=1).div(row_sums)
        for label_id, ratio in dominant_ratio.items():
            dominant_value = count_table.loc[label_id].idxmax() if not count_table.empty else "<NA>"
            summary_rows.append(
                {
                    "metadata_column": column,
                    "label_id": label_id,
                    "label_name": LABEL_NAMES.get(int(label_id), str(label_id)),
                    "dominant_value": dominant_value,
                    "dominant_ratio": float(ratio),
                    "unique_values": int(count_table.shape[1]),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "metadata_crosstab_summary.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("Saved metadata cross-tab outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
