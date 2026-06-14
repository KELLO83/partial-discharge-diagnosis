"""Create fixed train/valid split manifests for comparable experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.timeseries.src.data.loader import load_manifest

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("Train/manifest_random_split_seed42.csv"))
    parser.add_argument("--split-type", default="stratified_random", choices=["stratified_random"])
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_random_split(frame: pd.DataFrame, valid_ratio: float, seed: int) -> pd.DataFrame:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1.")
    parts = []
    for _, part in frame.groupby("label_id", sort=True):
        shuffled = part.sample(frac=1.0, random_state=seed).copy()
        n_valid = max(1, int(round(len(shuffled) * valid_ratio)))
        shuffled["split"] = "train"
        shuffled.iloc[:n_valid, shuffled.columns.get_loc("split")] = "valid"
        parts.append(shuffled)
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    manifest = load_manifest(args.manifest)
    if args.split_type == "stratified_random":
        output = stratified_random_split(manifest, valid_ratio=args.valid_ratio, seed=args.seed)
    else:
        raise ValueError(f"Unsupported split_type: {args.split_type}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    LOGGER.info("Split manifest saved: %s", args.output)
    LOGGER.info("Split counts: %s", output["split"].value_counts().to_dict())
    LOGGER.info("Label by split:\n%s", pd.crosstab(output["split"], output["label_id"]))


if __name__ == "__main__":
    main()
