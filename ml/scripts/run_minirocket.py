"""Run one MiniROCKET official sktime experiment."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from tqdm.auto import tqdm

from ml.src.data.loader import load_manifest, make_stratified_split, read_timeseries_csv
from ml.src.eval.metrics import classification_metrics
from ml.src.experiments.logger import append_experiment_result

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("Train/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_panel(paths: list[str]) -> np.ndarray:
    arrays = []
    for path in tqdm(paths, desc="load csv", leave=False):
        arrays.append(read_timeseries_csv(path))
    return np.stack(arrays, axis=0)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    manifest = load_manifest(args.manifest)
    split = make_stratified_split(manifest, valid_ratio=args.valid_ratio, seed=args.seed, sample_size=args.sample_size)

    LOGGER.info("Loading MiniROCKET panels: train=%s valid=%s", len(split.train), len(split.valid))
    x_train = load_panel(split.train["timeseries_path"].tolist())
    y_train = split.train["label_id"].to_numpy(dtype=int)
    x_valid = load_panel(split.valid["timeseries_path"].tolist())
    y_valid = split.valid["label_id"].to_numpy(dtype=int)

    model = make_pipeline(MiniRocketMultivariate(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    start_train = time.perf_counter()
    LOGGER.info("Training MiniROCKET official sktime pipeline")
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    LOGGER.info("Predicting MiniROCKET validation")
    pred = model.predict(x_valid)
    predict_time = time.perf_counter() - start_predict
    metrics = classification_metrics(y_valid, pred)
    LOGGER.info("Train: accuracy=%.6f macro_f1=%.6f", metrics.accuracy, metrics.macro_f1)

    append_experiment_result(
        args.output,
        {
            "experiment_id": f"minirocket_{args.sample_size}_seed{args.seed}",
            "model_name": "minirocket",
            "model_family": "non_transformer",
            "training_mode": "sktime_official",
            "pretrained": False,
            "device": "cpu",
            "sample_size": args.sample_size,
            "train_rows": len(split.train),
            "valid_rows": len(split.valid),
            "valid_ratio": args.valid_ratio,
            "split_seed": args.seed,
            "train_time_sec": round(train_time, 6),
            "predict_time_sec": round(predict_time, 6),
            "valid_accuracy": metrics.accuracy,
            "valid_macro_f1": metrics.macro_f1,
            "valid_per_class_f1": metrics.per_class_f1,
            "valid_confusion_matrix": metrics.confusion_matrix,
        },
    )


if __name__ == "__main__":
    main()
