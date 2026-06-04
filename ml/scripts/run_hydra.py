"""Run one HYDRA aeon experiment.

HYDRA is a CPU classical time-series classification baseline.  It is kept out
of train.py's GPU neural-model path on purpose.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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
    parser.add_argument("--n-jobs", type=int, default=14)
    return parser.parse_args()


def hydra_classifier(n_jobs: int) -> Any:
    try:
        module = importlib.import_module("aeon.classification.convolution_based")
    except ImportError as exc:
        raise ImportError(
            "HYDRA baseline requires aeon. Install a compatible aeon environment, "
            "then rerun ml/scripts/run_hydra.py."
        ) from exc

    classifier_cls = getattr(module, "HydraClassifier", None) or getattr(module, "MultiRocketHydraClassifier", None)
    if classifier_cls is None:
        raise ImportError(
            "Installed aeon does not expose HydraClassifier or MultiRocketHydraClassifier "
            "under aeon.classification.convolution_based."
        )

    try:
        return classifier_cls(n_jobs=n_jobs)
    except TypeError:
        return classifier_cls()


def load_panel(paths: list[str]) -> np.ndarray:
    arrays = []
    for path in tqdm(paths, desc="load csv", leave=False):
        arrays.append(read_timeseries_csv(path))
    return np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.float64)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    manifest = load_manifest(args.manifest)
    split = make_stratified_split(manifest, valid_ratio=args.valid_ratio, seed=args.seed, sample_size=args.sample_size)
    model = hydra_classifier(args.n_jobs)

    LOGGER.info("Loading HYDRA panels: train=%s valid=%s", len(split.train), len(split.valid))
    x_train = load_panel(split.train["timeseries_path"].tolist())
    y_train = split.train["label_id"].to_numpy(dtype=int)
    x_valid = load_panel(split.valid["timeseries_path"].tolist())
    y_valid = split.valid["label_id"].to_numpy(dtype=int)

    start_train = time.perf_counter()
    LOGGER.info("Training HYDRA aeon classifier")
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    LOGGER.info("Predicting HYDRA validation")
    pred = model.predict(x_valid)
    predict_time = time.perf_counter() - start_predict
    metrics = classification_metrics(y_valid, pred)
    LOGGER.info("Validation: accuracy=%.6f macro_f1=%.6f", metrics.accuracy, metrics.macro_f1)

    append_experiment_result(
        args.output,
        {
            "experiment_id": f"hydra_{args.sample_size}_seed{args.seed}",
            "model_name": "hydra",
            "model_family": "classical_tsc",
            "training_mode": "aeon_official",
            "pretrained": False,
            "device": "cpu",
            "manifest_path": str(args.manifest),
            "split_type": split.split_type,
            "sample_size": args.sample_size,
            "train_rows": len(split.train),
            "valid_rows": len(split.valid),
            "valid_ratio": args.valid_ratio,
            "split_seed": args.seed,
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
