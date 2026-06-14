"""Run one official sktime time-series classification experiment.

This runner is for CPU-only sktime classifiers.  It is intentionally separate
from train.py because train.py is reserved for one GPU neural model per run.
"""

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
from tqdm.auto import tqdm

from ml.timeseries.src.data.loader import load_manifest, make_stratified_split, read_timeseries_csv
from ml.timeseries.src.eval.metrics import classification_metrics
from ml.timeseries.src.experiments.logger import append_experiment_result

LOGGER = logging.getLogger(__name__)
EXPENSIVE_MODELS = {"random_interval", "arsenal", "tsfresh", "freshprince"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="catch22",
        choices=["summary", "catch22", "random_interval", "tsfresh", "freshprince", "rocket", "arsenal"],
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=14)
    parser.add_argument("--num-kernels", type=int, default=10000)
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--time-limit-minutes", type=float, default=0.0)
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="Required for RandomInterval, Arsenal, TSFresh, and FreshPRINCE. Use small subsets first.",
    )
    return parser.parse_args()


def load_panel(paths: list[str]) -> np.ndarray:
    arrays = []
    for path in tqdm(paths, desc="load csv", leave=False):
        arrays.append(read_timeseries_csv(path))
    return np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.float64)


def create_sktime_classifier(args: argparse.Namespace):
    if args.model == "summary":
        from sktime.classification.feature_based import SummaryClassifier

        return SummaryClassifier(n_jobs=args.n_jobs, random_state=args.seed)
    if args.model == "catch22":
        from sktime.classification.feature_based import Catch22Classifier

        return Catch22Classifier(replace_nans=True, n_jobs=args.n_jobs, random_state=args.seed)
    if args.model == "random_interval":
        from sktime.classification.feature_based import RandomIntervalClassifier

        return RandomIntervalClassifier(n_intervals=100, n_jobs=args.n_jobs, random_state=args.seed)
    if args.model == "tsfresh":
        from sktime.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(
            default_fc_parameters="efficient",
            relevant_feature_extractor=True,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if args.model == "freshprince":
        from sktime.classification.feature_based import FreshPRINCE

        return FreshPRINCE(
            default_fc_parameters="efficient",
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if args.model == "rocket":
        from sktime.classification.kernel_based import RocketClassifier

        return RocketClassifier(
            num_kernels=args.num_kernels,
            use_multivariate="auto",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if args.model == "arsenal":
        from sktime.classification.kernel_based import Arsenal

        return Arsenal(
            num_kernels=args.num_kernels,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            time_limit_in_minutes=args.time_limit_minutes,
        )
    raise ValueError(f"Unsupported sktime model: {args.model}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    if args.model in EXPENSIVE_MODELS and not args.allow_expensive:
        raise ValueError(
            f"{args.model} can be very slow. Rerun with --allow-expensive and a small --sample-size first."
        )
    if args.model in EXPENSIVE_MODELS and args.sample_size is None:
        raise ValueError(f"{args.model} must not start without --sample-size.")

    manifest = load_manifest(args.manifest)
    split = make_stratified_split(manifest, valid_ratio=args.valid_ratio, seed=args.seed, sample_size=args.sample_size)

    LOGGER.info(
        "Loading sktime panels: model=%s train=%s valid=%s sample_size=%s",
        args.model,
        len(split.train),
        len(split.valid),
        args.sample_size,
    )
    x_train = load_panel(split.train["timeseries_path"].tolist())
    y_train = split.train["label_id"].to_numpy(dtype=int)
    x_valid = load_panel(split.valid["timeseries_path"].tolist())
    y_valid = split.valid["label_id"].to_numpy(dtype=int)

    model = create_sktime_classifier(args)
    start_train = time.perf_counter()
    LOGGER.info("Training official sktime classifier: %s", args.model)
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    LOGGER.info("Predicting sktime validation: %s", args.model)
    pred = model.predict(x_valid)
    predict_time = time.perf_counter() - start_predict
    metrics = classification_metrics(y_valid, pred)
    LOGGER.info("Validation: accuracy=%.6f macro_f1=%.6f", metrics.accuracy, metrics.macro_f1)

    append_experiment_result(
        args.output,
        {
            "experiment_id": f"sktime_{args.model}_{args.sample_size}_seed{args.seed}",
            "model_name": f"sktime_{args.model}",
            "model_family": "classical_tsc",
            "training_mode": "sktime_official_classifier",
            "pretrained": False,
            "device": "cpu",
            "manifest_path": str(args.manifest),
            "split_type": split.split_type,
            "sample_size": args.sample_size,
            "train_rows": len(split.train),
            "valid_rows": len(split.valid),
            "valid_ratio": args.valid_ratio,
            "split_seed": args.seed,
            "n_jobs": args.n_jobs,
            "num_kernels": args.num_kernels if args.model in {"rocket", "arsenal"} else "",
            "n_estimators": args.n_estimators if args.model == "arsenal" else "",
            "time_limit_minutes": args.time_limit_minutes if args.model in EXPENSIVE_MODELS else "",
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
