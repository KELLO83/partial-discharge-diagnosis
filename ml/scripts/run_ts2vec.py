"""Run one TS2Vec official representation-learning experiment."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from ml.src.data.loader import load_manifest, make_stratified_split, read_timeseries_csv
from ml.src.eval.metrics import classification_metrics
from ml.src.experiments.logger import append_experiment_result
from ml.src.models.adapters import resize_time_axis_channel_first

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("Train/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_array(paths: list[str], seq_len: int) -> np.ndarray:
    arrays = []
    for path in tqdm(paths, desc="load csv", leave=False):
        signal = read_timeseries_csv(path)
        resized = resize_time_axis_channel_first(
            __import__("torch").from_numpy(signal).unsqueeze(0),
            seq_len,
        ).squeeze(0)
        arrays.append(resized.numpy().T)
    return np.stack(arrays, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    repo = Path("external/ts2vec").resolve()
    if not repo.exists():
        raise ImportError("TS2Vec official repo is required at external/ts2vec.")
    sys.path.insert(0, str(repo))
    from ts2vec import TS2Vec

    manifest = load_manifest(args.manifest)
    split = make_stratified_split(manifest, valid_ratio=args.valid_ratio, seed=args.seed, sample_size=args.sample_size)

    LOGGER.info("Loading TS2Vec arrays: train=%s valid=%s seq_len=%s", len(split.train), len(split.valid), args.seq_len)
    x_train = load_array(split.train["timeseries_path"].tolist(), args.seq_len)
    y_train = split.train["label_id"].to_numpy(dtype=int)
    x_valid = load_array(split.valid["timeseries_path"].tolist(), args.seq_len)
    y_valid = split.valid["label_id"].to_numpy(dtype=int)

    model = TS2Vec(input_dims=x_train.shape[-1], device=args.device, batch_size=min(8, len(x_train)), max_train_length=args.seq_len)
    start_train = time.perf_counter()
    LOGGER.info("Training official TS2Vec encoder")
    model.fit(x_train, n_epochs=args.epochs, verbose=True)
    train_repr = model.encode(x_train, encoding_window="full_series")
    valid_repr = model.encode(x_valid, encoding_window="full_series")
    if train_repr.ndim == 3:
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        valid_repr = valid_repr.reshape(valid_repr.shape[0], -1)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_repr, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    pred = classifier.predict(valid_repr)
    predict_time = time.perf_counter() - start_predict
    metrics = classification_metrics(y_valid, pred)
    LOGGER.info("Train: accuracy=%.6f macro_f1=%.6f", metrics.accuracy, metrics.macro_f1)

    append_experiment_result(
        args.output,
        {
            "experiment_id": f"ts2vec_{args.sample_size}_seed{args.seed}",
            "model_name": "ts2vec",
            "model_family": "representation",
            "training_mode": "official_encoder_linear_probe",
            "pretrained": False,
            "device": args.device,
            "sample_size": args.sample_size,
            "train_rows": len(split.train),
            "valid_rows": len(split.valid),
            "valid_ratio": args.valid_ratio,
            "split_seed": args.seed,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
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
