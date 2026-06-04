"""Train partial-discharge time-series classification models.

This is the top-level CLI for the CSV-only time-series track.  It keeps model
shape handling centralized: each registered model declares the input layout it
expects, and the shared runner builds the dataset as either (B, 20, 7680) or
(B, 7680, 20).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ml.src.experiments.runner import run_single_experiment
from ml.src.models.registry import MODEL_REGISTRY

LOGGER = logging.getLogger(__name__)

CORE_MODELS = ["gru", "inception_time", "patchtst", "timesnet", "moment"]
EXTENDED_MODELS = ["tcn", "resnet1d", "moderntcn", "itransformer", "timemixer", "units", "gpt4ts", "ts2vec"]
EXTENDED_COST_NOTES = {
    "tcn": "medium: convolution baseline, usually manageable after small smoke runs",
    "resnet1d": "medium: 1D CNN baseline, usually manageable after small smoke runs",
    "moderntcn": "medium_high: modern CNN/TCN, resize/seq_len controls cost",
    "itransformer": "medium_high: attention-based model, subset first",
    "timemixer": "high: modern long-sequence model, subset first",
    "units": "very_high: foundation/unified model, subset only before full runs",
    "gpt4ts": "very_high: GPT-2/LM transfer path, sinception_timeubset only before full runs",
    "ts2vec": "very_high: self-supervised representation training plus classifier",
}
CPU_ONLY_MODELS = [
    "minirocket",
    "multirocket",
    "hydra",
    "sktime_summary",
    "sktime_catch22",
    "sktime_random_interval",
    "sktime_tsfresh",
    "sktime_freshprince",
    "sktime_rocket",
    "sktime_arsenal",
    "feature_logistic",
    "feature_svm",
    "feature_random_forest",
    "feature_tabpfn",
]
CPU_BASELINE_COMMANDS = {
    "minirocket": "python ml/scripts/run_minirocket.py",
    "multirocket": "python ml/scripts/run_multirocket.py",
    "hydra": "python ml/scripts/run_hydra.py",
    "sktime_summary": "python ml/scripts/run_sktime_classifier.py --model summary",
    "sktime_catch22": "python ml/scripts/run_sktime_classifier.py --model catch22",
    "sktime_random_interval": (
        "python ml/scripts/run_sktime_classifier.py --model random_interval --allow-expensive"
    ),
    "sktime_tsfresh": (
        "python ml/scripts/run_sktime_classifier.py --model tsfresh --allow-expensive"
    ),
    "sktime_freshprince": (
        "python ml/scripts/run_sktime_classifier.py --model freshprince --allow-expensive"
    ),
    "sktime_rocket": "python ml/scripts/run_sktime_classifier.py --model rocket",
    "sktime_arsenal": (
        "python ml/scripts/run_sktime_classifier.py --model arsenal --allow-expensive"
    ),
    "feature_logistic": "python ml/scripts/run_feature_baseline.py --model logistic",
    "feature_svm": "python ml/scripts/run_feature_baseline.py --model svm",
    "feature_random_forest": "python ml/scripts/run_feature_baseline.py --model random_forest",
    "feature_tabpfn": "python ml/scripts/run_feature_baseline.py --model tabpfn",
}


@dataclass(frozen=True)
class TrainPreset:
    auto_batch_start_size: int
    learning_rate: float
    model_params: dict[str, Any] = field(default_factory=dict)
    runner: str = "torch"


MODEL_PRESETS: dict[str, TrainPreset] = {
    "gru": TrainPreset(auto_batch_start_size=16, learning_rate=1e-3),
    "tcn": TrainPreset(auto_batch_start_size=8, learning_rate=1e-3),
    "inception_time": TrainPreset(auto_batch_start_size=8, learning_rate=1e-3),
    "resnet1d": TrainPreset(auto_batch_start_size=8, learning_rate=1e-3),
    "moderntcn": TrainPreset(auto_batch_start_size=4, learning_rate=1e-4, model_params={"seq_len": 4096}),
    "patchtst": TrainPreset(auto_batch_start_size=4, learning_rate=2e-4),
    "timesnet": TrainPreset(auto_batch_start_size=4, learning_rate=1e-4, model_params={"seq_len": 4096}),
    "itransformer": TrainPreset(auto_batch_start_size=4, learning_rate=1e-4),
    "timemixer": TrainPreset(auto_batch_start_size=2, learning_rate=1e-4, model_params={"seq_len": 4096}),
    "moment": TrainPreset(
        auto_batch_start_size=1,
        learning_rate=1e-4,
        model_params={"seq_len": 512, "freeze_backbone": True},
    ),
    "units": TrainPreset(auto_batch_start_size=1, learning_rate=1e-4, model_params={"seq_len": 1024}),
    "gpt4ts": TrainPreset(auto_batch_start_size=1, learning_rate=1e-4, model_params={"seq_len": 1024}),
    "ts2vec": TrainPreset(
        auto_batch_start_size=0,
        learning_rate=0.0,
        model_params={"seq_len": 1024},
        runner="ts2vec",
    ),
}


def parse_key_value(raw: str) -> tuple[str, Any]:
    """Parse KEY=VALUE CLI overrides with basic Python scalar conversion."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        parsed: Any = lowered == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
    return key.strip(), parsed


def resolve_model(model_arg: str) -> str:
    if "," in model_arg:
        raise ValueError(
            "One train.py run must train exactly one model. "
            "Do not pass comma-separated model names."
        )
    model_name = model_arg.strip()
    if model_name in CPU_ONLY_MODELS:
        return model_name
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(name for name in MODEL_REGISTRY if name not in CPU_ONLY_MODELS))
        cpu_only = ", ".join(sorted(CPU_ONLY_MODELS))
        raise ValueError(f"Unsupported model: {model_name}. Supported GPU models: {supported}. CPU-only optional: {cpu_only}")
    return model_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gru",
        help="One concrete model name only. Groups are shown by --list-models but cannot be trained as one run.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("Train/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Manual batch size. If omitted, train.py auto-resolves a CUDA batch size under the GPU memory target.",
    )
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-auto-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None, help="Override per-model learning-rate preset.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--model-param",
        action="append",
        type=parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help="Override model config. Example: --model-param seq_len=2048",
    )
    parser.add_argument("--list-models", action="store_true", help="Print available model groups and exit.")
    return parser


def run_special_runner(
    runner: str,
    manifest: Path,
    output: Path,
    sample_size: int | None,
    valid_ratio: float,
    seed: int,
    epochs: int,
    model_params: dict[str, Any],
) -> None:
    if runner != "ts2vec":
        raise ValueError(f"Unsupported special GPU runner: {runner}")
    if sample_size is None:
        raise ValueError(f"{runner} loads data into memory. Set --sample-size explicitly before full-scale runs.")

    script = Path("ml/scripts/run_ts2vec.py")
    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--sample-size",
        str(sample_size),
        "--valid-ratio",
        str(valid_ratio),
        "--seed",
        str(seed),
    ]
    cmd += [
        "--epochs",
        str(epochs),
        "--device",
        "cuda",
        "--seq-len",
        str(model_params.get("seq_len", 1024)),
    ]

    LOGGER.info("Launching %s runner: %s", runner, " ".join(cmd))
    env = os.environ.copy()
    project_root = str(Path.cwd())
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_root if not current_pythonpath else f"{project_root}{os.pathsep}{current_pythonpath}"
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    if args.list_models:
        print("core:", ", ".join(CORE_MODELS))
        print("extended:", ", ".join(EXTENDED_MODELS))
        print("cpu_only:", ", ".join(CPU_ONLY_MODELS))
        print("cpu_only_runner_hints:")
        for name in CPU_ONLY_MODELS:
            print(f"  {name}: {CPU_BASELINE_COMMANDS[name]}")
        print("extended_cost_notes:")
        for name in EXTENDED_MODELS:
            print(f"  {name}: {EXTENDED_COST_NOTES[name]}")
        print("rule: one train.py run = one concrete model only")
        return

    try:
        model_name = resolve_model(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    cli_params = dict(args.model_param)
    LOGGER.info("Training model: %s", model_name)

    if model_name in CPU_ONLY_MODELS:
        LOGGER.info(
            "model=%s is a CPU-only baseline and is intentionally not trained by train.py.",
            model_name,
        )
        LOGGER.info("Use the dedicated one-model runner instead: %s", CPU_BASELINE_COMMANDS[model_name])
        return

    if not torch.cuda.is_available():
        LOGGER.info("CUDA GPU is not available. CPU training is disabled for this project; no training was started.")
        return
    LOGGER.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))

    preset = MODEL_PRESETS[model_name]
    model_params = dict(preset.model_params)
    model_params.update(cli_params)
    batch_size = args.batch_size
    auto_batch_start_size = preset.auto_batch_start_size
    learning_rate = args.learning_rate if args.learning_rate is not None else preset.learning_rate

    LOGGER.info(
        "Starting model=%s runner=%s batch_size=%s auto_batch_start_size=%s target_gpu_memory_utilization=%.2f lr=%s model_params=%s",
        model_name,
        preset.runner,
        batch_size if batch_size is not None else "auto",
        auto_batch_start_size,
        args.target_gpu_memory_utilization,
        learning_rate,
        model_params,
    )
    if preset.runner == "torch":
        run_single_experiment(
            model_name=model_name,
            manifest_path=args.manifest,
            output_path=args.output,
            sample_size=args.sample_size,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=batch_size,
            auto_batch_start_size=auto_batch_start_size,
            target_gpu_memory_utilization=args.target_gpu_memory_utilization,
            max_auto_batch_size=args.max_auto_batch_size,
            learning_rate=learning_rate,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            device="cuda",
            model_params=model_params,
        )
    else:
        run_special_runner(
            runner=preset.runner,
            manifest=args.manifest,
            output=args.output,
            sample_size=args.sample_size,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            epochs=args.epochs,
            model_params=model_params,
        )
    LOGGER.info("Finished model=%s", model_name)


if __name__ == "__main__":
    main()
