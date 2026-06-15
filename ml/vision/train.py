from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.vision.src.schema import VisionTrainingConfig
from ml.vision.src.training import run_vision_training
from ml.training.artifacts import timestamped_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight PRPD vision classifier.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/vision"))
    parser.add_argument("--model-name", default="small_prpd_cnn")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--auto-batch-start-size", type=int, default=64)
    parser.add_argument(
        "--gpu-memory-fraction",
        "--gpu-memory-fracion",
        dest="gpu_memory_fraction",
        type=float,
        default=0.9,
        help="Target fraction of total GPU memory for automatic batch sizing.",
    )
    parser.add_argument("--max-auto-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--scheduler", default="onecycle", choices=("onecycle", "none"))
    parser.add_argument("--mixed-precision", default="fp16", choices=("off", "fp16", "bf16"))
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> VisionTrainingConfig:
    publish_dir = args.output_dir
    output_dir = timestamped_model_dir(publish_dir, args.model_name)
    return VisionTrainingConfig(
        manifest_path=args.manifest,
        output_dir=output_dir,
        publish_dir=publish_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        sample_size=args.sample_size,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        auto_batch_start_size=args.auto_batch_start_size,
        gpu_memory_fraction=args.gpu_memory_fraction,
        max_auto_batch_size=args.max_auto_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        mixed_precision=args.mixed_precision,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        device=args.device,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    summary_path = run_vision_training(build_config(parse_args()))
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
