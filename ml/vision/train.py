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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> VisionTrainingConfig:
    return VisionTrainingConfig(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        sample_size=args.sample_size,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        dry_run=args.dry_run,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    summary_path = run_vision_training(build_config(parse_args()))
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
