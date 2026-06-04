"""Run one partial discharge time-series classification experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.experiments.runner import run_single_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gru", help="Model name registered in ml.src.models.registry.")
    parser.add_argument("--manifest", type=Path, default=Path("Train/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pinned memory.")
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="Training device. CPU training is disabled.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    run_single_experiment(
        model_name=args.model,
        manifest_path=args.manifest,
        output_path=args.output,
        sample_size=args.sample_size,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=args.device,
    )


if __name__ == "__main__":
    main()
