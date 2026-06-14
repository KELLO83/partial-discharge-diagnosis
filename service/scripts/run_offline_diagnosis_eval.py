from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.backend.app.application.offline import run_offline_mock_evaluation, summary_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/offline_diagnosis/mock_predictions.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/offline_diagnosis/summary.json"))
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_offline_mock_evaluation(args.manifest, args.output, args.sample_size)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(summary_to_json(summary), encoding="utf-8")
    print(summary_to_json(summary))


if __name__ == "__main__":
    main()
