from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import load_training_plan, run_training_plan
from ml.training.schema import TrainingPlan, TrainingRunOptions, TrainingRunResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML training jobs from one config file.")
    parser.add_argument("--config", type=Path, default=Path("ml/configs/training_smoke.yaml"))
    parser.add_argument("--only", action="append", default=[], help="Run only matching job names or task names.")
    parser.add_argument("--plan-only", action="store_true", help="Print commands without executing jobs.")
    parser.add_argument("--list-jobs", action="store_true", help="Print configured jobs without executing them.")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_training_plan(args.config)
    if args.list_jobs:
        print(json.dumps(_job_listing(plan), ensure_ascii=False, indent=2))
        return
    options = TrainingRunOptions(
        selectors=tuple(args.only),
        plan_only=args.plan_only,
        stop_on_failure=not args.continue_on_failure,
    )
    results = run_training_plan(plan, options)
    print(json.dumps(_result_payload(results), ensure_ascii=False, indent=2))
    if any(result.status == "failed" for result in results):
        raise SystemExit(1)


def _job_listing(plan: TrainingPlan) -> dict[str, object]:
    return {
        "version": plan.version,
        "jobs": [
            {
                "name": job.name,
                "task": job.task,
                "enabled": job.enabled,
                "description": job.description,
            }
            for job in plan.jobs
        ],
    }


def _result_payload(results: list[TrainingRunResult]) -> dict[str, object]:
    return {
        "results": [
            {
                **asdict(result),
                "command": " ".join(result.command),
            }
            for result in results
        ]
    }


if __name__ == "__main__":
    main()
