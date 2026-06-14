from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ml.training.schema import CliValue, TrainingJob, TrainingPlan, TrainingRunOptions, TrainingRunResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ENTRYPOINTS = {
    "timeseries": Path("ml/timeseries/train.py"),
    "vision": Path("ml/vision/train.py"),
    "vlm": Path("ml/vlm/train.py"),
}


def build_job_command(job: TrainingJob) -> tuple[str, ...]:
    script = TASK_ENTRYPOINTS[job.task]
    command = [sys.executable, str(script)]
    for key, value in job.args.items():
        command.extend(_arg_tokens(key, value))
    return tuple(command)


def run_training_plan(plan: TrainingPlan, options: TrainingRunOptions) -> list[TrainingRunResult]:
    results: list[TrainingRunResult] = []
    env = _training_env()
    for job in plan.jobs:
        command = build_job_command(job)
        if _should_skip(job, options):
            results.append(_skipped_result(job, command))
            continue
        if options.plan_only:
            results.append(_planned_result(job, command))
            continue
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        results.append(_finished_result(job, command, completed.returncode))
        if completed.returncode != 0 and options.stop_on_failure:
            break
    return results


def _should_skip(job: TrainingJob, options: TrainingRunOptions) -> bool:
    if not job.enabled:
        return True
    if not options.selectors:
        return False
    return not any(_matches_selector(job, selector) for selector in options.selectors)


def _matches_selector(job: TrainingJob, selector: str) -> bool:
    return selector in {job.name, job.task}


def _arg_tokens(key: str, value: CliValue | list[CliValue]) -> list[str]:
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_scalar_arg_tokens(key, item))
        return tokens
    return _scalar_arg_tokens(key, value)


def _scalar_arg_tokens(key: str, value: CliValue) -> list[str]:
    if value is None or value is False:
        return []
    flag = f"--{key.replace('_', '-')}"
    if value is True:
        return [flag]
    return [flag, str(value)]


def _training_env() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    project_root = str(PROJECT_ROOT)
    env["PYTHONPATH"] = project_root if not current_pythonpath else f"{project_root}{os.pathsep}{current_pythonpath}"
    return env


def _skipped_result(job: TrainingJob, command: tuple[str, ...]) -> TrainingRunResult:
    return TrainingRunResult(
        job_name=job.name,
        task=job.task,
        command=command,
        return_code=None,
        status="skipped",
    )


def _planned_result(job: TrainingJob, command: tuple[str, ...]) -> TrainingRunResult:
    return TrainingRunResult(
        job_name=job.name,
        task=job.task,
        command=command,
        return_code=None,
        status="planned",
    )


def _finished_result(job: TrainingJob, command: tuple[str, ...], return_code: int) -> TrainingRunResult:
    return TrainingRunResult(
        job_name=job.name,
        task=job.task,
        command=command,
        return_code=return_code,
        status="completed" if return_code == 0 else "failed",
    )
