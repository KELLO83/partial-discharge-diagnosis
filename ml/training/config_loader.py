from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.training.schema import CliArgs, TrainingJob, TrainingPlan, TrainingTask

SUPPORTED_TASKS = frozenset({"timeseries", "vision", "vlm"})


def load_training_plan(config_path: Path) -> TrainingPlan:
    payload = _load_mapping(config_path)
    version = _int_field(payload, "version", default=1)
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("training config must contain a jobs list.")
    jobs = tuple(_parse_job(raw_job) for raw_job in raw_jobs)
    if not jobs:
        raise ValueError("training config must contain at least one job.")
    return TrainingPlan(version=version, jobs=jobs)


def _load_mapping(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"training config not found: {config_path}")
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        payload = _load_yaml(text)
    else:
        raise ValueError("training config must be .json, .yaml, or .yml")
    if not isinstance(payload, dict):
        raise ValueError("training config root must be a mapping.")
    return payload


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML. Install ml/vlm/requirements.txt or use JSON config.") from exc
    return yaml.safe_load(text)


def _parse_job(raw_job: object) -> TrainingJob:
    if not isinstance(raw_job, dict):
        raise ValueError("each training job must be a mapping.")
    name = _string_field(raw_job, "name")
    task = _task_field(raw_job)
    raw_args = raw_job.get("args", {})
    if not isinstance(raw_args, dict):
        raise ValueError(f"job {name} args must be a mapping.")
    return TrainingJob(
        name=name,
        task=task,
        args=_parse_args(raw_args),
        enabled=bool(raw_job.get("enabled", True)),
        description=str(raw_job.get("description", "")),
    )


def _parse_args(raw_args: dict[str, object]) -> CliArgs:
    return {str(key): _parse_arg_value(value) for key, value in raw_args.items()}


def _parse_arg_value(value: object) -> object:
    if isinstance(value, list):
        return [_parse_scalar(item) for item in value]
    return _parse_scalar(value)


def _parse_scalar(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"unsupported CLI argument value type: {type(value).__name__}")


def _task_field(raw_job: dict[str, object]) -> TrainingTask:
    task = _string_field(raw_job, "task")
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"unsupported training task: {task}. Supported tasks: {supported}")
    return task  # type: ignore[return-value]


def _string_field(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _int_field(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be an integer.")
