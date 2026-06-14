from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TrainingTask = Literal["timeseries", "vision", "vlm"]
CliValue = str | int | float | bool | Path | None
CliArgs = dict[str, CliValue | list[CliValue]]


@dataclass(frozen=True, slots=True)
class TrainingJob:
    name: str
    task: TrainingTask
    args: CliArgs = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    version: int
    jobs: tuple[TrainingJob, ...]


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    job_name: str
    task: TrainingTask
    command: tuple[str, ...]
    return_code: int | None
    status: Literal["planned", "skipped", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class TrainingRunOptions:
    selectors: tuple[str, ...] = ()
    plan_only: bool = False
    stop_on_failure: bool = True
