from __future__ import annotations

import json
import sys
from pathlib import Path

from ml.training import build_job_command, load_training_plan, run_training_plan
from ml.training.schema import TrainingRunOptions


def test_training_plan_loads_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    {
                        "name": "vision_smoke",
                        "task": "vision",
                        "enabled": True,
                        "args": {"sample_size": 20, "dry_run": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = load_training_plan(config_path)

    assert plan.version == 1
    assert plan.jobs[0].name == "vision_smoke"
    assert plan.jobs[0].args["sample_size"] == 20


def test_build_job_command_converts_config_args_to_cli_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    {
                        "name": "ts_gru",
                        "task": "timeseries",
                        "args": {
                            "model": "gru",
                            "sample_size": 100,
                            "dry_run": False,
                            "model_param": ["seq_len=1024", "dropout=0.1"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    command = build_job_command(load_training_plan(config_path).jobs[0])

    assert command[0] == sys.executable
    assert Path(command[1]).as_posix() == "ml/timeseries/train.py"
    assert "--sample-size" in command
    assert "100" in command
    assert "--dry-run" not in command
    assert command.count("--model-param") == 2


def test_plan_only_marks_selected_jobs_without_running(tmp_path: Path) -> None:
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    {"name": "vision_job", "task": "vision", "args": {"dry_run": True}},
                    {"name": "vlm_job", "task": "vlm", "args": {"dry_run": True}},
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = load_training_plan(config_path)

    results = run_training_plan(plan, TrainingRunOptions(selectors=("vision",), plan_only=True))

    assert [result.status for result in results] == ["planned", "skipped"]
