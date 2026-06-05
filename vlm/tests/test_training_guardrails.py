from __future__ import annotations

import pytest

import json
from pathlib import Path

from vlm.scripts.train_sft import TrainingRiskError, build_training_config, write_dry_run_artifacts


def test_full_finetune_is_blocked_without_risk_override() -> None:
    with pytest.raises(TrainingRiskError):
        build_training_config(
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            dataset="dataset.jsonl",
            output_dir="adapter",
            load_in_4bit=True,
            full_finetune=True,
            risk_override=False,
            max_steps=1,
        )


def test_default_training_config_uses_8gb_qlora_guardrails() -> None:
    config = build_training_config(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        dataset="dataset.jsonl",
        output_dir="adapter",
        load_in_4bit=True,
        full_finetune=False,
        risk_override=False,
        max_steps=10,
    )

    assert config.load_in_4bit is True
    assert config.batch_size == 1
    assert config.gradient_checkpointing is True
    assert config.train_vision_tower is False


def test_training_dry_run_writes_config_and_summary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "adapter"
    config = build_training_config(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        dataset=str(dataset_path),
        output_dir=str(output_dir),
        load_in_4bit=True,
        full_finetune=False,
        risk_override=False,
        max_steps=10,
    )

    summary_path = write_dry_run_artifacts(config)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run_ready"
    assert payload["config"]["load_in_4bit"] is True
    assert payload["train_rows"] == 1
    assert payload["adapter_saved"] is False
    assert (output_dir / "training_config.json").exists()
