from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from vlm.scripts.build_instruction_dataset import build_instruction_dataset
from vlm.scripts.validate_instruction_dataset import validate_jsonl


def _write_manifest(path: Path, image_path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "image_path",
                "timeseries_path",
                "json_path",
                "label_id",
                "label_name",
                "equipment_name",
                "equipment_rated_voltage",
                "equipment_rated_current",
                "insulator_type",
                "insulator_name",
                "sensor_type",
                "temperature",
                "humidity",
                "clearance_distance",
                "defect_details",
                "defect_nums",
                "max_discharge_value",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "noise_secret_id",
                "split": "train",
                "image_path": str(image_path),
                "timeseries_path": "Train/노이즈/sample.csv",
                "json_path": "Train/노이즈/sample.json",
                "label_id": "1",
                "label_name": "노이즈",
                "equipment_name": "ACSR-OC",
                "equipment_rated_voltage": "22900V",
                "equipment_rated_current": "600A",
                "insulator_type": "고체",
                "insulator_name": "XLPE",
                "sensor_type": "HFCT",
                "temperature": "19",
                "humidity": "66",
                "clearance_distance": "1000mm",
                "defect_details": "secret defect",
                "defect_nums": "1",
                "max_discharge_value": "777",
            }
        )


def test_build_dataset_writes_image_text_messages_when_manifest_valid(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "dataset.jsonl"
    _write_manifest(manifest_path, image_path)

    summary = build_instruction_dataset(
        manifest_path=manifest_path,
        output_path=output_path,
        sample_size=1,
        ts_context_path=None,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary.rows_written == 1
    assert rows[0]["images"] == [str(image_path)]
    assert rows[0]["messages"][0]["content"][0]["type"] == "image"
    assert rows[0]["messages"][0]["content"][1]["type"] == "text"
    assert json.loads(rows[0]["messages"][1]["content"])["label_id"] == 1


def test_build_dataset_excludes_leakage_fields_from_prompt(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "dataset.jsonl"
    _write_manifest(manifest_path, image_path)

    build_instruction_dataset(
        manifest_path=manifest_path,
        output_path=output_path,
        sample_size=1,
        ts_context_path=None,
    )

    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    prompt = row["messages"][0]["content"][1]["text"]
    assert "ACSR-OC" in prompt
    assert "noise_secret_id" not in prompt
    assert "노이즈" not in prompt
    assert "secret defect" not in prompt
    assert "777" not in prompt
    assert "sample.csv" not in prompt


def test_validate_dataset_rejects_prompt_leakage(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_path)
    jsonl_path = tmp_path / "leaky.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "split": "train",
                "label_id": 1,
                "images": [str(image_path)],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "label_id: 1 defect_details"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "{\"label_id\":1,\"diagnosis\":\"노이즈\",\"risk_level\":\"낮음\",\"reason\":\"x\",\"recommended_action\":\"y\"}",
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_jsonl(jsonl_path)

    assert report.leakage_hits > 0
    assert report.valid is False


def test_build_dataset_fails_when_image_missing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "dataset.jsonl"
    _write_manifest(manifest_path, tmp_path / "missing.png")

    with pytest.raises(FileNotFoundError):
        build_instruction_dataset(
            manifest_path=manifest_path,
            output_path=output_path,
            sample_size=1,
            ts_context_path=None,
        )
