from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

run_dry_inference: Any = getattr(importlib.import_module("ml.vlm.scripts.run_inference"), "run_dry_inference")


def test_run_dry_inference_writes_prediction_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    target = {
        "label_id": 1,
        "diagnosis": "노이즈",
        "risk_level": "낮음",
        "reason": "x",
        "recommended_action": "y",
    }
    dataset_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "label_id": 1,
                "images": ["sample.png"],
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": "sample.png"}]},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_dry_inference(
        dataset_path=dataset_path,
        output_path=output_path,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        load_in_4bit=True,
        limit=1,
    )

    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["sample_id"] == "sample-1"
    assert row["label_id"] == 1
    assert row["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert row["attn_implementation"] == "sdpa"
    assert row["mode"] == "dry_run_target_echo"
    assert row["parsed_json"]["label_id"] == 1
    assert json.loads(row["raw_text"])["label_id"] == 1


def test_run_dry_inference_supports_single_index(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "sample_id": "sample-1",
            "label_id": 1,
            "images": ["sample-1.png"],
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": "sample-1.png"}]},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "label_id": 1,
                            "diagnosis": "노이즈",
                            "risk_level": "낮음",
                            "reason": "x",
                            "recommended_action": "y",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        },
        {
            "sample_id": "sample-2",
            "label_id": 2,
            "images": ["sample-2.png"],
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": "sample-2.png"}]},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "label_id": 2,
                            "diagnosis": "표면방전",
                            "risk_level": "주의",
                            "reason": "x",
                            "recommended_action": "y",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    rows_written = run_dry_inference(
        dataset_path=dataset_path,
        output_path=output_path,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        load_in_4bit=True,
        limit=None,
        index=1,
    )

    output_row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert rows_written == 1
    assert output_row["sample_id"] == "sample-2"
