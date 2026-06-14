from __future__ import annotations

import json
from pathlib import Path

from ml.vlm.scripts.evaluate_outputs import evaluate_predictions


def test_evaluate_predictions_counts_parse_success_and_accuracy(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "label_id": 1,
                        "raw_text": "{\"label_id\":1,\"diagnosis\":\"노이즈\",\"risk_level\":\"낮음\",\"reason\":\"x\",\"recommended_action\":\"y\"}",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "label_id": 2,
                        "raw_text": "not json",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_predictions(predictions_path)

    assert metrics.n_rows == 2
    assert metrics.json_parse_success_rate == 0.5
    assert metrics.label_accuracy == 1.0
    assert metrics.parse_failures == 1
    assert metrics.confusion_matrix[1][1] == 1
    assert metrics.hallucinated_field_count == 0


def test_evaluate_predictions_aligns_labels_after_parse_failure(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    valid_prediction = {
        "label_id": 4,
        "diagnosis": "보이드방전",
        "risk_level": "높음",
        "reason": "x",
        "recommended_action": "y",
    }
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps({"label_id": 0, "raw_text": "not json"}, ensure_ascii=False),
                json.dumps(
                    {"label_id": 4, "raw_text": json.dumps(valid_prediction, ensure_ascii=False)},
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_predictions(predictions_path)

    assert metrics.json_parse_success_rate == 0.5
    assert metrics.label_accuracy == 1.0


def test_evaluate_predictions_counts_hallucinated_and_forbidden_fields(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    prediction = {
        "label_id": 1,
        "diagnosis": "노이즈",
        "risk_level": "낮음",
        "reason": "x",
        "recommended_action": "y",
        "label_name": "leak",
        "extra_note": "unused",
    }
    predictions_path.write_text(
        json.dumps({"label_id": 1, "raw_text": json.dumps(prediction, ensure_ascii=False)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_predictions(predictions_path)

    assert metrics.hallucinated_field_count == 2
    assert metrics.forbidden_field_hit_count == 1
