from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from ml.vlm.src.schema import EvaluationMetrics, FORBIDDEN_PROMPT_FIELDS


REQUIRED_OUTPUT_FIELDS: frozenset[str] = frozenset(
    {"label_id", "diagnosis", "risk_level", "reason", "recommended_action"}
)


def evaluate_predictions(predictions_path: Path) -> EvaluationMetrics:
    y_true: list[int] = []
    valid_y_true: list[int] = []
    y_pred: list[int] = []
    parse_failures = 0
    schema_valid = 0
    hallucinated_field_count = 0
    forbidden_field_hit_count = 0
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        record = json.loads(line)
        y_true.append(int(record["label_id"]))
        raw_text = str(record.get("raw_text", ""))
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parse_failures += 1
            continue
        if _prediction_schema_valid(parsed):
            schema_valid += 1
            valid_y_true.append(int(record["label_id"]))
            y_pred.append(int(parsed["label_id"]))
            extra_fields = set(parsed) - REQUIRED_OUTPUT_FIELDS
            hallucinated_field_count += len(extra_fields)
            forbidden_field_hit_count += sum(1 for field in extra_fields if field in FORBIDDEN_PROMPT_FIELDS)
    parsed_count = len(y_pred)
    total = len(y_true)
    label_accuracy = _accuracy(valid_y_true, y_pred)
    return EvaluationMetrics(
        n_rows=total,
        json_parse_success_rate=parsed_count / total if total else 0.0,
        schema_validity_rate=schema_valid / total if total else 0.0,
        label_accuracy=label_accuracy,
        macro_f1=_macro_f1(valid_y_true, y_pred),
        parse_failures=parse_failures,
        confusion_matrix=_confusion_matrix(valid_y_true, y_pred),
        hallucinated_field_count=hallucinated_field_count,
        forbidden_field_hit_count=forbidden_field_hit_count,
    )


def _prediction_schema_valid(parsed: object) -> bool:
    if not isinstance(parsed, dict):
        return False
    return REQUIRED_OUTPUT_FIELDS.issubset(parsed)


def _accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_pred:
        return 0.0
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)
    return correct / len(y_pred)


def _macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    if not y_pred:
        return 0.0
    scores: list[float] = []
    for label_id in range(5):
        tp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label_id and predicted == label_id)
        fp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected != label_id and predicted == label_id)
        fn = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label_id and predicted != label_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _confusion_matrix(y_true: list[int], y_pred: list[int]) -> tuple[tuple[int, ...], ...]:
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    for expected, predicted in zip(y_true, y_pred):
        if 0 <= expected < 5 and 0 <= predicted < 5:
            matrix[expected][predicted] += 1
    return tuple(tuple(row) for row in matrix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_predictions(args.predictions)
    payload = {
        "n_rows": metrics.n_rows,
        "json_parse_success_rate": metrics.json_parse_success_rate,
        "schema_validity_rate": metrics.schema_validity_rate,
        "label_accuracy": metrics.label_accuracy,
        "macro_f1": metrics.macro_f1,
        "parse_failures": metrics.parse_failures,
        "confusion_matrix": metrics.confusion_matrix,
        "hallucinated_field_count": metrics.hallucinated_field_count,
        "forbidden_field_hit_count": metrics.forbidden_field_hit_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
