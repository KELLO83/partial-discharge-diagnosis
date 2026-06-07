from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from service.backend.app.guardrails import review_tool_outputs
from service.backend.app.policy import label_name
from service.backend.app.schemas import TimeSeriesResult, VlmResult


@dataclass(frozen=True, slots=True)
class OfflineEvaluationSummary:
    rows: int
    completed: int
    needs_review: int
    rejected: int
    ts_vlm_agreement_rate: float


def run_offline_mock_evaluation(manifest_path: Path, output_path: Path, sample_size: int | None = None) -> OfflineEvaluationSummary:
    rows = _read_manifest(manifest_path)
    selected_rows = rows[:sample_size] if sample_size is not None else rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_rows = [_evaluate_row(row) for row in selected_rows]
    _write_jsonl(output_path, result_rows)
    return _summarize(result_rows)


def _evaluate_row(row: dict[str, str]) -> dict[str, object]:
    label_id = int(row["label_id"])
    ts_result = _mock_timeseries_result(label_id)
    vlm_result = _mock_vlm_result(label_id)
    decision = review_tool_outputs(ts_result, vlm_result)
    return {
        "sample_id": row.get("sample_id", ""),
        "target_label_id": label_id,
        "ts_label_id": ts_result.label_id,
        "vlm_label_id": vlm_result.label_id,
        "status": decision.status,
        "reason": decision.reason,
        "requires_human_review": decision.requires_human_review,
    }


def _mock_timeseries_result(label_id: int) -> TimeSeriesResult:
    probabilities = {str(idx): 0.025 for idx in range(5)}
    probabilities[str(label_id)] = 0.90
    return TimeSeriesResult(
        model_name="offline_mock_timeseries",
        model_version="pre_model_mock",
        label_id=label_id,
        label_name=label_name(label_id),
        confidence=0.90,
        probabilities=probabilities,
        features={"rms": 0.0, "std": 0.0, "abs_p99": 0.0, "pulse_rate": 0.0, "spectral_energy": 0.0},
    )


def _mock_vlm_result(label_id: int) -> VlmResult:
    return VlmResult(
        model_name="offline_mock_vlm",
        model_version="pre_model_mock",
        label_id=label_id,
        diagnosis=label_name(label_id),
        risk_level="낮음" if label_id < 2 else "주의",
        confidence=0.90,
        reason="오프라인 mock 평가를 위한 결정론적 VLM 출력입니다.",
        recommended_action="모델 연결 전 workflow 계약 검증을 수행하세요.",
    )


def _summarize(rows: list[dict[str, object]]) -> OfflineEvaluationSummary:
    total = len(rows)
    completed = sum(1 for row in rows if row["status"] == "completed")
    needs_review = sum(1 for row in rows if row["status"] == "needs_review")
    rejected = sum(1 for row in rows if row["status"] == "rejected")
    agreements = sum(1 for row in rows if row["ts_label_id"] == row["vlm_label_id"])
    return OfflineEvaluationSummary(
        rows=total,
        completed=completed,
        needs_review=needs_review,
        rejected=rejected,
        ts_vlm_agreement_rate=agreements / total if total else 0.0,
    )


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summary_to_json(summary: OfflineEvaluationSummary) -> str:
    return json.dumps(asdict(summary), ensure_ascii=False, indent=2)
