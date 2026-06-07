from __future__ import annotations

from dataclasses import dataclass

from service.backend.app.policy import MIN_CONFIDENCE, PROBABILITY_SUM_TOLERANCE, valid_label_id
from service.backend.app.schemas import TimeSeriesResult, VlmResult


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    status: str
    reason: str
    requires_human_review: bool


def review_tool_outputs(ts_result: TimeSeriesResult | None, vlm_result: VlmResult | None) -> ReviewDecision:
    if ts_result is not None:
        ts_issue = validate_timeseries_result(ts_result)
        if ts_issue:
            return _needs_review(ts_issue)
    if vlm_result is not None:
        vlm_issue = validate_vlm_result(vlm_result)
        if vlm_issue:
            return _needs_review(vlm_issue)
    if ts_result is not None and vlm_result is not None and ts_result.label_id != vlm_result.label_id:
        return _needs_review("시계열 모델과 VLM의 예측 라벨이 불일치합니다.")
    if ts_result is None and vlm_result is None:
        return ReviewDecision("rejected", "검토할 도구 출력이 없습니다.", False)
    return ReviewDecision("completed", "tool 기반 추론 결과가 일관되어 최종 진단을 확정했습니다.", False)


def validate_timeseries_result(result: TimeSeriesResult) -> str | None:
    if not valid_label_id(result.label_id):
        return f"시계열 label_id가 허용 범위를 벗어났습니다: {result.label_id}"
    if result.confidence < MIN_CONFIDENCE:
        return f"시계열 confidence가 기준 미만입니다: {result.confidence:.2f}"
    probability_issue = _validate_probabilities(result.probabilities)
    if probability_issue is not None:
        return f"시계열 확률 출력이 유효하지 않습니다: {probability_issue}"
    return None


def validate_vlm_result(result: VlmResult) -> str | None:
    if not valid_label_id(result.label_id):
        return f"VLM label_id가 허용 범위를 벗어났습니다: {result.label_id}"
    if result.confidence < MIN_CONFIDENCE:
        return f"VLM confidence가 기준 미만입니다: {result.confidence:.2f}"
    if result.reason.strip() == "":
        return "VLM reason이 비어 있습니다."
    if result.recommended_action.strip() == "":
        return "VLM recommended_action이 비어 있습니다."
    return None


def _validate_probabilities(probabilities: dict[str, float]) -> str | None:
    expected_keys = {str(label_id) for label_id in range(5)}
    if set(probabilities) != expected_keys:
        return "0~4 클래스 확률이 모두 필요합니다."
    values = list(probabilities.values())
    if any(value < 0.0 or value > 1.0 for value in values):
        return "확률값은 0과 1 사이여야 합니다."
    probability_sum = sum(values)
    if abs(probability_sum - 1.0) > PROBABILITY_SUM_TOLERANCE:
        return f"확률 합이 1에서 벗어났습니다: {probability_sum:.4f}"
    return None


def _needs_review(reason: str) -> ReviewDecision:
    return ReviewDecision("needs_review", reason, True)
