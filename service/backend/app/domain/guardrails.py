from __future__ import annotations

from dataclasses import dataclass

from service.backend.app.domain.policy import MIN_CONFIDENCE, PROBABILITY_SUM_TOLERANCE, valid_label_id
from service.backend.app.schemas import RagResult, TimeSeriesResult, VisionResult, VlmResult


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    status: str
    reason: str
    requires_human_review: bool


def review_tool_outputs(
    ts_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
    vlm_result: VlmResult | None,
    rag_result: RagResult | None = None,
) -> ReviewDecision:
    if ts_result is not None:
        ts_issue = validate_timeseries_result(ts_result)
        if ts_issue:
            return _needs_review(ts_issue)
    if vision_result is not None:
        vision_issue = validate_vision_result(vision_result)
        if vision_issue:
            return _needs_review(vision_issue)
    if vlm_result is not None:
        vlm_issue = validate_vlm_result(vlm_result)
        if vlm_issue:
            return _needs_review(vlm_issue)
    if rag_result is not None:
        rag_issue = validate_rag_result(rag_result)
        if rag_issue:
            return _needs_review(rag_issue)
    disagreement = _label_disagreement(ts_result, vision_result, vlm_result)
    if disagreement is not None:
        return _needs_review(disagreement)
    if ts_result is None and vision_result is None and vlm_result is None:
        return ReviewDecision("rejected", "검토할 도구 출력이 없습니다.", False)
    return ReviewDecision("completed", "모델 및 지식 검색 근거가 일관되어 최종 진단을 확정했습니다.", False)


def validate_timeseries_result(result: TimeSeriesResult) -> str | None:
    if not valid_label_id(result.label_id):
        return f"시계열 라벨 ID가 허용 범위를 벗어났습니다: {result.label_id}"
    if result.confidence < MIN_CONFIDENCE:
        return f"시계열 신뢰도가 기준 미만입니다: {result.confidence:.2f}"
    probability_issue = _validate_probabilities(result.probabilities)
    if probability_issue is not None:
        return f"시계열 확률 출력이 유효하지 않습니다: {probability_issue}"
    return None


def validate_vision_result(result: VisionResult) -> str | None:
    if not valid_label_id(result.label_id):
        return f"비전 라벨 ID가 허용 범위를 벗어났습니다: {result.label_id}"
    if result.confidence < MIN_CONFIDENCE:
        return f"비전 신뢰도가 기준 미만입니다: {result.confidence:.2f}"
    probability_issue = _validate_probabilities(result.probabilities)
    if probability_issue is not None:
        return f"비전 확률 출력이 유효하지 않습니다: {probability_issue}"
    return None


def validate_vlm_result(result: VlmResult) -> str | None:
    if not valid_label_id(result.label_id):
        return f"VLM 라벨 ID가 허용 범위를 벗어났습니다: {result.label_id}"
    if result.confidence < MIN_CONFIDENCE:
        return f"VLM 신뢰도가 기준 미만입니다: {result.confidence:.2f}"
    if result.reason.strip() == "":
        return "VLM 판단 사유가 비어 있습니다."
    if result.recommended_action.strip() == "":
        return "VLM 권고 조치가 비어 있습니다."
    return None


def validate_rag_result(result: RagResult) -> str | None:
    if result.query.strip() == "":
        return "지식 검색 질의가 비어 있습니다."
    if not result.documents:
        return "지식 검색 근거 문서가 검색되지 않았습니다."
    if max(document.relevance for document in result.documents) < MIN_CONFIDENCE:
        return "지식 검색 상위 근거의 관련도가 기준 미만입니다."
    return None


def _label_disagreement(
    ts_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
    vlm_result: VlmResult | None,
) -> str | None:
    labels: dict[str, tuple[int, str]] = {}
    if ts_result is not None:
        labels["시계열"] = (ts_result.label_id, ts_result.label_name)
    if vision_result is not None:
        labels["비전"] = (vision_result.label_id, vision_result.label_name)
    if vlm_result is not None:
        labels["VLM"] = (vlm_result.label_id, vlm_result.diagnosis)
    if len({label_id for label_id, _ in labels.values()}) <= 1:
        return None
    label_text = ", ".join(
        f"{source}={label_name}({label_id})"
        for source, (label_id, label_name) in labels.items()
    )
    return f"모델 간 예측 라벨이 불일치합니다: {label_text}"


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
