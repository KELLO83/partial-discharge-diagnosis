from __future__ import annotations

from service.backend.app.guardrails import review_tool_outputs
from service.backend.app.policy import label_name
from service.backend.app.schemas import TimeSeriesResult, VlmResult


def test_reviewer_completes_when_tool_outputs_agree() -> None:
    decision = review_tool_outputs(_timeseries_result(label_id=3), _vlm_result(label_id=3))

    assert decision.status == "completed"
    assert decision.requires_human_review is False


def test_reviewer_marks_low_confidence_as_needs_review() -> None:
    decision = review_tool_outputs(_timeseries_result(label_id=3, confidence=0.42), _vlm_result(label_id=3))

    assert decision.status == "needs_review"
    assert "confidence" in decision.reason
    assert decision.requires_human_review is True


def test_reviewer_marks_label_disagreement_as_needs_review() -> None:
    decision = review_tool_outputs(_timeseries_result(label_id=2), _vlm_result(label_id=3))

    assert decision.status == "needs_review"
    assert "불일치" in decision.reason


def test_reviewer_marks_invalid_probabilities_as_needs_review() -> None:
    ts_result = _timeseries_result(label_id=3, probabilities={"0": 0.5, "1": 0.5})
    decision = review_tool_outputs(ts_result, _vlm_result(label_id=3))

    assert decision.status == "needs_review"
    assert "확률" in decision.reason


def _timeseries_result(
    label_id: int,
    confidence: float = 0.9,
    probabilities: dict[str, float] | None = None,
) -> TimeSeriesResult:
    if probabilities is None:
        probabilities = {str(idx): 0.025 for idx in range(5)}
        probabilities[str(label_id)] = 0.90
    return TimeSeriesResult(
        model_name="test_ts",
        model_version="test",
        label_id=label_id,
        label_name=label_name(label_id),
        confidence=confidence,
        probabilities=probabilities,
        features={"rms": 1.0},
    )


def _vlm_result(label_id: int, confidence: float = 0.9) -> VlmResult:
    return VlmResult(
        model_name="test_vlm",
        model_version="test",
        label_id=label_id,
        diagnosis=label_name(label_id),
        risk_level="주의",
        confidence=confidence,
        reason="test reason",
        recommended_action="test action",
    )
