from __future__ import annotations

from service.backend.app.domain.guardrails import review_tool_outputs
from service.backend.app.domain.policy import label_name
from service.backend.app.schemas import RagDocument, RagResult, TimeSeriesResult, VisionResult, VlmResult


def test_reviewer_completes_when_tool_outputs_agree() -> None:
    decision = review_tool_outputs(_timeseries_result(label_id=3), _vision_result(label_id=3), _vlm_result(label_id=3))

    assert decision.status == "completed"
    assert decision.requires_human_review is False


def test_reviewer_marks_low_confidence_as_needs_review() -> None:
    decision = review_tool_outputs(_timeseries_result(label_id=3, confidence=0.42), _vision_result(label_id=3), _vlm_result(label_id=3))

    assert decision.status == "needs_review"
    assert "신뢰도" in decision.reason
    assert decision.requires_human_review is True


def test_reviewer_marks_label_disagreement_as_needs_review() -> None:
    decision = review_tool_outputs(_timeseries_result(label_id=2), _vision_result(label_id=2), _vlm_result(label_id=3))

    assert decision.status == "needs_review"
    assert "불일치" in decision.reason
    assert "시계열=표면 방전(2)" in decision.reason
    assert "비전=표면 방전(2)" in decision.reason
    assert "VLM=코로나 방전(3)" in decision.reason


def test_reviewer_marks_time_series_vision_disagreement_as_needs_review() -> None:
    decision = review_tool_outputs(_timeseries_result(label_id=2), _vision_result(label_id=3), None)

    assert decision.status == "needs_review"
    assert "불일치" in decision.reason
    assert "시계열=표면 방전(2)" in decision.reason
    assert "비전=코로나 방전(3)" in decision.reason


def test_reviewer_marks_invalid_probabilities_as_needs_review() -> None:
    ts_result = _timeseries_result(label_id=3, probabilities={"0": 0.5, "1": 0.5})
    decision = review_tool_outputs(ts_result, _vision_result(label_id=3), _vlm_result(label_id=3))

    assert decision.status == "needs_review"
    assert "확률" in decision.reason


def test_reviewer_marks_empty_rag_result_as_needs_review() -> None:
    decision = review_tool_outputs(
        _timeseries_result(label_id=3),
        _vision_result(label_id=3),
        _vlm_result(label_id=3),
        _rag_result(documents=[]),
    )

    assert decision.status == "needs_review"
    assert "지식 검색" in decision.reason


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


def _vision_result(
    label_id: int,
    confidence: float = 0.9,
    probabilities: dict[str, float] | None = None,
) -> VisionResult:
    if probabilities is None:
        probabilities = {str(idx): 0.025 for idx in range(5)}
        probabilities[str(label_id)] = 0.90
    return VisionResult(
        model_name="test_vision",
        model_version="test",
        label_id=label_id,
        label_name=label_name(label_id),
        confidence=confidence,
        probabilities=probabilities,
        evidence={"phase_uniformity_score": 0.2},
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


def _rag_result(documents: list[RagDocument] | None = None) -> RagResult:
    if documents is None:
        documents = [
            RagDocument(
                document_id="test-rule",
                title="test rule",
                source="test",
                excerpt="test",
                relevance=0.9,
            )
        ]
    return RagResult(
        retriever_name="test_rag",
        retriever_version="test",
        query="test query",
        documents=documents,
    )
