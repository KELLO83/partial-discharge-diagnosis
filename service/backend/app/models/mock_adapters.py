from __future__ import annotations

from service.backend.app.application.contracts import (
    TimeSeriesInferenceAdapter,
    TimeSeriesToolInput,
    VisionInferenceAdapter,
    VisionToolInput,
    VlmInferenceAdapter,
    VlmToolInput,
)
from service.backend.app.domain.fusion import time_series_evidence, vision_evidence, vlm_evidence
from service.backend.app.domain.policy import label_name, recommended_action, risk_level
from service.backend.app.schemas import RagResult, SimilarCase, TimeSeriesResult, VisionResult, VlmResult


PRE_MODEL_VERSION = "pre_model_mock"
CORONA_LABEL_ID = 3


class MockTimeSeriesInferenceAdapter(TimeSeriesInferenceAdapter):
    model_name = "mock_patchtst"
    model_version = PRE_MODEL_VERSION

    def run(self, tool_input: TimeSeriesToolInput) -> TimeSeriesResult:
        return run_timeseries_inference(tool_input)


class MockVisionInferenceAdapter(VisionInferenceAdapter):
    model_name = "mock_prpd_small_cnn"
    model_version = PRE_MODEL_VERSION

    def run(self, tool_input: VisionToolInput) -> VisionResult:
        return run_vision_inference(tool_input)


class MockVlmInferenceAdapter(VlmInferenceAdapter):
    model_name = "mock_qwen3_vl_2b"
    model_version = PRE_MODEL_VERSION

    def run(self, tool_input: VlmToolInput) -> VlmResult:
        return run_vlm_inference(tool_input)


def run_timeseries_inference(tool_input: TimeSeriesToolInput | None = None) -> TimeSeriesResult:
    result = TimeSeriesResult(
        model_name=MockTimeSeriesInferenceAdapter.model_name,
        model_version=PRE_MODEL_VERSION,
        label_id=CORONA_LABEL_ID,
        label_name=label_name(CORONA_LABEL_ID),
        confidence=0.87,
        probabilities={"0": 0.02, "1": 0.04, "2": 0.06, "3": 0.87, "4": 0.01},
        features=_timeseries_features(tool_input),
    )
    return result.model_copy(update={"standard_evidence": time_series_evidence(result)})


def run_vision_inference(tool_input: VisionToolInput | None = None) -> VisionResult:
    result = VisionResult(
        model_name=MockVisionInferenceAdapter.model_name,
        model_version=PRE_MODEL_VERSION,
        label_id=CORONA_LABEL_ID,
        label_name=label_name(CORONA_LABEL_ID),
        confidence=0.82,
        probabilities={"0": 0.03, "1": 0.05, "2": 0.07, "3": 0.82, "4": 0.03},
        evidence=_vision_evidence(tool_input),
    )
    return result.model_copy(update={"standard_evidence": vision_evidence(result)})


def run_vlm_inference(tool_input: VlmToolInput) -> VlmResult:
    result = VlmResult(
        model_name=MockVlmInferenceAdapter.model_name,
        model_version=PRE_MODEL_VERSION,
        label_id=CORONA_LABEL_ID,
        diagnosis=label_name(CORONA_LABEL_ID),
        risk_level=risk_level(CORONA_LABEL_ID),
        confidence=_vlm_confidence(tool_input),
        reason=_vlm_reason(tool_input),
        recommended_action=recommended_action(CORONA_LABEL_ID),
    )
    return result.model_copy(update={"standard_evidence": vlm_evidence(result)})


def _timeseries_features(tool_input: TimeSeriesToolInput | None) -> dict[str, float]:
    features = {
        "rms": 30.37,
        "std": 4.96,
        "abs_p99": 39.0,
        "pulse_rate": 0.0069,
        "spectral_energy": 13982100.0,
    }
    if tool_input is not None:
        features["csv_size_marker"] = float(tool_input.csv_path.exists())
    return features


def _vision_evidence(tool_input: VisionToolInput | None) -> dict[str, float | str]:
    evidence: dict[str, float | str] = {
        "phase_uniformity_score": 0.18,
        "phase_localization_score": 0.81,
        "band_like_noise_score": 0.09,
        "ood_score": 0.12,
        "visual_evidence_summary": "PRPD 이미지 근거는 위상 국부화된 코로나 유사 활동을 가리킵니다.",
    }
    if tool_input is not None:
        evidence["image_size_marker"] = float(tool_input.image_path.exists())
    return evidence


def _vlm_confidence(tool_input: VlmToolInput) -> float:
    ts_result = tool_input.timeseries_result
    vision_result = tool_input.vision_result
    if ts_result is not None and vision_result is not None and ts_result.label_id == vision_result.label_id:
        return 0.90
    if ts_result is not None or vision_result is not None:
        return 0.87
    return 0.84


def _vlm_reason(tool_input: VlmToolInput) -> str:
    top_document = _top_document_title(tool_input.rag_result)
    case_text = _similar_case_reference(tool_input.rag_result)
    return (
        f"{tool_input.safe_metadata.equipment_name} 설비의 PRPD 이미지와 "
        f"시계열/비전 요약 정보가 코로나 방전 패턴과 일치하며, 지식 검색 근거({top_document})와도 정합됩니다."
        f"{case_text}"
    )


def _top_document_title(rag_result: RagResult | None) -> str:
    if rag_result is None or not rag_result.documents:
        return "PD 규칙서"
    return rag_result.documents[0].title


def _similar_case_reference(rag_result: RagResult | None) -> str:
    if rag_result is None or not rag_result.similar_cases:
        return ""
    return _case_text(rag_result.similar_cases[0])


def _case_text(case: SimilarCase) -> str:
    return f" 유사 사례({case.sample_id}, {case.label_name}, 유사도 {case.similarity:.2f})도 참고되었습니다."
