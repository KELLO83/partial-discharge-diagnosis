from __future__ import annotations

from service.backend.app.policy import label_name, recommended_action, risk_level
from service.backend.app.schemas import MetadataInput, TimeSeriesResult, VlmResult
from service.backend.app.tool_contracts import TimeSeriesInferenceAdapter, TimeSeriesToolInput, VlmInferenceAdapter, VlmToolInput


class MockTimeSeriesInferenceAdapter(TimeSeriesInferenceAdapter):
    model_name = "mock_patchtst"
    model_version = "pre_model_mock"

    def run(self, tool_input: TimeSeriesToolInput) -> TimeSeriesResult:
        return run_timeseries_inference(tool_input)


class MockVlmInferenceAdapter(VlmInferenceAdapter):
    model_name = "mock_qwen3_vl_2b"
    model_version = "pre_model_mock"

    def run(self, tool_input: VlmToolInput) -> VlmResult:
        return run_vlm_inference(tool_input.safe_metadata, tool_input.timeseries_result)


def run_timeseries_inference(tool_input: TimeSeriesToolInput | None = None) -> TimeSeriesResult:
    features = {
        "rms": 30.37,
        "std": 4.96,
        "abs_p99": 39.0,
        "pulse_rate": 0.0069,
        "spectral_energy": 13982100.0,
    }
    if tool_input is not None:
        features["csv_size_marker"] = float(tool_input.csv_path.exists())
    return TimeSeriesResult(
        model_name="mock_patchtst",
        model_version="pre_model_mock",
        label_id=3,
        label_name=label_name(3),
        confidence=0.87,
        probabilities={"0": 0.02, "1": 0.04, "2": 0.06, "3": 0.87, "4": 0.01},
        features=features,
    )


def run_vlm_inference(metadata: MetadataInput, ts_result: TimeSeriesResult | None) -> VlmResult:
    confidence = 0.84 if ts_result is None else 0.89
    reason = (
        f"{metadata.equipment_name} 설비의 PRPD 이미지와 "
        "제공된 요약 정보가 코로나 방전 패턴과 일치합니다."
    )
    return VlmResult(
        model_name="mock_qwen3_vl_2b",
        model_version="pre_model_mock",
        label_id=3,
        diagnosis=label_name(3),
        risk_level=risk_level(3),
        confidence=confidence,
        reason=reason,
        recommended_action=recommended_action(3),
    )


time_series_adapter = MockTimeSeriesInferenceAdapter()
vlm_adapter = MockVlmInferenceAdapter()
