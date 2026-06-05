from __future__ import annotations

from service.backend.app.schemas import MetadataInput, TimeSeriesResult, VlmResult


LABELS: dict[int, str] = {
    0: "정상",
    1: "노이즈",
    2: "표면 방전",
    3: "코로나 방전",
    4: "보이드 방전",
}


def run_timeseries_inference() -> TimeSeriesResult:
    return TimeSeriesResult(
        model_name="mock_patchtst",
        label_id=3,
        label_name=LABELS[3],
        confidence=0.87,
        probabilities={"0": 0.02, "1": 0.04, "2": 0.06, "3": 0.87, "4": 0.01},
        features={
            "rms": 30.37,
            "std": 4.96,
            "abs_p99": 39.0,
            "pulse_rate": 0.0069,
            "spectral_energy": 13982100.0,
        },
    )


def run_vlm_inference(metadata: MetadataInput, ts_result: TimeSeriesResult | None) -> VlmResult:
    confidence = 0.84 if ts_result is None else 0.89
    reason = (
        f"{metadata.equipment_name} 설비의 PRPD 이미지와 "
        "제공된 요약 정보가 코로나 방전 패턴과 일치합니다."
    )
    return VlmResult(
        model_name="mock_qwen3_vl_2b",
        label_id=3,
        diagnosis=LABELS[3],
        risk_level="주의",
        confidence=confidence,
        reason=reason,
        recommended_action="고전압 접속부와 전계 집중 부위를 점검하고 추세를 모니터링하세요.",
    )
