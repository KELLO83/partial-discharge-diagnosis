from __future__ import annotations

import json

from ml.vlm.src.schema import ManifestVlmRow, PD_LABELS_KO, TimeSeriesContext, VisionContext


def build_prompt_text(
    row: ManifestVlmRow,
    context: TimeSeriesContext | None,
    vision_context: VisionContext | None = None,
) -> str:
    ts_context = context if context is not None else TimeSeriesContext.unavailable(row.sample_id)
    image_context = vision_context if vision_context is not None else VisionContext.unavailable(row.sample_id)
    metadata_lines = [
        f"equipment_name: {row.equipment_name}",
        f"equipment_rated_voltage: {row.equipment_rated_voltage}",
        f"equipment_rated_current: {row.equipment_rated_current}",
        f"insulator_type: {row.insulator_type}",
        f"insulator_name: {row.insulator_name}",
        f"sensor_type: {row.sensor_type}",
        f"temperature: {row.temperature}",
        f"humidity: {row.humidity}",
        f"clearance_distance: {row.clearance_distance}",
    ]
    ts_lines = [
        f"ts_model_name: {ts_context.ts_model_name}",
        f"ts_pred_class: {_format_optional(ts_context.ts_pred_label_id)}",
        f"ts_confidence: {_format_optional(ts_context.ts_confidence)}",
        "ts_probabilities: "
        f"[{_format_optional(ts_context.ts_prob_0)}, "
        f"{_format_optional(ts_context.ts_prob_1)}, "
        f"{_format_optional(ts_context.ts_prob_2)}, "
        f"{_format_optional(ts_context.ts_prob_3)}, "
        f"{_format_optional(ts_context.ts_prob_4)}]",
        f"rms: {_format_optional(ts_context.rms)}",
        f"std: {_format_optional(ts_context.std)}",
        f"abs_p99: {_format_optional(ts_context.abs_p99)}",
        f"pulse_rate: {_format_optional(ts_context.pulse_rate)}",
        f"spectral_energy: {_format_optional(ts_context.spectral_energy)}",
    ]
    vision_lines = [
        f"vision_model_name: {image_context.vision_model_name}",
        f"vision_pred_class: {_format_optional(image_context.vision_pred_label_id)}",
        f"vision_confidence: {_format_optional(image_context.vision_confidence)}",
        "vision_probabilities: "
        f"[{_format_optional(image_context.vision_prob_0)}, "
        f"{_format_optional(image_context.vision_prob_1)}, "
        f"{_format_optional(image_context.vision_prob_2)}, "
        f"{_format_optional(image_context.vision_prob_3)}, "
        f"{_format_optional(image_context.vision_prob_4)}]",
    ]
    return "\n".join(
        [
            "당신은 산업 전력설비 부분방전 진단 보조 모델입니다.",
            "제공된 PRPD 이미지와 텍스트 정보만 사용하세요.",
            "추측하지 말고 반드시 JSON만 출력하세요.",
            "",
            "[설비 및 환경 정보]",
            *metadata_lines,
            "",
            "[시계열 모델 분석]",
            *ts_lines,
            "",
            "[비전 모델 분석]",
            *vision_lines,
            "",
            "[출력 형식]",
            "정답 분류 번호, 진단명, 위험도, 판단 근거, 권장 조치를 포함한 JSON 객체",
        ]
    )


def build_target_json(row: ManifestVlmRow) -> str:
    diagnosis = PD_LABELS_KO[row.label_id]
    target = {
        "label_id": row.label_id,
        "diagnosis": diagnosis,
        "risk_level": _risk_level(row.label_id),
        "reason": f"PRPD 이미지와 제공된 시계열 요약 정보가 {diagnosis} 진단과 일치합니다.",
        "recommended_action": _recommended_action(row.label_id),
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def _format_optional(value: float | int | None) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _risk_level(label_id: int) -> str:
    match label_id:
        case 0:
            return "낮음"
        case 1:
            return "낮음"
        case 2:
            return "주의"
        case 3:
            return "주의"
        case 4:
            return "주의"
        case _:
            return "확인필요"


def _recommended_action(label_id: int) -> str:
    match label_id:
        case 0:
            return "정상 상태로 판단되며 정기 모니터링을 유지하세요."
        case 1:
            return "센서 접촉 상태와 주변 전자기 간섭 가능성을 점검하세요."
        case 2:
            return "절연체 표면 오염과 트래킹 흔적을 점검하세요."
        case 3:
            return "전계 집중 부위와 고전압 접속부를 점검하세요."
        case 4:
            return "절연체 내부 결함 가능성을 고려해 정밀 진단을 진행하세요."
        case _:
            return "추가 데이터 확인 후 재진단하세요."
