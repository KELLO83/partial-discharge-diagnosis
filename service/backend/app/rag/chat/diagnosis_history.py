from __future__ import annotations

import re
from dataclasses import dataclass

from service.backend.app.domain.policy import label_name as policy_label_name
from service.backend.app.rag.chat.constants import ANSWER_MODE_DIAGNOSIS_HISTORY, LOCAL_DIAGNOSIS_HISTORY_MODEL
from service.backend.app.rag.chat.models import DiagnosisHistoryStore, RagChatInput
from service.backend.app.schemas import DiagnosisDetailResponse, RagChatResponse, TraceResponse


DIAGNOSIS_ID_PATTERN = re.compile(r"\b(?:diag_[0-9a-z_]+|demo_[0-9a-z_]+)\b", re.IGNORECASE)
REASON_LABEL_ID_PATTERN = re.compile(r"(시계열|비전|VLM)=(\d+)")


@dataclass(frozen=True, slots=True)
class DiagnosisSignal:
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class DiagnosisHistoryLookup:
    diagnosis_id: str
    detail: DiagnosisDetailResponse | None


def answer_diagnosis_history_question(
    chat_input: RagChatInput,
    store: DiagnosisHistoryStore | None,
) -> RagChatResponse | None:
    lookup = lookup_diagnosis_history_question(chat_input, store)
    if lookup is None:
        return None
    if lookup.detail is None:
        return RagChatResponse(
            answer=missing_diagnosis_answer(lookup.diagnosis_id),
            documents=[],
            model=LOCAL_DIAGNOSIS_HISTORY_MODEL,
            ready=True,
            answer_mode=ANSWER_MODE_DIAGNOSIS_HISTORY,
        )
    return RagChatResponse(
        answer=diagnosis_history_answer(detail),
        documents=[],
        model=LOCAL_DIAGNOSIS_HISTORY_MODEL,
        ready=True,
        answer_mode=ANSWER_MODE_DIAGNOSIS_HISTORY,
    )


def lookup_diagnosis_history_question(
    chat_input: RagChatInput,
    store: DiagnosisHistoryStore | None,
) -> DiagnosisHistoryLookup | None:
    diagnosis_id = extract_diagnosis_id(chat_input.question)
    if diagnosis_id is None:
        return None
    active_store = store or default_diagnosis_store()
    return DiagnosisHistoryLookup(diagnosis_id, active_store.detail(diagnosis_id))


def extract_diagnosis_id(question: str) -> str | None:
    match = DIAGNOSIS_ID_PATTERN.search(question)
    return match.group(0) if match else None


def default_diagnosis_store() -> DiagnosisHistoryStore:
    from service.backend.app.infrastructure.store import trace_store

    return trace_store


def missing_diagnosis_answer(diagnosis_id: str) -> str:
    return "\n".join(
        [
            "핵심 판단:",
            f"- 진단 이력 {diagnosis_id}를 현재 저장소에서 찾지 못했습니다.",
            "판단 근거:",
            "- 서버 재시작 전 메모리 저장 이력이거나 삭제된 테스트 이력일 수 있습니다.",
            "- PostgreSQL diagnosis.records에 저장된 진단 ID만 조회할 수 있습니다.",
            "참고 사례:",
            "- 진단 이력 화면에서 존재하는 ID를 다시 선택해 질문하세요.",
        ]
    )


def diagnosis_history_answer(detail: DiagnosisDetailResponse) -> str:
    diagnosis = detail.diagnosis
    final_label = diagnosis.diagnosis or fusion_final_label(detail.trace)
    confidence = diagnosis.confidence if diagnosis.confidence is not None else fusion_confidence(detail.trace)
    signals = trace_signals(detail.trace, final_label)
    input_summary = input_summary_lines(detail.trace)
    reference_cases = similar_case_titles(detail.trace, final_label)
    reason = normalize_reason_label_ids(diagnosis.reason)
    lines = [
        "진단 요약",
        f"- 진단 ID: {diagnosis.diagnosis_id}",
        f"- 처리 경로: {route_label(diagnosis.route)}",
        f"- 처리 상태: {status_label(diagnosis.status)}",
        f"- 최종 판정: {final_label or '없음'}",
        f"- 신뢰도: {confidence_text(confidence)}",
        f"- 검토 필요: {'필요' if diagnosis.requires_human_review else '불필요'}",
        "",
        "입력 데이터",
        *input_summary,
        "",
        "모델별 판단",
        *signal_lines(signals),
        "",
        "판정 근거",
        f"- {reason}",
        "",
        "권고 조치",
        f"- {recommended_action(detail.trace)}",
        "",
        "참고 사례",
        *reference_case_lines(reference_cases),
    ]
    return "\n".join(lines)


def trace_signals(trace: TraceResponse, final_label: str | None) -> list[DiagnosisSignal]:
    return [
        DiagnosisSignal("시계열", model_signal(trace, "time_series_tool")),
        DiagnosisSignal("비전", model_signal(trace, "vision_tool")),
        DiagnosisSignal("VLM", model_signal(trace, "vlm_tool")),
        DiagnosisSignal("융합", fusion_signal(trace)),
        DiagnosisSignal("RAG", rag_signal(trace, final_label)),
    ]


def model_signal(trace: TraceResponse, event_name: str) -> str:
    summary = event_summary(trace, event_name)
    if summary is None:
        return "실행 안 됨"
    label = text_value(summary.get("label_name")) or text_value(summary.get("diagnosis")) or "없음"
    confidence = confidence_text(float_value(summary.get("confidence")))
    model_name = text_value(summary.get("model_name")) or "모델명 없음"
    return f"{label} / 신뢰도 {confidence} / {model_name}"


def fusion_signal(trace: TraceResponse) -> str:
    summary = event_summary(trace, "fusion_engine")
    if summary is None:
        return "없음"
    label = fusion_final_label(trace) or "없음"
    agreement = text_value(summary.get("agreement_level")) or "none"
    confidence = confidence_text(fusion_confidence(trace))
    return f"{label} / 합의 수준 {agreement} / 신뢰도 {confidence}"


def rag_signal(trace: TraceResponse, final_label: str | None) -> str:
    summary = event_summary(trace, "rag_tool")
    if summary is None:
        return "검색 안 됨"
    document_count = summary.get("document_count")
    top_title = text_value(summary.get("top_title")) or "상위 문서 없음"
    if has_label_mismatch(top_title, final_label):
        return f"{document_count}개 문서 / 최종 판정과 같은 상위 근거 없음"
    return f"{document_count}개 문서 / 상위 근거 {top_title}"


def signal_lines(signals: list[DiagnosisSignal]) -> list[str]:
    return [f"- {signal.label}: {signal.value}" for signal in signals]


def similar_case_titles(trace: TraceResponse, final_label: str | None) -> list[str]:
    summary = event_summary(trace, "similar_case_tool")
    if summary is None:
        return []
    cases = summary.get("cases")
    if not isinstance(cases, list):
        return []
    titles: list[str] = []
    for item in cases:
        if isinstance(item, dict):
            sample_id = text_value(item.get("sample_id"))
            label = text_value(item.get("label_name"))
            if has_label_mismatch(label, final_label):
                continue
            similarity = confidence_text(float_value(item.get("similarity")))
            if sample_id:
                titles.append(f"{sample_id} / {label or '라벨 없음'} / 유사도 {similarity}")
            if len(titles) >= 3:
                break
    return titles


def reference_case_lines(reference_cases: list[str]) -> list[str]:
    if not reference_cases:
        return ["- 최종 판정과 같은 참조 사례 없음"]
    return [f"- {case}" for case in reference_cases]


def input_summary_lines(trace: TraceResponse) -> list[str]:
    metadata = event_summary(trace, "metadata_context") or {}
    artifacts = event_summary(trace, "input_artifacts") or {}
    lines = [
        f"- 설비/센서: {equipment_text(metadata)}",
        f"- 운전 조건: {environment_text(metadata)}",
        f"- 입력 파일: {artifact_text(artifacts)}",
    ]
    signal = artifacts.get("timeseries_signal")
    if isinstance(signal, dict):
        lines.append(f"- 시계열 요약: {timeseries_signal_text(signal)}")
    return lines


def equipment_text(metadata: dict[str, object]) -> str:
    if not metadata:
        return "메타데이터 없음"
    equipment = text_value(metadata.get("equipment_name")) or "설비 없음"
    sensor = text_value(metadata.get("sensor_type")) or "센서 없음"
    voltage = text_value(metadata.get("rated_voltage")) or "전압 없음"
    current = text_value(metadata.get("rated_current")) or "전류 없음"
    insulator = text_value(metadata.get("insulator_type")) or "절연 없음"
    return f"{equipment} / {sensor} / {voltage} / {current} / 절연 {insulator}"


def environment_text(metadata: dict[str, object]) -> str:
    if not metadata:
        return "메타데이터 없음"
    temperature = display_value(metadata.get("temperature"), "도")
    humidity = display_value(metadata.get("humidity"), "%")
    clearance = text_value(metadata.get("clearance_distance")) or "이격 없음"
    condition = text_value(metadata.get("operating_condition")) or "운전 조건 없음"
    return f"온도 {temperature}, 습도 {humidity}, 이격 {clearance}, {condition}"


def artifact_text(artifacts: dict[str, object]) -> str:
    if not artifacts:
        return "첨부 파일 정보 없음"
    parts = []
    if text_value(artifacts.get("prpd_image_url")):
        parts.append("PRPD 이미지")
    if text_value(artifacts.get("timeseries_csv_url")):
        parts.append("시계열 CSV")
    return ", ".join(parts) if parts else "첨부 파일 정보 없음"


def timeseries_signal_text(signal: dict[str, object]) -> str:
    frame_count = display_value(signal.get("frame_count"))
    rms = display_value(signal.get("rms"))
    peak_abs = display_value(signal.get("peak_abs"))
    return f"프레임 {frame_count}, RMS {rms}, 최대절대값 {peak_abs}"


def recommended_action(trace: TraceResponse) -> str:
    vlm_summary = event_summary(trace, "vlm_tool")
    if vlm_summary is not None:
        action = text_value(vlm_summary.get("recommended_action"))
        if action:
            return action
    report_summary = event_summary(trace, "report_agent")
    if report_summary is not None:
        return "리포트 화면에서 상세 권고 조치를 확인하세요."
    return "운영자 검토가 필요합니다."


def event_summary(trace: TraceResponse, event_name: str) -> dict[str, object] | None:
    for event in trace.events:
        if isinstance(event, dict) and event.get("name") == event_name:
            summary = event.get("summary")
            return dict(summary) if isinstance(summary, dict) else None
    return None


def route_label(route: str) -> str:
    return {
        "hybrid": "종합 진단",
        "timeseries_only": "시계열 진단",
        "vlm_only": "비전 진단",
        "insufficient_input": "입력 반려",
    }.get(route, route)


def status_label(status: str) -> str:
    return {
        "completed": "완료",
        "needs_review": "검토 필요",
        "rejected": "반려",
    }.get(status, status)


def fusion_final_label(trace: TraceResponse) -> str | None:
    summary = event_summary(trace, "fusion_engine")
    if summary is None:
        return None
    return text_value(summary.get("final_label_name"))


def fusion_confidence(trace: TraceResponse) -> float | None:
    summary = event_summary(trace, "fusion_engine")
    if summary is None:
        return None
    return float_value(summary.get("confidence"))


def normalize_reason_label_ids(reason: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        source = match.group(1)
        label_id = int(match.group(2))
        try:
            label = policy_label_name(label_id)
        except ValueError:
            return match.group(0)
        return f"{source}={label}({label_id})"

    return REASON_LABEL_ID_PATTERN.sub(replacement, reason)


def has_label_mismatch(candidate: str | None, final_label: str | None) -> bool:
    candidate_label = inferred_label_name(candidate)
    if candidate_label is None or final_label is None:
        return False
    return compact_label(candidate_label) != compact_label(final_label)


def inferred_label_name(value: str | None) -> str | None:
    if value is None:
        return None
    compact_value = compact_label(value)
    for label_name, tokens in label_tokens().items():
        if any(token in compact_value for token in tokens):
            return label_name
    return None


def label_tokens() -> dict[str, tuple[str, ...]]:
    return {
        "정상": ("정상",),
        "노이즈": ("노이즈",),
        "표면 방전": ("표면방전", "표면"),
        "코로나 방전": ("코로나방전", "코로나"),
        "보이드 방전": ("보이드방전", "보이드"),
    }


def compact_label(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def confidence_text(value: float | None) -> str:
    if value is None:
        return "없음"
    if value <= 1:
        return f"{value:.0%}"
    return f"{value:.3g}"


def text_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def display_value(value: object, suffix: str = "") -> str:
    if isinstance(value, int | float):
        return f"{value:g}{suffix}"
    text = text_value(value)
    return f"{text}{suffix}" if text else "없음"


def float_value(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
