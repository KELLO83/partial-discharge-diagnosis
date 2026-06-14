from __future__ import annotations

import json

from service.backend.app.application.contracts import VlmToolInput
from service.backend.app.schemas import MetadataInput, RagDocument, SimilarCase, TimeSeriesResult, VisionResult


MAX_DOCUMENTS = 6
MAX_CASES = 5
MAX_EXCERPT_CHARS = 900
MISSING_VALUE = "미제공"


def build_llm_rag_messages(tool_input: VlmToolInput) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_prompt(tool_input)},
    ]


def _system_prompt() -> str:
    return "\n".join(
        [
            "당신은 부분방전(PRPD/PRPS/HFCT) 진단 리포트 생성기입니다.",
            "제공된 모델 출력, RAG 문서, 유사 사례 근거만 사용합니다.",
            "근거가 부족하면 확정 표현 대신 검토 필요성을 reason에 명시합니다.",
            "반드시 JSON 객체 하나만 반환합니다.",
        ]
    )


def _user_prompt(tool_input: VlmToolInput) -> str:
    documents, similar_cases = _rag_evidence(tool_input)
    return "\n\n".join(
        [
            _schema_instruction(),
            _label_instruction(),
            _metadata_context(tool_input.safe_metadata),
            _timeseries_context(tool_input.timeseries_result),
            _vision_context(tool_input.vision_result),
            _rag_context(documents),
            _similar_case_context(similar_cases),
        ]
    )


def _rag_evidence(tool_input: VlmToolInput) -> tuple[list[RagDocument], list[SimilarCase]]:
    if tool_input.rag_result is None:
        return [], []
    return tool_input.rag_result.documents, tool_input.rag_result.similar_cases


def _schema_instruction() -> str:
    return (
        "출력 JSON 스키마:\n"
        '{"label_id":0,"diagnosis":"정상","risk_level":"낮음","confidence":0.0,'
        '"reason":"근거 설명","recommended_action":"운영 조치"}'
    )


def _label_instruction() -> str:
    return "\n".join(
        [
            "라벨 체계:",
            "0=정상, risk=낮음",
            "1=노이즈, risk=낮음",
            "2=표면 방전, risk=주의",
            "3=코로나 방전, risk=주의",
            "4=보이드 방전, risk=위험",
        ]
    )


def _metadata_context(metadata: MetadataInput) -> str:
    return "\n".join(
        [
            "설비 메타데이터:",
            f"- 설비명: {metadata.equipment_name}",
            f"- 설비 유형: {_text_or_missing(metadata.equipment_type)}",
            f"- 정격 전압: {metadata.equipment_rated_voltage}",
            f"- 정격 전류: {metadata.equipment_rated_current}",
            f"- 센서: {metadata.sensor_type}",
            f"- 측정 위치: {_text_or_missing(metadata.measurement_location)}",
            f"- 운전 상태: {_text_or_missing(metadata.operating_condition)}",
            f"- 온도/습도: {metadata.temperature}/{metadata.humidity}",
            f"- 절연 유형: {_text_or_missing(metadata.insulator_type)}",
            f"- 이격 거리: {_text_or_missing(metadata.clearance_distance)}",
        ]
    )


def _timeseries_context(result: TimeSeriesResult | None) -> str:
    if result is None:
        return "시계열 모델 결과: 미제공"
    return "\n".join(
        [
            "시계열 모델 결과:",
            f"- 모델: {result.model_name} {result.model_version}",
            f"- 라벨: {result.label_id} {result.label_name}",
            f"- 신뢰도: {result.confidence:.3f}",
            f"- 주요 특징: {_json_text(result.features)}",
        ]
    )


def _vision_context(result: VisionResult | None) -> str:
    if result is None:
        return "비전 모델 결과: 미제공"
    return "\n".join(
        [
            "비전 모델 결과:",
            f"- 모델: {result.model_name} {result.model_version}",
            f"- 라벨: {result.label_id} {result.label_name}",
            f"- 신뢰도: {result.confidence:.3f}",
            f"- 근거: {_json_text(result.evidence)}",
        ]
    )


def _rag_context(documents: list[RagDocument]) -> str:
    if not documents:
        return "RAG 문서 근거: 검색 결과 없음"
    lines = ["RAG 문서 근거:"]
    for index, document in enumerate(documents[:MAX_DOCUMENTS], start=1):
        lines.extend(_document_lines(index, document))
    return "\n".join(lines)


def _document_lines(index: int, document: RagDocument) -> list[str]:
    return [
        f"[문서 {index}] {document.title}",
        f"- 출처: {document.source}",
        f"- 소스 유형: {document.source_type or 'unknown'}",
        f"- 관련도: {document.relevance:.3f}",
        f"- 본문: {_trim(document.excerpt)}",
    ]


def _similar_case_context(cases: list[SimilarCase]) -> str:
    if not cases:
        return "유사 사례 근거: 검색 결과 없음"
    lines = ["유사 사례 근거:"]
    for index, case in enumerate(cases[:MAX_CASES], start=1):
        lines.extend(_case_lines(index, case))
    return "\n".join(lines)


def _case_lines(index: int, case: SimilarCase) -> list[str]:
    return [
        f"[사례 {index}] {case.sample_id}",
        f"- 라벨: {case.label_id} {case.label_name}",
        f"- 설비/센서/절연: {case.equipment_name}/{case.sensor_type}/{case.insulator_type}",
        f"- 유사도: {case.similarity:.3f}",
        f"- 근거: {case.reason}",
    ]


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _text_or_missing(value: str | None) -> str:
    if value is None or value.strip() == "":
        return MISSING_VALUE
    return value


def _trim(text: str) -> str:
    clean = " ".join(text.split())
    if len(clean) <= MAX_EXCERPT_CHARS:
        return clean
    return f"{clean[:MAX_EXCERPT_CHARS]}..."
