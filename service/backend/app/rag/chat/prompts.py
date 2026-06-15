from __future__ import annotations

from service.backend.app.rag.chat.constants import (
    MAX_CHAT_DOCUMENTS,
    MAX_CHAT_EXCERPT_CHARS,
    MAX_CHAT_HISTORY,
    NO_RAG_EVIDENCE_NOTICE,
)
from service.backend.app.schemas import RagChatMessage, RagDocument


CONTEXT_METADATA_FIELDS = (
    ("sample_id", "샘플ID"),
    ("label", "라벨"),
    ("label_name", "라벨"),
    ("equipment_name", "설비"),
    ("equipment_type", "설비"),
    ("sensor_type", "센서"),
    ("insulator_type", "절연"),
    ("equipment_rated_voltage", "전압"),
    ("recording_time", "기록시각"),
    ("power_supply_frequency", "전원주파수"),
    ("temperature", "온도"),
    ("humidity", "습도"),
    ("max_discharge_value", "피크"),
    ("defect_details", "결함상세"),
)


def build_rag_chat_messages(
    question: str,
    history: list[RagChatMessage],
    documents: list[RagDocument],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(has_rag_evidence=bool(documents))},
        *history_messages(history),
        {"role": "user", "content": build_user_prompt(question, documents)},
    ]


def build_diagnosis_history_chat_messages(
    question: str,
    history: list[RagChatMessage],
    diagnosis_context: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_diagnosis_history_system_prompt()},
        *history_messages(history),
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    f"사용자 질문:\n{question}",
                    f"진단 이력 근거:\n{diagnosis_context}",
                ]
            ),
        },
    ]


def build_system_prompt(has_rag_evidence: bool = True) -> str:
    if not has_rag_evidence:
        return build_general_domain_system_prompt()

    return "\n".join(
        [
            "당신은 부분방전 진단 RAG 챗봇입니다.",
            "반드시 제공된 RAG 근거와 이전 대화만 사용해 한국어로 답변합니다.",
            "근거가 부족하면 추정하지 말고 추가 확인이 필요하다고 말합니다.",
            "검색 모드가 exact_sample_id이면 해당 샘플ID의 직접 조회 결과로 설명합니다.",
            "검색 모드가 semantic_similarity이면 유사 근거일 뿐 정답 일치라고 말하지 않습니다.",
            "검색 모드가 metadata_filter이면 조건에 일치하는 데이터셋 사례로 설명합니다.",
            "답변은 반드시 아래 형식을 지킵니다.",
            "핵심 판단:",
            "- 한 문장으로 진단 결론을 씁니다.",
            "판단 근거:",
            "- 가장 중요한 근거 2~3개만 짧게 씁니다.",
            "참고 사례:",
            "- 데이터셋 사례명은 최대 3개만 한 줄씩 씁니다.",
            "섹션 제목 줄에는 본문을 붙이지 않습니다. 예: '판단 근거: - 값' 형식은 금지합니다.",
            "영어 표현을 섞지 말고 한국어로만 설명합니다.",
            "한 줄에 여러 수치와 메타데이터를 길게 나열하지 말고 중요한 항목만 선별합니다.",
            "장비, 전압, 전류, 절연, clearance를 모두 나열하지 말고 질문에 필요한 항목만 사용합니다.",
            '반드시 JSON 객체 하나만 반환합니다. 형식: {"answer":"답변"}',
        ]
    )


def build_diagnosis_history_system_prompt() -> str:
    return "\n".join(
        [
            "당신은 부분방전 진단 이력을 설명하는 RAG 챗봇입니다.",
            "반드시 제공된 진단 이력 근거만 사용해 한국어로 답변합니다.",
            "검토 필요 상태는 판정 실패가 아니라 모델 불일치나 추가 확인이 필요한 상태로 설명합니다.",
            "최종 판정, 처리 상태, 모델별 판단을 서로 섞지 말고 분리해서 설명합니다.",
            "시계열/비전/VLM 라벨이 다르면 어떤 모델이 무엇을 다르게 봤는지 라벨명과 번호를 함께 설명합니다.",
            "RAG 또는 유사 사례가 최종 판정과 다르면 정답이 아니라 검토 근거 또는 불일치 신호라고 설명합니다.",
            "사용자가 정답 여부를 묻지 않았으면 정답이라고 단정하지 않습니다.",
            "답변은 핵심 판단, 왜 그렇게 표시됐나, 확인 포인트 순서로 짧게 씁니다.",
            '반드시 JSON 객체 하나만 반환합니다. 형식: {"answer":"답변"}',
        ]
    )


def build_general_domain_system_prompt() -> str:
    return "\n".join(
        [
            "당신은 부분방전 진단 도메인 설명 챗봇입니다.",
            "제공된 RAG 검색 근거가 없으면 부분방전, PRPD, HFCT, 방전 유형 등 도메인 일반 개념은 일반 기술 지식으로 설명할 수 있습니다.",
            f"답변 첫 줄에는 반드시 '{NO_RAG_EVIDENCE_NOTICE}'이라고 씁니다.",
            "특정 설비, 업로드 데이터, 진단 ID, 위험도, 조치 판정은 검색 근거 없이 단정하지 않습니다.",
            "부분방전 진단 도메인 밖 질문은 앱 범위 밖이라고 짧게 안내합니다.",
            "한국어로만 답변하고, 질문에 맞는 핵심 설명만 2~4문장으로 씁니다.",
            '반드시 JSON 객체 하나만 반환합니다. 형식: {"answer":"답변"}',
        ]
    )


def history_messages(history: list[RagChatMessage]) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in history[-MAX_CHAT_HISTORY:]
    ]


def build_user_prompt(question: str, documents: list[RagDocument]) -> str:
    return "\n\n".join(
        [
            f"사용자 질문:\n{question}",
            rag_context(documents),
        ]
    )


def rag_context(documents: list[RagDocument]) -> str:
    if not documents:
        return "RAG 검색 근거: 없음"

    lines = ["RAG 검색 근거:"]
    for index, document in enumerate(documents[:MAX_CHAT_DOCUMENTS], start=1):
        lines.extend(document_lines(index, document))
    return "\n".join(lines)


def document_lines(index: int, document: RagDocument) -> list[str]:
    lines = [
        f"[근거 {index}] {document.title}",
        f"- 유형: {document.source_type or 'unknown'}",
        f"- 검색 모드: {document.retrieval_mode or document.metadata.get('retrieval_mode') or 'semantic_similarity'}",
        f"- 출처: {document.source}",
        f"- 관련도: {document.relevance:.3f}",
    ]
    metadata = document_metadata_line(document)
    if metadata:
        lines.append(f"- 메타데이터: {metadata}")
    lines.append(f"- 본문: {trim_excerpt(document.excerpt)}")
    return lines


def document_metadata_line(document: RagDocument) -> str:
    fields = []
    seen_labels: set[str] = set()
    for key, label in CONTEXT_METADATA_FIELDS:
        value = document.metadata.get(key)
        if value is None or label in seen_labels:
            continue
        fields.append(f"{label}={value}")
        seen_labels.add(label)
    return "; ".join(fields)


def trim_excerpt(text: str) -> str:
    clean_text = " ".join(text.split())
    if len(clean_text) <= MAX_CHAT_EXCERPT_CHARS:
        return clean_text
    return f"{clean_text[:MAX_CHAT_EXCERPT_CHARS]}..."
