from __future__ import annotations

import json

from pydantic import ValidationError

from service.backend.app.rag.chat.constants import (
    ANSWER_MODE_DIAGNOSIS_HISTORY,
    ANSWER_MODE_GENERAL_DOMAIN,
    ANSWER_MODE_RAG_EVIDENCE,
    LOCAL_DIAGNOSIS_HISTORY_MODEL,
    LOCAL_RAG_EVIDENCE_FALLBACK_MODEL,
    NO_RAG_EVIDENCE_NOTICE,
)
from service.backend.app.rag.chat.diagnosis_history import (
    DiagnosisHistoryLookup,
    diagnosis_history_answer,
    lookup_diagnosis_history_question,
    missing_diagnosis_answer,
)
from service.backend.app.rag.chat.guard import build_non_domain_response
from service.backend.app.rag.chat.intent import QueryIntent, QueryIntentKind, classify_question
from service.backend.app.rag.chat.models import ChatCompleter, RagChatDependencies, RagChatInput
from service.backend.app.rag.chat.parser import parse_rag_chat_payload
from service.backend.app.rag.chat.prompts import build_diagnosis_history_chat_messages, build_rag_chat_messages
from service.backend.app.rag.openrouter_client import (
    OpenRouterClient,
    OpenRouterConfigError,
    OpenRouterRequestError,
    OpenRouterSettings,
)
from service.backend.app.rag.retriever import PgvectorRagRetrievalAdapter
from service.backend.app.rag.vector_store import PgvectorStoreError
from service.backend.app.schemas import RagChatResponse, RagDocument


FALLBACK_METADATA_FIELDS = (
    ("sample_id", "샘플ID"),
    ("label", "라벨"),
    ("label_name", "라벨"),
    ("equipment_name", "설비"),
    ("equipment_type", "설비"),
    ("sensor_type", "센서"),
    ("insulator_type", "절연"),
    ("recording_time", "기록시각"),
    ("power_supply_frequency", "전원주파수"),
    ("temperature", "온도"),
    ("humidity", "습도"),
    ("max_discharge_value", "피크"),
    ("defect_details", "결함상세"),
)


def answer_rag_chat(
    chat_input: RagChatInput,
    dependencies: RagChatDependencies | None = None,
) -> RagChatResponse:
    active_dependencies = dependencies or RagChatDependencies()
    history_lookup = lookup_diagnosis_history_question(chat_input, active_dependencies.diagnosis_store)
    if history_lookup is not None:
        return answer_with_diagnosis_history(chat_input, active_dependencies, history_lookup)

    intent = classify_question(chat_input.question)
    if not intent.should_retrieve:
        return build_non_domain_response()

    return answer_with_rag(chat_input, active_dependencies, intent)


def answer_with_diagnosis_history(
    chat_input: RagChatInput,
    dependencies: RagChatDependencies,
    lookup: DiagnosisHistoryLookup,
) -> RagChatResponse:
    if lookup.detail is None:
        return RagChatResponse(
            answer=missing_diagnosis_answer(lookup.diagnosis_id),
            documents=[],
            model=LOCAL_DIAGNOSIS_HISTORY_MODEL,
            ready=True,
            answer_mode=ANSWER_MODE_DIAGNOSIS_HISTORY,
        )

    local_answer = diagnosis_history_answer(lookup.detail)
    try:
        settings = dependencies.settings or OpenRouterSettings.from_env()
    except OpenRouterConfigError as exc:
        return local_diagnosis_history_fallback_response(exc, local_answer)

    messages = build_diagnosis_history_chat_messages(chat_input.question, chat_input.history, local_answer)
    try:
        payload = parse_rag_chat_payload(chat_client(dependencies, settings).complete(messages))
    except (OpenRouterRequestError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        return local_diagnosis_history_fallback_response(exc, local_answer)

    return RagChatResponse(
        answer=payload.answer.strip(),
        documents=[],
        model=settings.model,
        ready=True,
        answer_mode=ANSWER_MODE_DIAGNOSIS_HISTORY,
    )


def answer_with_rag(
    chat_input: RagChatInput,
    dependencies: RagChatDependencies,
    intent: QueryIntent,
) -> RagChatResponse:
    try:
        documents = chat_documents(intent, retrieve_documents(chat_input, dependencies, intent))
    except (PgvectorStoreError, RuntimeError) as exc:
        return chat_error_response(exc, [], configured_model_name(dependencies))
    try:
        settings = dependencies.settings or OpenRouterSettings.from_env()
    except OpenRouterConfigError as exc:
        return local_rag_fallback_response(exc, documents)
    messages = build_rag_chat_messages(chat_input.question, chat_input.history, documents)
    try:
        payload = parse_rag_chat_payload(chat_client(dependencies, settings).complete(messages))
    except (OpenRouterRequestError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        return local_rag_fallback_response(exc, documents)

    return RagChatResponse(
        answer=chat_answer(payload.answer, documents),
        documents=documents,
        model=settings.model,
        ready=True,
        answer_mode=chat_answer_mode(documents),
    )


def retrieve_documents(
    chat_input: RagChatInput,
    dependencies: RagChatDependencies,
    intent: QueryIntent,
) -> list[RagDocument]:
    retriever = dependencies.retriever or PgvectorRagRetrievalAdapter()
    return retriever.search_text(
        chat_input.question,
        top_k=chat_input.top_k,
        source_types=intent.source_types,
    )


def chat_documents(intent: QueryIntent, documents: list[RagDocument]) -> list[RagDocument]:
    if intent.kind != QueryIntentKind.concept_explanation or intent.source_types is None:
        return documents
    return [document for document in documents if document.source_type in intent.source_types]


def chat_client(
    dependencies: RagChatDependencies,
    settings: OpenRouterSettings,
) -> ChatCompleter:
    return dependencies.client or OpenRouterClient(settings)


def chat_answer(answer: str, documents: list[RagDocument]) -> str:
    clean_answer = answer.strip()
    if documents or clean_answer.startswith(NO_RAG_EVIDENCE_NOTICE):
        return clean_answer
    if clean_answer == "":
        return NO_RAG_EVIDENCE_NOTICE
    return f"{NO_RAG_EVIDENCE_NOTICE}\n\n{clean_answer}"


def chat_answer_mode(documents: list[RagDocument]) -> str:
    return ANSWER_MODE_RAG_EVIDENCE if documents else ANSWER_MODE_GENERAL_DOMAIN


def configured_model_name(dependencies: RagChatDependencies) -> str | None:
    return dependencies.settings.model if dependencies.settings is not None else None


def local_rag_fallback_response(exc: Exception, documents: list[RagDocument]) -> RagChatResponse:
    return RagChatResponse(
        answer=local_rag_fallback_answer(exc, documents),
        documents=documents,
        model=LOCAL_RAG_EVIDENCE_FALLBACK_MODEL,
        ready=True,
        answer_mode=chat_answer_mode(documents),
    )


def local_diagnosis_history_fallback_response(exc: Exception, local_answer: str) -> RagChatResponse:
    return RagChatResponse(
        answer=local_diagnosis_history_fallback_answer(exc, local_answer),
        documents=[],
        model=LOCAL_DIAGNOSIS_HISTORY_MODEL,
        ready=True,
        answer_mode=ANSWER_MODE_DIAGNOSIS_HISTORY,
    )


def local_diagnosis_history_fallback_answer(exc: Exception, local_answer: str) -> str:
    return "\n\n".join(
        [
            "핵심 판단:",
            f"- OpenRouter 응답 생성이 실패해 저장된 진단 이력 요약으로 답변합니다. ({short_error_text(exc)})",
            local_answer,
        ]
    )


def local_rag_fallback_answer(exc: Exception, documents: list[RagDocument]) -> str:
    lines = [
        "핵심 판단:",
        f"- OpenRouter 응답 생성이 실패해 로컬 RAG 근거 요약으로 답변합니다. ({short_error_text(exc)})",
        "판단 근거:",
    ]
    if not documents:
        lines.extend(
            [
                f"- {NO_RAG_EVIDENCE_NOTICE}: 현재 질의와 일치하는 RAG 근거를 찾지 못했습니다.",
                "- OpenRouter 설정을 복구하면 일반 도메인 설명 생성도 다시 사용할 수 있습니다.",
                "참고 사례:",
                "- 없음",
            ]
        )
        return "\n".join(lines)

    lines.extend(fallback_evidence_lines(documents))
    lines.append("참고 사례:")
    lines.extend(fallback_reference_lines(documents))
    return "\n".join(lines)


def fallback_evidence_lines(documents: list[RagDocument]) -> list[str]:
    return [fallback_evidence_line(document) for document in documents[:3]]


def fallback_evidence_line(document: RagDocument) -> str:
    metadata = fallback_metadata_text(document)
    suffix = f" / {metadata}" if metadata else ""
    return f"- {document.title}: 관련도 {document.relevance:.0%}{suffix}"


def fallback_reference_lines(documents: list[RagDocument]) -> list[str]:
    return [f"- {document.title}" for document in documents[:3]]


def fallback_metadata_text(document: RagDocument) -> str:
    fields = []
    for key, label in FALLBACK_METADATA_FIELDS:
        value = document.metadata.get(key)
        if value is not None and value != "":
            fields.append(f"{label} {value}")
    return " · ".join(fields)


def short_error_text(exc: Exception) -> str:
    text = str(exc).strip()
    if len(text) <= 160:
        return text
    return f"{text[:157]}..."


def chat_error_response(
    exc: Exception,
    documents: list[RagDocument],
    model: str | None,
) -> RagChatResponse:
    return RagChatResponse(
        answer="",
        documents=documents,
        model=model,
        ready=False,
        error=str(exc),
        answer_mode=chat_answer_mode(documents),
    )
