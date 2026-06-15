from __future__ import annotations

from service.backend.app.rag.chat import (
    LOCAL_DIAGNOSIS_HISTORY_MODEL,
    LOCAL_CHAT_MODEL,
    LOCAL_RAG_EVIDENCE_FALLBACK_MODEL,
    RagChatDependencies,
    RagChatInput,
    answer_rag_chat,
    build_rag_chat_messages,
    classify_question,
    parse_rag_chat_payload,
)
from service.backend.app.rag.openrouter_client import OpenRouterRequestError, OpenRouterSettings
from service.backend.app.rag.vector_store import PgvectorStoreError
from service.backend.app.schemas import DiagnosisDetailResponse, DiagnosisListItem, RagChatMessage, RagDocument, TraceResponse


def test_answer_rag_chat_falls_back_when_openrouter_key_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    response = answer_rag_chat(
        chat_input("코로나 방전 근거는?"),
        RagChatDependencies(retriever=_FakeRetriever()),
    )

    assert response.ready is True
    assert response.error is None
    assert response.model == LOCAL_RAG_EVIDENCE_FALLBACK_MODEL
    assert response.documents[0].title == "코로나 방전 판단 기준"
    assert "OPENROUTER_API_KEY is not set" in response.answer


def test_answer_rag_chat_falls_back_when_openrouter_request_fails() -> None:
    response = answer_rag_chat(
        chat_input("코로나 방전 근거는?"),
        RagChatDependencies(
            retriever=_FakeRetriever(),
            client=_RequestFailingClient(),
            settings=OpenRouterSettings(api_key="bad-key", model="test-model"),
        ),
    )

    assert response.ready is True
    assert response.error is None
    assert response.model == LOCAL_RAG_EVIDENCE_FALLBACK_MODEL
    assert response.answer_mode == "rag_evidence"
    assert "OpenRouter request failed with HTTP 401" in response.answer
    assert "코로나 방전 판단 기준" in response.answer


def test_answer_rag_chat_uses_retrieved_documents() -> None:
    response = answer_rag_chat(
        chat_input(
            "코로나 방전 근거는?",
            [RagChatMessage(role="user", content="HFCT 기준으로 알려줘")],
        ),
        RagChatDependencies(
            retriever=_FakeRetriever(),
            client=_FakeClient('{"answer":"코로나 방전은 위상 국부화와 고전계 접속부 근거를 함께 봅니다."}'),
            settings=OpenRouterSettings(api_key="test-key", model="test-model"),
        ),
    )

    assert response.ready is True
    assert response.model == "test-model"
    assert response.answer_mode == "rag_evidence"
    assert response.answer.startswith("코로나 방전")
    assert response.documents[0].title == "코로나 방전 판단 기준"


def test_answer_rag_chat_allows_domain_answer_without_documents() -> None:
    response = answer_rag_chat(
        chat_input("부분방전이란 무엇이지?"),
        RagChatDependencies(
            retriever=_EmptyRetriever(),
            client=_NoEvidenceClient(
                '{"answer":"부분방전은 절연계 내부나 표면에서 국부적으로 발생하는 방전입니다."}'
            ),
            settings=OpenRouterSettings(api_key="test-key", model="test-model"),
        ),
    )

    assert response.ready is True
    assert response.answer_mode == "general_domain"
    assert response.documents == []
    assert response.answer.startswith("검색 근거 없음")


def test_answer_rag_chat_ignores_dataset_cases_for_concept_question() -> None:
    response = answer_rag_chat(
        chat_input("보이드방전이란?"),
        RagChatDependencies(
            retriever=_DatasetOnlyConceptRetriever(),
            client=_NoEvidenceClient('{"answer":"보이드 방전은 절연체 내부 공극에서 발생하는 부분방전입니다."}'),
            settings=OpenRouterSettings(api_key="test-key", model="test-model"),
        ),
    )

    assert response.ready is True
    assert response.answer_mode == "general_domain"
    assert response.documents == []
    assert response.answer.startswith("검색 근거 없음")


def test_answer_rag_chat_reports_retrieval_failure() -> None:
    response = answer_rag_chat(
        chat_input("코로나 방전 근거는?"),
        RagChatDependencies(
            retriever=_FailingRetriever(),
            client=_ExplodingClient(),
            settings=OpenRouterSettings(api_key="test-key", model="test-model"),
        ),
    )

    assert response.ready is False
    assert response.model == "test-model"
    assert response.documents == []
    assert response.error == "RAG database unavailable"


def test_answer_rag_chat_skips_retrieval_for_general_question() -> None:
    response = answer_rag_chat(
        chat_input("DEIT 모델은 무엇인가요?"),
        exploding_dependencies(),
    )

    assert response.ready is True
    assert response.model == LOCAL_CHAT_MODEL
    assert response.documents == []
    assert "부분방전 진단 RAG 전용" in response.answer


def test_answer_rag_chat_skips_retrieval_for_greeting() -> None:
    response = answer_rag_chat(
        chat_input("안녕"),
        exploding_dependencies(),
    )

    assert response.ready is True
    assert response.model == LOCAL_CHAT_MODEL
    assert response.documents == []


def test_answer_rag_chat_explains_diagnosis_history_without_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    response = answer_rag_chat(
        chat_input("이력 diag_111111111111 에대해서 설명해줘 및 보고해줘"),
        RagChatDependencies(
            diagnosis_store=_DiagnosisStore(_diagnosis_detail()),
            retriever=_ExplodingRetriever(),
            client=_ExplodingClient(),
        ),
    )

    assert response.ready is True
    assert response.model == LOCAL_DIAGNOSIS_HISTORY_MODEL
    assert response.documents == []
    assert "diag_111111111111" in response.answer
    assert "최종 판정: 코로나 방전" in response.answer
    assert "시계열: 코로나 방전" in response.answer
    assert "권고 조치" in response.answer
    assert "진단 요약" in response.answer
    assert "입력 데이터" in response.answer
    assert "ACSR-OC / HFCT / 22900V" in response.answer
    assert "PRPD 이미지, 시계열 CSV" in response.answer


def test_answer_rag_chat_explains_demo_diagnosis_history_without_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    response = answer_rag_chat(
        chat_input("demo_disagree_0001 보고진행"),
        RagChatDependencies(
            diagnosis_store=_DiagnosisStore(_diagnosis_detail("demo_disagree_0001")),
            retriever=_ExplodingRetriever(),
            client=_ExplodingClient(),
        ),
    )

    assert response.ready is True
    assert response.model == LOCAL_DIAGNOSIS_HISTORY_MODEL
    assert response.documents == []
    assert "demo_disagree_0001" in response.answer
    assert "진단 요약" in response.answer


def test_answer_rag_chat_hides_mismatched_history_references(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    response = answer_rag_chat(
        chat_input("이력 diag_111111111111 보고해줘"),
        RagChatDependencies(
            diagnosis_store=_DiagnosisStore(_diagnosis_detail_with_mismatched_references()),
            retriever=_ExplodingRetriever(),
            client=_ExplodingClient(),
        ),
    )

    assert response.ready is True
    assert "최종 판정: 코로나 방전" in response.answer
    assert "노이즈_고체_ACSR-OC_001" not in response.answer
    assert "최종 판정과 같은 참조 사례 없음" in response.answer
    assert "최종 판정과 같은 상위 근거 없음" in response.answer


def test_answer_rag_chat_reports_missing_diagnosis_history(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    response = answer_rag_chat(
        chat_input("diag_222222222222 설명해줘"),
        RagChatDependencies(
            diagnosis_store=_DiagnosisStore(None),
            retriever=_ExplodingRetriever(),
            client=_ExplodingClient(),
        ),
    )

    assert response.ready is True
    assert response.model == LOCAL_DIAGNOSIS_HISTORY_MODEL
    assert response.documents == []
    assert "찾지 못했습니다" in response.answer


def test_build_rag_chat_messages_includes_history_and_context() -> None:
    messages = build_rag_chat_messages(
        "근거 요약해줘",
        [RagChatMessage(role="assistant", content="이전 답변")],
        [_document()],
    )

    prompt = messages[-1]["content"]

    assert messages[1]["content"] == "이전 답변"
    assert "RAG 검색 근거" in prompt
    assert "코로나 방전 판단 기준" in prompt


def test_build_rag_chat_messages_includes_dataset_peak_metadata() -> None:
    messages = build_rag_chat_messages(
        "피크 값이 82인 데이터",
        [],
        [_dataset_document()],
    )

    prompt = messages[-1]["content"]

    assert "메타데이터" in prompt
    assert "피크=82" in prompt
    assert "센서=HFCT" in prompt


def test_build_rag_chat_messages_marks_missing_search_evidence() -> None:
    messages = build_rag_chat_messages("부분방전이란?", [], [])

    assert "일반 기술 지식으로 설명" in messages[0]["content"]
    assert "검색 근거 없음" in messages[0]["content"]
    assert "RAG 검색 근거: 없음" in messages[-1]["content"]


def test_classify_question_routes_concepts_to_knowledge_sources() -> None:
    intent = classify_question("보이드방전이란?")

    assert intent.kind == "concept_explanation"
    assert intent.source_types == ("rulebook", "sop")
    assert intent.answer_mode == "general_domain"


def test_parse_rag_chat_payload_accepts_json_fence() -> None:
    payload = parse_rag_chat_payload('```json\n{"answer":"근거 기반 답변"}\n```')

    assert payload.answer == "근거 기반 답변"


class _FakeRetriever:
    def search_text(self, query: str, top_k: int, source_types: tuple[str, ...] | None = None):
        assert query == "코로나 방전 근거는?"
        assert top_k == 3
        assert source_types is None
        return [_document()]


class _FailingRetriever:
    def search_text(self, query: str, top_k: int, source_types: tuple[str, ...] | None = None):
        raise PgvectorStoreError("RAG database unavailable")


class _EmptyRetriever:
    def search_text(self, query: str, top_k: int, source_types: tuple[str, ...] | None = None):
        assert query == "부분방전이란 무엇이지?"
        assert top_k == 3
        assert source_types == ("rulebook", "sop")
        return []


class _DatasetOnlyConceptRetriever:
    def search_text(self, query: str, top_k: int, source_types: tuple[str, ...] | None = None):
        assert query == "보이드방전이란?"
        assert top_k == 3
        assert source_types == ("rulebook", "sop")
        return [_dataset_document()]


class _FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, messages: list[dict[str, str]]) -> str:
        assert any("코로나 방전 판단 기준" in message["content"] for message in messages)
        return self._response


class _RequestFailingClient:
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise OpenRouterRequestError('OpenRouter request failed with HTTP 401: {"error":"User not found"}')


class _NoEvidenceClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, messages: list[dict[str, str]]) -> str:
        assert "검색 근거 없음" in messages[0]["content"]
        assert "RAG 검색 근거: 없음" in messages[-1]["content"]
        return self._response


class _ExplodingRetriever:
    def search_text(self, query: str, top_k: int, source_types: tuple[str, ...] | None = None):
        raise AssertionError("RAG retrieval should not run for non-domain chat input")


class _ExplodingClient:
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("OpenRouter should not run for non-domain chat input")


class _DiagnosisStore:
    def __init__(
        self,
        detail: DiagnosisDetailResponse | None,
        expected_ids: set[str] | None = None,
    ) -> None:
        self._detail = detail
        self._expected_ids = expected_ids

    def detail(self, diagnosis_id: str) -> DiagnosisDetailResponse | None:
        if self._expected_ids is not None:
            assert diagnosis_id in self._expected_ids
        return self._detail


def chat_input(
    question: str,
    history: list[RagChatMessage] | None = None,
) -> RagChatInput:
    return RagChatInput(question=question, history=history or [], top_k=3)


def exploding_dependencies() -> RagChatDependencies:
    return RagChatDependencies(
        retriever=_ExplodingRetriever(),
        client=_ExplodingClient(),
        settings=OpenRouterSettings(api_key="test-key", model="test-model"),
    )


def _document() -> RagDocument:
    return RagDocument(
        document_id="rulebook-corona",
        title="코로나 방전 판단 기준",
        source="rulebook/corona_discharge.md",
        excerpt="위상 국부 로브와 고전계 접속부 근거를 함께 확인합니다.",
        relevance=0.91,
        source_type="rulebook",
    )


def _dataset_document() -> RagDocument:
    return RagDocument(
        document_id="dataset_case:noise-82#0",
        title="노이즈_고체_ACSR-OC_230910_210042_HFCT_1000",
        source="dataset_case:noise-82#0",
        excerpt="label=노이즈; sensor=HFCT; max_discharge=82",
        relevance=0.89,
        source_type="dataset_case",
        metadata={
            "label": "노이즈",
            "sensor_type": "HFCT",
            "max_discharge_value": 82,
        },
    )


def _diagnosis_detail(diagnosis_id: str = "diag_111111111111") -> DiagnosisDetailResponse:
    return DiagnosisDetailResponse(
        diagnosis=DiagnosisListItem(
            diagnosis_id=diagnosis_id,
            trace_id=f"trace_{diagnosis_id}",
            route="hybrid",
            status="completed",
            diagnosis="코로나 방전",
            risk_level="주의",
            confidence=0.9,
            reason="모델 및 지식 검색 근거가 일관되어 최종 진단을 확정했습니다.",
            requires_human_review=False,
            created_at="2026-06-15T00:00:00+00:00",
        ),
        trace=TraceResponse(
            diagnosis_id="diag_111111111111",
            trace_id="trace_111111111111",
            route="hybrid",
            status="completed",
            steps=["input_router", "time_series_tool", "vision_tool", "vlm_tool"],
            summary={"reason": "모델 및 지식 검색 근거가 일관되어 최종 진단을 확정했습니다."},
            events=[
                {
                    "name": "input_artifacts",
                    "kind": "context",
                    "summary": {
                        "prpd_image_url": f"/diagnoses/{diagnosis_id}/artifacts/prpd-image",
                        "timeseries_csv_url": f"/diagnoses/{diagnosis_id}/artifacts/timeseries-csv",
                        "timeseries_signal": {
                            "frame_count": 1200,
                            "rms": 0.14,
                            "peak_abs": 0.82,
                        },
                    },
                },
                {
                    "name": "metadata_context",
                    "kind": "context",
                    "summary": {
                        "equipment_name": "ACSR-OC",
                        "rated_voltage": "22900V",
                        "rated_current": "268A",
                        "sensor_type": "HFCT",
                        "temperature": 19,
                        "humidity": 66,
                        "insulator_type": "고체",
                        "clearance_distance": "1500mm",
                    },
                },
                {
                    "name": "time_series_tool",
                    "kind": "tool",
                    "summary": {
                        "model_name": "mock_patchtst",
                        "label_name": "코로나 방전",
                        "confidence": 0.87,
                    },
                },
                {
                    "name": "vision_tool",
                    "kind": "tool",
                    "summary": {
                        "model_name": "mock_prpd_small_cnn",
                        "label_name": "코로나 방전",
                        "confidence": 0.82,
                    },
                },
                {
                    "name": "similar_case_tool",
                    "kind": "tool",
                    "summary": {
                        "cases": [
                            {
                                "sample_id": "코로나방전_고체_ACSR-OC_001",
                                "label_name": "코로나 방전",
                                "similarity": 0.91,
                            }
                        ]
                    },
                },
                {
                    "name": "rag_tool",
                    "kind": "tool",
                    "summary": {
                        "document_count": 2,
                        "top_title": "코로나 방전 판단 기준",
                    },
                },
                {
                    "name": "vlm_tool",
                    "kind": "tool",
                    "summary": {
                        "model_name": "mock_qwen3_vl_2b",
                        "label_name": "코로나 방전",
                        "confidence": 0.9,
                        "recommended_action": "고전압 접속부를 점검하세요.",
                    },
                },
                {
                    "name": "fusion_engine",
                    "kind": "fusion",
                    "summary": {
                        "final_label_name": "코로나 방전",
                        "agreement_level": "agreement",
                        "confidence": 0.9,
                    },
                },
            ],
        ),
    )


def _diagnosis_detail_with_mismatched_references() -> DiagnosisDetailResponse:
    detail = _diagnosis_detail()
    for event in detail.trace.events:
        if event.get("name") == "similar_case_tool":
            event["summary"] = {
                "cases": [
                    {
                        "sample_id": "노이즈_고체_ACSR-OC_001",
                        "label_name": "노이즈",
                        "similarity": 0.99,
                    }
                ]
            }
        if event.get("name") == "rag_tool":
            event["summary"] = {
                "document_count": 1,
                "top_title": "데이터셋 사례 노이즈_고체_ACSR-OC_001",
            }
    return detail
