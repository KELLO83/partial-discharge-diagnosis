from __future__ import annotations

from pathlib import Path

from service.backend.app.application.contracts import VlmToolInput
from service.backend.app.rag.llm_prompt_builder import build_llm_rag_messages
from service.backend.app.rag.llm_reporter import (
    OpenRouterLlmRagInferenceAdapter,
    build_llm_rag_reporter,
    parse_llm_rag_payload,
)
from service.backend.app.schemas import MetadataInput, RagDocument, RagResult, SimilarCase, VlmResult


def test_openrouter_llm_rag_adapter_returns_standard_vlm_result(tmp_path: Path) -> None:
    image_path = tmp_path / "prpd.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = OpenRouterLlmRagInferenceAdapter(
        client=_FakeClient(
            '{"label_id":3,"diagnosis":"코로나 방전","risk_level":"주의",'
            '"confidence":0.86,"reason":"RAG 근거와 유사 사례가 코로나 방전을 지지합니다.",'
            '"recommended_action":"고전계 접속부를 점검하세요."}'
        ),
        model_version="test/openrouter-model",
    )

    result = adapter.run(_vlm_input(image_path))

    assert result.model_name == "openrouter_llm_rag"
    assert result.model_version == "test/openrouter-model"
    assert result.label_id == 3
    assert result.diagnosis == "코로나 방전"
    assert result.standard_evidence is not None


def test_build_llm_rag_messages_includes_retrieved_documents(tmp_path: Path) -> None:
    messages = build_llm_rag_messages(_vlm_input(tmp_path / "prpd.png"))

    prompt = messages[1]["content"]

    assert "RAG 문서 근거" in prompt
    assert "코로나 방전 판단 기준" in prompt
    assert "유사 사례 근거" in prompt
    assert "case-001" in prompt


def test_parse_llm_rag_payload_accepts_json_fence() -> None:
    payload = parse_llm_rag_payload(
        '```json\n{"label_id":1,"diagnosis":"노이즈","risk_level":"낮음",'
        '"confidence":0.71,"reason":"위상 전 구간 대역형 분포입니다.",'
        '"recommended_action":"센서 접촉과 외란을 확인하세요."}\n```'
    )

    assert payload.label_id == 1
    assert payload.confidence == 0.71


def test_build_llm_rag_reporter_falls_back_without_openrouter_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_RAG_PROVIDER", "auto")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fallback = _FallbackAdapter()

    adapter, status = build_llm_rag_reporter(fallback)

    assert adapter is fallback
    assert status.provider == "auto"
    assert status.active_adapter == "test_fallback_vlm"
    assert status.ready is False


class _FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, messages: list[dict[str, str]]) -> str:
        assert messages
        return self._response


class _FallbackAdapter:
    model_name = "test_fallback_vlm"
    model_version = "test"

    def run(self, tool_input: VlmToolInput) -> VlmResult:
        return VlmResult(
            model_name=self.model_name,
            model_version=self.model_version,
            label_id=0,
            diagnosis="정상",
            risk_level="낮음",
            confidence=0.5,
            reason="fallback",
            recommended_action="monitor",
        )


def _vlm_input(image_path: Path) -> VlmToolInput:
    return VlmToolInput(
        image_path=image_path,
        image_sha256="abc",
        safe_metadata=_metadata(),
        timeseries_result=None,
        vision_result=None,
        rag_result=_rag_result(),
    )


def _metadata() -> MetadataInput:
    return MetadataInput(
        equipment_name="ACSR-OC",
        equipment_rated_voltage="22900V",
        equipment_rated_current="268A",
        sensor_type="HFCT",
        temperature=19,
        humidity=66,
    )


def _rag_result() -> RagResult:
    return RagResult(
        retriever_name="pgvector_rulebook_case_rag",
        retriever_version="test",
        query="HFCT 코로나 방전",
        documents=[
            RagDocument(
                document_id="rulebook-corona",
                title="코로나 방전 판단 기준",
                source="rulebook/corona_discharge.md",
                excerpt="위상 국부 로브와 고전계 접속부 근거를 함께 확인합니다.",
                relevance=0.91,
                source_type="rulebook",
            )
        ],
        similar_cases=[
            SimilarCase(
                sample_id="case-001",
                label_id=3,
                label_name="코로나 방전",
                equipment_name="ACSR-OC",
                insulator_type="폴리머",
                sensor_type="HFCT",
                clearance_distance="120mm",
                similarity=0.88,
                reason="phase-localized lobe가 유사합니다.",
                image_url="/dataset/cases/case-001/image",
                metadata={},
            )
        ],
    )
