from __future__ import annotations

from service.backend.app.application.contracts import RagToolInput
from service.backend.app.domain.similar_cases import dataset_case_repository
from service.backend.app.rag.documents import RagSearchHit
from service.backend.app.rag.embeddings import DeterministicTextEmbeddingModel
from service.backend.app.rag.query import infer_label_ids_from_text
from service.backend.app.rag.query_constraints import (
    applied_query_filters,
    extract_query_constraints,
    extract_sample_id,
    hit_contains_metadata_term,
)
from service.backend.app.rag.reranker import rerank_hits
from service.backend.app.rag.retriever import PgvectorRagRetrievalAdapter
from service.backend.app.rag.settings import RagSettings
from service.backend.app.rag.sources import load_markdown_documents
from service.backend.app.rag.vector_store import PgvectorStoreError, _constraint_sql_parts
from service.backend.app.schemas import MetadataInput


def test_deterministic_embedding_is_384_dimensional() -> None:
    model = DeterministicTextEmbeddingModel(vector_dim=384)

    embedding = model.embed_query("코로나 방전 HFCT 근거")

    assert len(embedding) == 384
    assert abs(sum(value * value for value in embedding) - 1.0) < 1e-9


def test_markdown_sources_load_rulebook_and_sop_documents() -> None:
    documents = load_markdown_documents()

    source_types = {document.source_type for document in documents}

    assert "rulebook" in source_types
    assert "sop" in source_types
    assert any(document.title == "코로나 방전 판단 기준" for document in documents)
    assert any(document.title == "부분방전 기본 개념" for document in documents)
    assert any(document.title == "보이드 방전 기본 개념" for document in documents)


def test_pgvector_rag_adapter_falls_back_when_store_is_unavailable() -> None:
    settings = RagSettings(
        database_url="postgresql://localhost/unavailable",
        embedding_model="dragonkue/multilingual-e5-small-ko-v2",
        vector_dim=384,
        top_k=6,
        source_types=("rulebook", "sop", "dataset_case"),
        allow_deterministic_fallback=True,
    )
    adapter = PgvectorRagRetrievalAdapter(
        settings=settings,
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_UnavailableStore(),
    )

    result = adapter.run(
        RagToolInput(
            safe_metadata=_metadata(),
            route="hybrid",
            timeseries_result=None,
            vision_result=None,
            similar_case_result=None,
        )
    )

    assert result.retriever_name == "pgvector_rulebook_case_rag"
    assert result.documents
    assert result.documents[0].metadata["fallback"] == "deterministic_local_knowledge"


def test_reranker_promotes_label_and_metadata_match() -> None:
    metadata = _metadata()
    mismatched_high_vector_hit = _hit(
        chunk_key="dataset_case:noise#0",
        relevance=0.9,
        label_id=0,
        sensor_type="UHF",
        source_type="dataset_case",
    )
    matched_rulebook_hit = _hit(
        chunk_key="rulebook:corona#0",
        relevance=0.78,
        label_id=3,
        sensor_type="HFCT",
        source_type="rulebook",
    )

    ranked = rerank_hits(
        [mismatched_high_vector_hit, matched_rulebook_hit],
        metadata=metadata,
        candidate_label_ids=(3,),
        top_k=2,
    )

    assert ranked[0].hit.chunk_key == "rulebook:corona#0"
    assert ranked[0].score > ranked[1].score


def test_reranker_normalizes_korean_concept_suffixes() -> None:
    partial_discharge_hit = _hit(
        chunk_key="rulebook:partial-discharge#0",
        relevance=0.2,
        label_id=0,
        sensor_type="HFCT",
        source_type="rulebook",
        title="부분방전 기본 개념",
        text="부분방전은 절연 내부 또는 표면에서 국부적으로 발생하는 방전 현상이다.",
    )
    prpd_hit = _hit(
        chunk_key="rulebook:prpd#0",
        relevance=0.21,
        label_id=0,
        sensor_type="HFCT",
        source_type="rulebook",
        title="PRPD 이미지 기본 개념",
        text="PRPD는 위상에 따른 부분방전 펄스 분포이다.",
    )

    ranked = rerank_hits(
        [prpd_hit, partial_discharge_hit],
        metadata=None,
        candidate_label_ids=(),
        top_k=2,
        query_text="부분방전이란 무엇이지?",
    )

    assert ranked[0].hit.chunk_key == "rulebook:partial-discharge#0"


def test_text_search_infers_label_from_compact_korean_query() -> None:
    corona_hit = _hit(
        chunk_key="dataset_case:corona#0",
        relevance=0.04,
        label_id=3,
        sensor_type="HFCT",
        source_type="dataset_case",
        text="label=코로나 방전 sensor=HFCT",
    )
    noise_hit = _hit(
        chunk_key="dataset_case:noise#0",
        relevance=0.2,
        label_id=1,
        sensor_type="HFCT",
        source_type="dataset_case",
        text="label=노이즈 sensor=HFCT",
    )

    ranked = rerank_hits(
        [noise_hit, corona_hit],
        metadata=None,
        candidate_label_ids=infer_label_ids_from_text("코로나방전"),
        top_k=2,
        query_text="코로나방전",
    )

    assert ranked[0].hit.chunk_key == "dataset_case:corona#0"
    assert ranked[0].score >= 0.35


def test_text_search_merges_inferred_label_candidates() -> None:
    settings = RagSettings(
        database_url="postgresql://localhost/test",
        embedding_model="deterministic",
        vector_dim=384,
        top_k=2,
        source_types=("rulebook", "dataset_case"),
        allow_deterministic_fallback=False,
    )
    adapter = PgvectorRagRetrievalAdapter(
        settings=settings,
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_LabelAwareStore(),
    )

    documents = adapter.search_text("보이드 방전", top_k=2)

    assert documents[0].title == "보이드 사례"
    assert documents[0].metadata["label_id"] == 4


def test_text_search_returns_documents_when_query_logging_fails() -> None:
    settings = RagSettings(
        database_url="postgresql://localhost/test",
        embedding_model="deterministic",
        vector_dim=384,
        top_k=2,
        source_types=("rulebook", "dataset_case"),
        allow_deterministic_fallback=False,
    )
    adapter = PgvectorRagRetrievalAdapter(
        settings=settings,
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_LogFailingLabelStore(),
    )

    documents = adapter.search_text("보이드 방전", top_k=2)

    assert documents[0].title == "보이드 사례"


def test_text_search_routes_rulebook_query_to_rulebook_source() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_source_routing_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_SourceRoutingStore(("rulebook",), "rulebook", "코로나 방전 판단 기준"),
    )

    documents = adapter.search_text("코로나 방전 판단 기준", top_k=1)

    assert documents[0].source_type == "rulebook"
    assert documents[0].retrieval_mode == "rulebook_semantic"
    assert documents[0].metadata["retrieval_mode"] == "rulebook_semantic"


def test_text_search_routes_sop_query_to_sop_source() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_source_routing_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_SourceRoutingStore(("sop",), "sop", "노이즈 재측정 SOP"),
    )

    documents = adapter.search_text("노이즈 재측정 SOP 절차", top_k=1)

    assert documents[0].source_type == "sop"
    assert documents[0].retrieval_mode == "sop_semantic"


def test_text_search_can_limit_concept_query_to_knowledge_sources() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_source_routing_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_SourceRoutingStore(("rulebook", "sop"), "rulebook", "보이드 방전 개념"),
    )

    documents = adapter.search_text("보이드방전이란?", top_k=1, source_types=("rulebook", "sop"))

    assert documents[0].source_type == "rulebook"


def test_peak_query_constraint_handles_compact_korean_text() -> None:
    constraints = extract_query_constraints("피크가 82인데이터보여줘")

    assert constraints.peak_value == 82


def test_peak_query_constraint_handles_spaced_korean_text() -> None:
    constraints = extract_query_constraints("피크 값이 82인 데이터")

    assert constraints.peak_value == 82


def test_voltage_query_constraint_handles_korean_text() -> None:
    constraints = extract_query_constraints("전압 21000V 보여줘")

    assert constraints.voltage_value == 21000


def test_voltage_query_constraint_normalizes_kilovolts() -> None:
    constraints = extract_query_constraints("전압 21kV 보여줘")

    assert constraints.voltage_value == 21000


def test_insulator_query_constraint_handles_korean_text() -> None:
    constraints = extract_query_constraints("절연 액체 상태보여줘")

    assert constraints.insulator_type == "액체"


def test_insulator_query_constraint_handles_short_spaced_text() -> None:
    constraints = extract_query_constraints("절연 액체")

    assert constraints.insulator_type == "액체"


def test_insulator_constraint_sql_uses_chunk_column_and_metadata() -> None:
    constraints = extract_query_constraints("절연 액체")

    clauses, params = _constraint_sql_parts(constraints)

    assert "c.insulator_type" in clauses[0]
    assert "metadata->>'insulator_type'" in clauses[0]
    assert params == ["액체", "액체"]


def test_equipment_query_constraint_handles_common_typo() -> None:
    constraints = extract_query_constraints("단상 유압변압기인 설비 보여줘")

    assert constraints.equipment_name == "단상 유입변압기"


def test_metadata_terms_handle_recording_time() -> None:
    constraints = extract_query_constraints("231006_183134기록시간 데이터보여줘")

    assert constraints.recording_time == "231006_183134"
    assert constraints.metadata_terms == ()


def test_environment_query_constraints_handle_korean_text() -> None:
    constraints = extract_query_constraints("습도 65 와 온도 19 이면서 기록시간이 230910_195248 인것 보여줘")

    assert constraints.humidity_value == 65
    assert constraints.temperature_value == 19
    assert constraints.recording_time == "230910_195248"
    assert constraints.metadata_terms == ()


def test_environment_query_constraints_handle_unit_only_text() -> None:
    constraints = extract_query_constraints("19도 65% HFCT 기록시간 230910_195248 데이터 보여줘")

    assert constraints.temperature_value == 19
    assert constraints.humidity_value == 65
    assert constraints.sensor_type == "HFCT"
    assert constraints.recording_time == "230910_195248"


def test_applied_query_filters_describe_complex_constraints() -> None:
    constraints = extract_query_constraints("습도 65 와 온도 19 이면서 HFCT 기록시간이 230910_195248 인것 보여줘")

    filters = applied_query_filters(constraints)

    assert [(item.key, item.value) for item in filters] == [
        ("temperature_value", "19도"),
        ("humidity_value", "65%"),
        ("sensor_type", "HFCT"),
        ("recording_time", "230910_195248"),
    ]


def test_or_query_constraints_split_into_groups() -> None:
    constraints = extract_query_constraints("기록시간 230910_195248 또는 기록시간 230910_195243 보여줘")

    assert constraints.has_or_groups
    assert [group.recording_time for group in constraints.or_groups] == ["230910_195248", "230910_195243"]


def test_power_frequency_query_constraint_ignores_hz_case_and_command_text() -> None:
    constraints = extract_query_constraints("전원주파수 60HZ 보여줘")

    assert constraints.power_frequency_value == 60


def test_metadata_term_match_requires_every_term() -> None:
    hit = _hit(
        chunk_key="dataset_case:metadata-terms#0",
        relevance=0.8,
        label_id=1,
        sensor_type="HFCT",
        source_type="dataset_case",
        text="recording_time=230910_195248 sensor=HFCT",
        metadata={"recording_time": "230910_195248", "sensor_type": "HFCT"},
    )

    assert hit_contains_metadata_term(hit, ("230910195248", "hfct"))
    assert not hit_contains_metadata_term(hit, ("230910195248", "1500"))


def test_metadata_terms_ignore_general_domain_question() -> None:
    constraints = extract_query_constraints("부분방전이란 무엇이지?")

    assert constraints.metadata_terms == ()
    assert constraints.has_constraints is False


def test_label_query_constraint_distinguishes_label_from_insulator() -> None:
    constraints = extract_query_constraints("라벨은 기체 보여줘")

    assert constraints.label_name == "기체"


def test_sample_id_constraint_handles_korean_suffix() -> None:
    constraints = extract_query_constraints("노이즈_고체_ACSR-OC_230910_195250_HFCT_500인 샘플ID보여줘")

    assert constraints.sample_id == "노이즈_고체_ACSR-OC_230910_195250_HFCT_500"


def test_sample_id_constraint_handles_mixed_korean_equipment_name() -> None:
    constraints = extract_query_constraints("노이즈_기체_22.9kV배전반_231004_111120_HFCT_1200 샘플ID")

    assert constraints.sample_id == "노이즈_기체_22.9kV배전반_231004_111120_HFCT_1200"


def test_sample_id_pattern_covers_manifest_cases() -> None:
    misses = [
        case.sample_id
        for case in dataset_case_repository.cases
        if extract_sample_id(case.sample_id) != case.sample_id
    ]

    assert misses == []


def test_text_search_returns_only_exact_sample_id_match() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_ExactSampleIdStore(),
    )

    documents = adapter.search_text("노이즈_고체_ACSR-OC_230910_195250_HFCT_500인 샘플ID보여줘", top_k=3)

    assert len(documents) == 1
    assert documents[0].metadata["sample_id"] == "노이즈_고체_ACSR-OC_230910_195250_HFCT_500"
    assert documents[0].retrieval_mode == "exact_sample_id"
    assert documents[0].relevance == 1.0


def test_text_search_returns_empty_for_invalid_label_name() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_LabelNameAwareStore(),
    )

    documents = adapter.search_text("라벨 기체 보여줘", top_k=3)

    assert documents == []


def test_text_search_filters_by_label_name() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_LabelNameAwareStore(),
    )

    documents = adapter.search_text("라벨 노이즈 보여줘", top_k=3)

    assert documents
    assert {document.metadata["label_name"] for document in documents} == {"노이즈"}
    assert {document.retrieval_mode for document in documents} == {"metadata_filter"}


def test_text_search_filters_by_peak_constraint() -> None:
    settings = RagSettings(
        database_url="postgresql://localhost/test",
        embedding_model="deterministic",
        vector_dim=384,
        top_k=3,
        source_types=("dataset_case",),
        allow_deterministic_fallback=False,
    )
    adapter = PgvectorRagRetrievalAdapter(
        settings=settings,
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_PeakAwareStore(),
    )

    documents = adapter.search_text("피크 값이 82인 데이터", top_k=3)

    assert len(documents) == 2
    assert {document.retrieval_mode for document in documents} == {"metadata_filter"}
    assert {document.metadata["max_discharge_value"] for document in documents} == {"82"}


def test_text_search_filters_by_voltage_constraint() -> None:
    settings = RagSettings(
        database_url="postgresql://localhost/test",
        embedding_model="deterministic",
        vector_dim=384,
        top_k=3,
        source_types=("dataset_case",),
        allow_deterministic_fallback=False,
    )
    adapter = PgvectorRagRetrievalAdapter(
        settings=settings,
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_VoltageAwareStore(),
    )

    documents = adapter.search_text("전압 21000V 보여줘", top_k=3)

    assert len(documents) == 2
    assert {document.metadata["equipment_rated_voltage"] for document in documents} == {"21000V"}


def test_text_search_filters_by_insulator_constraint() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_InsulatorAwareStore(),
    )

    documents = adapter.search_text("절연 액체 상태보여줘", top_k=3)

    assert len(documents) == 2
    assert {document.metadata["insulator_type"] for document in documents} == {"액체"}


def test_text_search_filters_by_short_spaced_insulator_constraint() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_InsulatorAwareStore(),
    )

    documents = adapter.search_text("절연 액체", top_k=3)

    assert len(documents) == 2
    assert {document.metadata["insulator_type"] for document in documents} == {"액체"}


def test_text_search_filters_by_equipment_constraint() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_EquipmentAwareStore(),
    )

    documents = adapter.search_text("단상 유압변압기인 설비 보여줘", top_k=3)

    assert len(documents) == 2
    assert {document.metadata["equipment_name"] for document in documents} == {"단상 유입변압기"}


def test_text_search_filters_by_metadata_term() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_MetadataTermAwareStore(),
    )

    documents = adapter.search_text("231006_183134기록시간 데이터보여줘", top_k=3)

    assert len(documents) == 1
    assert documents[0].metadata["recording_time"] == "231006_183134"


def test_text_search_requires_environment_and_recording_time_constraints() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_EnvironmentAwareStore(),
    )

    documents = adapter.search_text("습도 65 와 온도 19 이면서 기록시간이 230910_195248 인것 보여줘", top_k=3)

    assert len(documents) == 1
    assert documents[0].metadata["recording_time"] == "230910_195248"
    assert documents[0].metadata["temperature"] == "19"
    assert documents[0].metadata["humidity"] == "65"


def test_text_search_requires_unit_only_environment_sensor_and_recording_time_constraints() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_EnvironmentAwareStore(),
    )

    documents = adapter.search_text("19도 65% HFCT 기록시간 230910_195248 데이터 보여줘", top_k=3)

    assert len(documents) == 1
    assert documents[0].metadata["recording_time"] == "230910_195248"
    assert documents[0].metadata["temperature"] == "19"
    assert documents[0].metadata["humidity"] == "65"
    assert documents[0].metadata["sensor_type"] == "HFCT"


def test_text_search_matches_any_or_constraint_group() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_OrRecordingTimeStore(),
    )

    documents = adapter.search_text("기록시간 230910_195248 또는 기록시간 230910_195243 보여줘", top_k=3)

    assert [document.metadata["recording_time"] for document in documents] == ["230910_195248", "230910_195243"]


def test_text_search_excludes_label_mismatch_hits_when_query_has_label() -> None:
    adapter = PgvectorRagRetrievalAdapter(
        settings=_constraint_test_settings(),
        embedding_model=DeterministicTextEmbeddingModel(vector_dim=384),
        store=_LabelMismatchStore(),
    )

    documents = adapter.search_text("코로나 방전 사례 보여줘", top_k=3)

    assert documents
    assert {document.metadata["label_name"] for document in documents} == {"코로나 방전"}


class _UnavailableStore:
    def has_indexed_chunks(self) -> bool:
        raise PgvectorStoreError("database unavailable")

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        raise PgvectorStoreError("database unavailable")

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        raise AssertionError("log_query should not run when search fails")


class _LabelAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:corona#0",
                relevance=0.35,
                label_id=3,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="label=코로나 방전",
                title="코로나 사례",
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        assert label_ids == (4,)
        return [
            _hit(
                chunk_key="dataset_case:void#0",
                relevance=0.05,
                label_id=4,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="label=보이드 방전",
                title="보이드 사례",
            )
        ]

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        return []

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _LogFailingLabelStore(_LabelAwareStore):
    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        raise PgvectorStoreError("query log unavailable")


class _SourceRoutingStore:
    def __init__(self, expected_source_types: tuple[str, ...], source_type: str, title: str) -> None:
        self.expected_source_types = expected_source_types
        self.source_type = source_type
        self.title = title

    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        assert source_types == self.expected_source_types
        return [
            _hit(
                chunk_key=f"{self.source_type}:routed#0",
                relevance=0.8,
                label_id=3,
                sensor_type="HFCT",
                source_type=self.source_type,
                text=self.title,
                title=self.title,
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        assert source_types == self.expected_source_types
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        assert source_types == self.expected_source_types
        return []

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        assert source_types == self.expected_source_types
        return []

    def search_by_sample_id(self, source_types: tuple[str, ...], sample_id: str):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        assert metadata is not None
        assert metadata["source_types"] == list(self.expected_source_types)


class _LabelNameAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:gas-insulator#0",
                relevance=0.9,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="label=노이즈 insulator=기체",
                title="절연 기체 사례",
                metadata={"label_name": "노이즈", "insulator_type": "기체"},
                insulator_type="기체",
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        if getattr(constraints, "label_name", None) != "노이즈":
            return []
        return [
            _hit(
                chunk_key="dataset_case:noise-label#0",
                relevance=0.85,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="label=노이즈",
                title="라벨 노이즈 사례",
                metadata={"label_name": "노이즈", "insulator_type": "고체"},
            )
        ]

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def search_by_sample_id(self, source_types: tuple[str, ...], sample_id: str):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _PeakAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:noise-90#0",
                relevance=0.9,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="max_discharge=90",
                title="피크 90 사례",
                metadata={"max_discharge_value": "90"},
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:noise-82-a#0",
                relevance=0.2,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="max_discharge=82",
                title="피크 82 사례 A",
                metadata={"max_discharge_value": "82"},
            ),
            _hit(
                chunk_key="dataset_case:noise-82-b#0",
                relevance=0.18,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="max_discharge=82",
                title="피크 82 사례 B",
                metadata={"max_discharge_value": "82"},
            ),
        ]

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _VoltageAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:voltage-22900#0",
                relevance=0.9,
                label_id=2,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="voltage=22900V",
                title="전압 22900V 사례",
                metadata={"equipment_rated_voltage": "22900V"},
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:voltage-21000-a#0",
                relevance=0.2,
                label_id=2,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="voltage=21000V",
                title="전압 21000V 사례 A",
                metadata={"equipment_rated_voltage": "21000V"},
            ),
            _hit(
                chunk_key="dataset_case:voltage-21000-b#0",
                relevance=0.18,
                label_id=2,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="voltage=21000V",
                title="전압 21000V 사례 B",
                metadata={"equipment_rated_voltage": "21000V"},
            ),
        ]

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _InsulatorAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:gas#0",
                relevance=0.9,
                label_id=3,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="insulator=기체",
                title="기체 절연 사례",
                metadata={"insulator_type": "기체"},
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:liquid-a#0",
                relevance=0.2,
                label_id=2,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="insulator=액체",
                title="액체 절연 사례 A",
                metadata={"insulator_type": "액체"},
                insulator_type="액체",
            ),
            _hit(
                chunk_key="dataset_case:liquid-b#0",
                relevance=0.18,
                label_id=2,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="insulator=액체",
                title="액체 절연 사례 B",
                metadata={"insulator_type": "액체"},
                insulator_type="액체",
            ),
        ]

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _EquipmentAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:power-transformer#0",
                relevance=0.9,
                label_id=2,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="equipment=전력용 유입변압기",
                title="전력용 유입변압기 사례",
                metadata={"equipment_name": "전력용 유입변압기"},
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:single-transformer-a#0",
                relevance=0.2,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="equipment=단상 유입변압기",
                title="단상 유입변압기 사례 A",
                metadata={"equipment_name": "단상 유입변압기"},
            ),
            _hit(
                chunk_key="dataset_case:single-transformer-b#0",
                relevance=0.18,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="equipment=단상 유입변압기",
                title="단상 유입변압기 사례 B",
                metadata={"equipment_name": "단상 유입변압기"},
            ),
        ]

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _MetadataTermAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:other-time#0",
                relevance=0.9,
                label_id=3,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="recording_time=231006_183135",
                title="다른 기록시각 사례",
                metadata={"recording_time": "231006_183135"},
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        return []

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return [
            _hit(
                chunk_key="dataset_case:target-time#0",
                relevance=0.85,
                label_id=3,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="recording_time=231006_183134",
                title="대상 기록시각 사례",
                metadata={"recording_time": "231006_183134"},
            )
        ]

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _EnvironmentAwareStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return self._candidate_hits()

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        assert getattr(constraints, "temperature_value") == 19
        assert getattr(constraints, "humidity_value") == 65
        if getattr(constraints, "sensor_type") is not None:
            assert getattr(constraints, "sensor_type") == "HFCT"
        return self._candidate_hits()

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        assert metadata_terms == ("230910195248",)
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None

    def _candidate_hits(self) -> list[RagSearchHit]:
        return [
            self._target_hit(),
            _hit(
                chunk_key="dataset_case:wrong-time-a#0",
                relevance=0.88,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="recording_time=230910_195243 temperature=19 humidity=65 power_frequency=60Hz",
                title="다른 기록시각 사례 A",
                metadata={
                    "recording_time": "230910_195243",
                    "temperature": "19",
                    "humidity": "65",
                    "sensor_type": "HFCT",
                    "power_supply_frequency": "60Hz",
                },
            ),
            _hit(
                chunk_key="dataset_case:wrong-time-b#0",
                relevance=0.87,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="recording_time=230910_195250 temperature=19 humidity=65 power_frequency=60Hz",
                title="다른 기록시각 사례 B",
                metadata={
                    "recording_time": "230910_195250",
                    "temperature": "19",
                    "humidity": "65",
                    "sensor_type": "HFCT",
                    "power_supply_frequency": "60Hz",
                },
            ),
        ]

    def _target_hit(self) -> RagSearchHit:
        return _hit(
            chunk_key="dataset_case:target-environment#0",
            relevance=0.91,
            label_id=1,
            sensor_type="HFCT",
            source_type="dataset_case",
            text="recording_time=230910_195248 temperature=19 humidity=65 power_frequency=60Hz",
            title="대상 환경 조건 사례",
            metadata={
                "recording_time": "230910_195248",
                "temperature": "19",
                "humidity": "65",
                "sensor_type": "HFCT",
                "power_supply_frequency": "60Hz",
            },
        )


class _OrRecordingTimeStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return self._candidate_hits()

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        return []

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        assert [group.recording_time for group in getattr(constraints, "or_groups")] == [
            "230910_195248",
            "230910_195243",
        ]
        return self._candidate_hits()

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None

    def _candidate_hits(self) -> list[RagSearchHit]:
        return [
            _recording_time_hit("230910_195248", 0.91),
            _recording_time_hit("230910_195243", 0.89),
            _recording_time_hit("230910_195250", 0.95),
        ]


class _LabelMismatchStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        return [
            _hit(
                chunk_key="dataset_case:noise-high#0",
                relevance=0.95,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="label=노이즈",
                title="노이즈 사례",
                metadata={"label_name": "노이즈"},
            )
        ]

    def search_by_label_ids(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        label_ids: tuple[int, ...],
        top_k: int,
    ):
        assert label_ids == (3,)
        return [
            _hit(
                chunk_key="dataset_case:corona-low#0",
                relevance=0.2,
                label_id=3,
                sensor_type="HFCT",
                source_type="dataset_case",
                text="label=코로나 방전",
                title="코로나 사례",
                metadata={"label_name": "코로나 방전"},
            )
        ]

    def search_by_constraints(self, query_embedding: list[float], source_types: tuple[str, ...], constraints: object, top_k: int):
        return []

    def search_by_metadata_terms(
        self,
        query_embedding: list[float],
        source_types: tuple[str, ...],
        metadata_terms: tuple[str, ...],
        top_k: int,
    ):
        return []

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        return None


class _ExactSampleIdStore:
    def has_indexed_chunks(self) -> bool:
        return True

    def search_by_sample_id(self, source_types: tuple[str, ...], sample_id: str):
        assert source_types == ("dataset_case",)
        assert sample_id == "노이즈_고체_ACSR-OC_230910_195250_HFCT_500"
        return [
            _hit(
                chunk_key=f"dataset_case:{sample_id}#0",
                relevance=1.0,
                label_id=1,
                sensor_type="HFCT",
                source_type="dataset_case",
                text=f"sample_id={sample_id}",
                title=f"데이터셋 사례 {sample_id}",
                metadata={"sample_id": sample_id},
            )
        ]

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        assert metadata is not None
        assert metadata["retrieval_mode"] == "exact_sample_id"


def _constraint_test_settings() -> RagSettings:
    return RagSettings(
        database_url="postgresql://localhost/test",
        embedding_model="deterministic",
        vector_dim=384,
        top_k=3,
        source_types=("dataset_case",),
        allow_deterministic_fallback=False,
    )


def _source_routing_settings() -> RagSettings:
    return RagSettings(
        database_url="postgresql://localhost/test",
        embedding_model="deterministic",
        vector_dim=384,
        top_k=3,
        source_types=("rulebook", "sop", "dataset_case"),
        allow_deterministic_fallback=False,
    )


def _metadata() -> MetadataInput:
    return MetadataInput(
        equipment_name="ACSR-OC",
        equipment_rated_voltage="22900V",
        equipment_rated_current="268A",
        sensor_type="HFCT",
        temperature=19,
        humidity=66,
        insulator_type="고체",
    )


def _hit(
    chunk_key: str,
    relevance: float,
    label_id: int,
    sensor_type: str,
    source_type: str,
    text: str = "테스트 본문",
    title: str = "테스트 근거",
    metadata: dict[str, object] | None = None,
    insulator_type: str = "고체",
) -> RagSearchHit:
    return RagSearchHit(
        chunk_key=chunk_key,
        document_key=chunk_key.split("#", 1)[0],
        source_type=source_type,
        title=title,
        text=text,
        source=chunk_key,
        relevance=relevance,
        label_id=label_id,
        sensor_type=sensor_type,
        equipment_type="가공선",
        insulator_type=insulator_type,
        metadata=metadata or {},
    )


def _recording_time_hit(recording_time: str, relevance: float) -> RagSearchHit:
    return _hit(
        chunk_key=f"dataset_case:{recording_time}#0",
        relevance=relevance,
        label_id=1,
        sensor_type="HFCT",
        source_type="dataset_case",
        text=f"recording_time={recording_time}",
        title=f"기록시각 {recording_time} 사례",
        metadata={"recording_time": recording_time},
    )
