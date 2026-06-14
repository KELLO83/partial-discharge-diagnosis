from __future__ import annotations

from service.backend.app.rag.embeddings import DeterministicTextEmbeddingModel
from service.backend.app.rag.documents import RagSearchHit
from service.backend.app.rag.reranker import rerank_hits
from service.backend.app.rag.retriever import PgvectorRagRetrievalAdapter
from service.backend.app.rag.settings import RagSettings
from service.backend.app.rag.sources import load_markdown_documents
from service.backend.app.rag.vector_store import PgvectorStoreError
from service.backend.app.schemas import MetadataInput
from service.backend.app.application.contracts import RagToolInput


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


class _UnavailableStore:
    def has_indexed_chunks(self) -> bool:
        raise PgvectorStoreError("database unavailable")

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int):
        raise PgvectorStoreError("database unavailable")

    def log_query(self, query_text: str, hits: list[object], metadata: dict[str, object] | None = None, diagnosis_id: str | None = None) -> None:
        raise AssertionError("log_query should not run when search fails")


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
) -> RagSearchHit:
    return RagSearchHit(
        chunk_key=chunk_key,
        document_key=chunk_key.split("#", 1)[0],
        source_type=source_type,
        title="테스트 근거",
        text="테스트 본문",
        source=chunk_key,
        relevance=relevance,
        label_id=label_id,
        sensor_type=sensor_type,
        equipment_type="가공선",
        insulator_type="고체",
        metadata={},
    )
