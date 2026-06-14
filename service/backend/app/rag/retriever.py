from __future__ import annotations

from service.backend.app.domain.knowledge import retrieve_pd_knowledge
from service.backend.app.domain.policy import MIN_CONFIDENCE
from service.backend.app.rag.embeddings import TextEmbeddingModel, build_embedding_model
from service.backend.app.rag.query import RagQueryInput, build_rag_query, candidate_label_ids
from service.backend.app.rag.reranker import RerankedHit, rerank_hits
from service.backend.app.rag.settings import RagSettings
from service.backend.app.rag.vector_store import PgvectorRagStore, PgvectorStoreError
from service.backend.app.schemas import MetadataInput, RagDocument, RagResult
from service.backend.app.application.contracts import RagRetrievalAdapter, RagToolInput


RERANK_CANDIDATE_MULTIPLIER = 5


class PgvectorRagRetrievalAdapter(RagRetrievalAdapter):
    model_name = "pgvector_rulebook_case_rag"
    model_version = "dragonkue_multilingual_e5_small_ko_v2"

    def __init__(
        self,
        settings: RagSettings | None = None,
        embedding_model: TextEmbeddingModel | None = None,
        store: PgvectorRagStore | None = None,
    ) -> None:
        self.settings = settings or RagSettings.from_env()
        self._embedding_model = embedding_model
        self._store = store or PgvectorRagStore(self.settings.database_url)

    def run(self, tool_input: RagToolInput) -> RagResult:
        query = build_rag_query(
            RagQueryInput(
                metadata=tool_input.safe_metadata,
                time_series=tool_input.timeseries_result,
                vision=tool_input.vision_result,
                similar_case=tool_input.similar_case_result,
            )
        )
        try:
            if not self._store.has_indexed_chunks():
                if not self.settings.allow_deterministic_fallback:
                    raise PgvectorStoreError("RAG index is empty")
                return self._fallback_result(tool_input, query)
            documents = self._search_documents(
                query=query,
                metadata=tool_input.safe_metadata,
                label_ids=candidate_label_ids(tool_input.timeseries_result, tool_input.vision_result),
            )
            if self._needs_fallback(documents):
                documents = self._fallback_documents(tool_input)
        except (PgvectorStoreError, RuntimeError):
            if not self.settings.allow_deterministic_fallback:
                raise
            documents = self._fallback_documents(tool_input)
        return self._result(tool_input, query, documents)

    def _fallback_result(self, tool_input: RagToolInput, query: str) -> RagResult:
        return self._result(tool_input, query, self._fallback_documents(tool_input))

    def _result(self, tool_input: RagToolInput, query: str, documents: list[RagDocument]) -> RagResult:
        return RagResult(
            retriever_name=self.model_name,
            retriever_version=self.model_version,
            query=query,
            documents=documents,
            similar_cases=tool_input.similar_case_result.cases if tool_input.similar_case_result is not None else [],
        )

    def search_text(
        self,
        query: str,
        top_k: int | None = None,
        metadata: MetadataInput | None = None,
        label_ids: tuple[int, ...] = (),
    ) -> list[RagDocument]:
        if not self._store.has_indexed_chunks():
            return []
        return self._search_documents(query=query, metadata=metadata, label_ids=label_ids, top_k=top_k)

    def _search_documents(
        self,
        query: str,
        metadata: MetadataInput | None,
        label_ids: tuple[int, ...],
        top_k: int | None = None,
    ) -> list[RagDocument]:
        result_limit = top_k or self.settings.top_k
        candidate_limit = max(result_limit, result_limit * RERANK_CANDIDATE_MULTIPLIER)
        embedding = self._embedding().embed_query(query)
        hits = self._store.search(embedding, self.settings.source_types, candidate_limit)
        ranked_hits = rerank_hits(hits, metadata, label_ids, result_limit)
        self._store.log_query(
            query,
            [ranked.hit for ranked in ranked_hits],
            {
                "candidate_label_ids": list(label_ids),
                "candidate_limit": candidate_limit,
                "reranker": "metadata_label_reranker_v1",
                "source_types": list(self.settings.source_types),
                "top_k": result_limit,
            },
        )
        return [_document_from_hit(ranked) for ranked in ranked_hits]

    @staticmethod
    def _fallback_documents(tool_input: RagToolInput) -> list[RagDocument]:
        _, documents = retrieve_pd_knowledge(
            tool_input.safe_metadata,
            tool_input.timeseries_result,
            tool_input.vision_result,
        )
        return [
            document.model_copy(
                update={
                    "source_type": "rulebook" if document.source.startswith("pd_rulebook") else "sop",
                    "metadata": {"fallback": "deterministic_local_knowledge"},
                }
            )
            for document in documents
        ]

    def _embedding(self):
        if self._embedding_model is None:
            self._embedding_model = build_embedding_model(self.settings)
        return self._embedding_model

    def _needs_fallback(self, documents: list[RagDocument]) -> bool:
        if not self.settings.allow_deterministic_fallback:
            return False
        if not documents:
            return True
        return max(document.relevance for document in documents) < MIN_CONFIDENCE


def _document_metadata(metadata: dict[str, object], source_type: str) -> dict[str, str | int | float | None]:
    clean: dict[str, str | int | float | None] = {"source_type": source_type}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float)):
            clean[key] = value
    return clean


def _document_from_hit(ranked: RerankedHit) -> RagDocument:
    hit = ranked.hit
    metadata = _document_metadata(hit.metadata, hit.source_type)
    metadata.update(
        {
            "label_id": hit.label_id,
            "sensor_type": hit.sensor_type,
            "equipment_type": hit.equipment_type,
            "insulator_type": hit.insulator_type,
            "vector_relevance": hit.relevance,
            "rerank_score": ranked.score,
        }
    )
    return RagDocument(
        document_id=hit.chunk_key,
        title=hit.title,
        source=hit.source,
        excerpt=hit.text,
        relevance=ranked.score,
        source_type=hit.source_type,
        metadata=metadata,
    )
