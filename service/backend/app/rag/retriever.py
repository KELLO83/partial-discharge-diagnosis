from __future__ import annotations

from service.backend.app.domain.knowledge import retrieve_pd_knowledge
from service.backend.app.domain.policy import MIN_CONFIDENCE
from service.backend.app.rag.documents import RagSearchHit
from service.backend.app.rag.embeddings import TextEmbeddingModel, build_embedding_model
from service.backend.app.rag.query import RagQueryInput, build_rag_query, candidate_label_ids, infer_label_ids_from_text
from service.backend.app.rag.query_constraints import RagQueryConstraints, extract_query_constraints, filter_hits_by_constraints
from service.backend.app.rag.reranker import RerankedHit, rerank_hits
from service.backend.app.rag.settings import RagSettings
from service.backend.app.rag.vector_store import PgvectorRagStore, PgvectorStoreError
from service.backend.app.schemas import MetadataInput, RagDocument, RagResult
from service.backend.app.application.contracts import RagRetrievalAdapter, RagToolInput


RERANK_CANDIDATE_MULTIPLIER = 5
DATASET_CASE_SOURCE = "dataset_case"
RULEBOOK_SOURCE = "rulebook"
SOP_SOURCE = "sop"
EXACT_SAMPLE_ID_MODE = "exact_sample_id"
METADATA_FILTER_MODE = "metadata_filter"
RULEBOOK_SEMANTIC_MODE = "rulebook_semantic"
SOP_SEMANTIC_MODE = "sop_semantic"
SEMANTIC_SIMILARITY_MODE = "semantic_similarity"
SOP_QUERY_TERMS = (
    "sop",
    "절차",
    "조치",
    "재측정",
    "재시험",
    "검수",
    "리뷰",
    "review",
    "현장",
    "출동",
    "확인",
)
RULEBOOK_QUERY_TERMS = (
    "기준",
    "판단",
    "판정",
    "패턴",
    "특징",
    "근거",
    "분류",
    "식별",
)


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
        source_types: tuple[str, ...] | None = None,
    ) -> list[RagDocument]:
        if not self._store.has_indexed_chunks():
            return []
        active_label_ids = label_ids or infer_label_ids_from_text(query)
        return self._search_documents(
            query=query,
            metadata=metadata,
            label_ids=active_label_ids,
            top_k=top_k,
            configured_source_types=source_types,
        )

    def _search_documents(
        self,
        query: str,
        metadata: MetadataInput | None,
        label_ids: tuple[int, ...],
        top_k: int | None = None,
        configured_source_types: tuple[str, ...] | None = None,
    ) -> list[RagDocument]:
        result_limit = top_k or self.settings.top_k
        candidate_limit = max(result_limit, result_limit * RERANK_CANDIDATE_MULTIPLIER)
        constraints = extract_query_constraints(query)
        exact_sample_hits = self._exact_sample_hits(query, constraints)
        if exact_sample_hits:
            return exact_sample_hits

        retrieval_mode = _retrieval_mode(query, constraints)
        active_source_types = configured_source_types or self.settings.source_types
        source_types = _source_types_for_mode(retrieval_mode, active_source_types)
        embedding = self._embedding().embed_query(query)
        hits = self._store.search(embedding, source_types, candidate_limit)
        hits = _merge_hits(
            hits,
            self._store.search_by_label_ids(
                embedding,
                source_types,
                label_ids,
                candidate_limit,
            ),
        )
        hits = _merge_hits(
            hits,
            self._store.search_by_constraints(
                embedding,
                source_types,
                constraints,
                candidate_limit,
            ),
        )
        hits = _merge_hits(
            hits,
            _search_by_constraint_metadata_terms(
                self._store,
                embedding,
                source_types,
                constraints,
                candidate_limit,
            ),
        )
        hits = _filter_hits_by_candidate_labels(hits, label_ids)
        hits = filter_hits_by_constraints(hits, constraints)
        ranked_hits = rerank_hits(hits, metadata, label_ids, result_limit, query_text=query, constraints=constraints)
        self._log_query(
            query,
            [ranked.hit for ranked in ranked_hits],
            {
                "candidate_label_ids": list(label_ids),
                "candidate_limit": candidate_limit,
                "constraints": {
                    **_constraint_log_metadata(constraints),
                },
                "retrieval_mode": retrieval_mode,
                "reranker": "metadata_label_reranker_v1",
                "source_types": list(source_types),
                "top_k": result_limit,
            },
        )
        return [_document_from_hit(ranked, retrieval_mode) for ranked in ranked_hits]

    def _exact_sample_hits(self, query: str, constraints: RagQueryConstraints) -> list[RagDocument]:
        if constraints.sample_id is None:
            return []
        hits = self._store.search_by_sample_id(self.settings.source_types, constraints.sample_id)
        if not hits:
            return []
        ranked_hits = [RerankedHit(hit=hit, score=1.0) for hit in hits]
        self._log_query(
            query,
            hits,
            {
                "constraints": {"sample_id": constraints.sample_id},
                "retrieval_mode": "exact_sample_id",
                "source_types": list(self.settings.source_types),
                "top_k": 1,
            },
        )
        return [_document_from_hit(ranked, EXACT_SAMPLE_ID_MODE) for ranked in ranked_hits]

    def _log_query(self, query: str, hits: list[RagSearchHit], metadata: dict[str, object]) -> None:
        try:
            self._store.log_query(query, hits, metadata)
        except (PgvectorStoreError, RuntimeError):
            return None

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
                    "retrieval_mode": "deterministic_fallback",
                    "metadata": {
                        "fallback": "deterministic_local_knowledge",
                        "retrieval_mode": "deterministic_fallback",
                    },
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


def _retrieval_mode(query: str, constraints: RagQueryConstraints) -> str:
    if _has_metadata_filter_constraints(constraints):
        return METADATA_FILTER_MODE
    if _is_sop_query(query):
        return SOP_SEMANTIC_MODE
    if _is_rulebook_query(query):
        return RULEBOOK_SEMANTIC_MODE
    return SEMANTIC_SIMILARITY_MODE


def _has_metadata_filter_constraints(constraints: RagQueryConstraints) -> bool:
    if constraints.has_or_groups:
        return any(_has_metadata_filter_constraints(group) for group in constraints.or_groups)
    return (
        constraints.has_label_name
        or constraints.has_peak
        or constraints.has_voltage
        or constraints.has_temperature
        or constraints.has_humidity
        or constraints.has_power_frequency
        or constraints.has_equipment_name
        or constraints.has_sensor_type
        or constraints.has_insulator_type
        or constraints.has_recording_time
        or _has_structured_metadata_terms(constraints.metadata_terms)
    )


def _has_structured_metadata_terms(metadata_terms: tuple[str, ...]) -> bool:
    return any(any(character.isdigit() for character in term) for term in metadata_terms)


def _search_by_constraint_metadata_terms(
    store: PgvectorRagStore,
    embedding: list[float],
    source_types: tuple[str, ...],
    constraints: RagQueryConstraints,
    candidate_limit: int,
) -> list[RagSearchHit]:
    if constraints.has_or_groups:
        hits: list[RagSearchHit] = []
        for group in constraints.or_groups:
            hits = _merge_hits(
                hits,
                _search_by_constraint_metadata_terms(store, embedding, source_types, group, candidate_limit),
            )
        return hits
    return store.search_by_metadata_terms(
        embedding,
        source_types,
        _constraint_metadata_terms(constraints),
        candidate_limit,
    )


def _constraint_metadata_terms(constraints: RagQueryConstraints) -> tuple[str, ...]:
    terms = list(constraints.metadata_terms)
    for value in (constraints.sample_id, constraints.recording_time):
        if value is not None:
            terms.append(_compact_text(value))
    return tuple(dict.fromkeys(term for term in terms if term))


def _constraint_log_metadata(constraints: RagQueryConstraints) -> dict[str, object]:
    if constraints.has_or_groups:
        return {
            "operator": "or",
            "groups": [_constraint_log_metadata(group) for group in constraints.or_groups],
        }
    return {
        "sample_id": constraints.sample_id,
        "label_name": constraints.label_name,
        "peak_value": constraints.peak_value,
        "voltage_value": constraints.voltage_value,
        "temperature_value": constraints.temperature_value,
        "humidity_value": constraints.humidity_value,
        "power_frequency_value": constraints.power_frequency_value,
        "equipment_name": constraints.equipment_name,
        "sensor_type": constraints.sensor_type,
        "insulator_type": constraints.insulator_type,
        "recording_time": constraints.recording_time,
        "metadata_terms": list(constraints.metadata_terms),
    }


def _source_types_for_mode(retrieval_mode: str, configured_source_types: tuple[str, ...]) -> tuple[str, ...]:
    if retrieval_mode == METADATA_FILTER_MODE:
        return _available_sources((DATASET_CASE_SOURCE,), configured_source_types)
    if retrieval_mode == SOP_SEMANTIC_MODE:
        return _available_sources((SOP_SOURCE,), configured_source_types)
    if retrieval_mode == RULEBOOK_SEMANTIC_MODE:
        return _available_sources((RULEBOOK_SOURCE,), configured_source_types)
    return configured_source_types


def _available_sources(preferred_source_types: tuple[str, ...], configured_source_types: tuple[str, ...]) -> tuple[str, ...]:
    available = tuple(source_type for source_type in preferred_source_types if source_type in configured_source_types)
    return available or preferred_source_types


def _is_sop_query(query: str) -> bool:
    compact_query = _compact_text(query)
    return any(_compact_text(term) in compact_query for term in SOP_QUERY_TERMS)


def _is_rulebook_query(query: str) -> bool:
    compact_query = _compact_text(query)
    return any(_compact_text(term) in compact_query for term in RULEBOOK_QUERY_TERMS)


def _compact_text(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _merge_hits(primary_hits: list[RagSearchHit], label_hits: list[RagSearchHit]) -> list[RagSearchHit]:
    merged: list[RagSearchHit] = []
    seen_keys: set[str] = set()
    for hit in [*primary_hits, *label_hits]:
        chunk_key = getattr(hit, "chunk_key", "")
        if chunk_key in seen_keys:
            continue
        merged.append(hit)
        seen_keys.add(chunk_key)
    return merged


def _filter_hits_by_candidate_labels(
    hits: list[RagSearchHit],
    label_ids: tuple[int, ...],
) -> list[RagSearchHit]:
    if not label_ids:
        return hits
    return [
        hit
        for hit in hits
        if hit.source_type != DATASET_CASE_SOURCE or hit.label_id is None or hit.label_id in label_ids
    ]


def _document_from_hit(ranked: RerankedHit, retrieval_mode: str) -> RagDocument:
    hit = ranked.hit
    metadata = _document_metadata(hit.metadata, hit.source_type)
    metadata.update(
        {
            "label_id": hit.label_id,
            "sensor_type": hit.sensor_type,
            "equipment_type": hit.equipment_type,
            "insulator_type": hit.insulator_type,
            "retrieval_mode": retrieval_mode,
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
        retrieval_mode=retrieval_mode,
        metadata=metadata,
    )
