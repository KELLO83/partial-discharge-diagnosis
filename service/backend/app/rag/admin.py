from __future__ import annotations

from urllib.parse import urlparse

from service.backend.app.rag.embeddings import build_embedding_model
from service.backend.app.rag.retriever import PgvectorRagRetrievalAdapter
from service.backend.app.rag.settings import RagSettings
from service.backend.app.rag.sources import load_dataset_case_documents, load_markdown_documents
from service.backend.app.rag.vector_store import PgvectorRagStore, PgvectorStoreError
from service.backend.app.schemas import (
    RagDocumentListItem,
    RagDocumentListResponse,
    RagQueryLogItem,
    RagQueryLogResponse,
    RagReindexResponse,
    RagSearchResponse,
    RagStatusResponse,
)


def read_rag_status(settings: RagSettings | None = None) -> RagStatusResponse:
    active_settings = settings or RagSettings.from_env()
    store = PgvectorRagStore(active_settings.database_url)
    try:
        status = store.status()
    except Exception as exc:
        return _offline_status(active_settings, str(exc))
    return RagStatusResponse(
        ready=bool(status["vector_extension"]) and int(status["chunk_count"]) > 0,
        database_name=str(status["database_name"]),
        vector_extension=status["vector_extension"] if isinstance(status["vector_extension"], str) else None,
        embedding_model=active_settings.embedding_model,
        vector_dim=active_settings.vector_dim,
        top_k=active_settings.top_k,
        source_types=list(active_settings.source_types),
        document_count=int(status["document_count"]),
        chunk_count=int(status["chunk_count"]),
        query_log_count=int(status["query_log_count"]),
        source_counts=status["source_counts"],  # type: ignore[arg-type]
    )


def list_rag_documents(source_type: str | None = None, limit: int = 50) -> RagDocumentListResponse:
    store = PgvectorRagStore(RagSettings.from_env().database_url)
    try:
        rows = store.list_documents(source_type=source_type, limit=limit)
    except Exception as exc:
        return RagDocumentListResponse(items=[], error=str(exc))
    return RagDocumentListResponse(items=[RagDocumentListItem.model_validate(row) for row in rows])


def list_rag_query_logs(limit: int = 20) -> RagQueryLogResponse:
    store = PgvectorRagStore(RagSettings.from_env().database_url)
    try:
        rows = store.recent_query_logs(limit=limit)
    except Exception as exc:
        return RagQueryLogResponse(items=[], error=str(exc))
    return RagQueryLogResponse(items=[RagQueryLogItem.model_validate(row) for row in rows])


def search_rag_documents(query: str, top_k: int) -> RagSearchResponse:
    adapter = PgvectorRagRetrievalAdapter()
    try:
        documents = adapter.search_text(query=query, top_k=top_k)
    except (PgvectorStoreError, RuntimeError) as exc:
        return RagSearchResponse(query=query, documents=[], error=str(exc))
    return RagSearchResponse(query=query, documents=documents)


def reindex_rag_documents(dataset_limit: int | None = None) -> RagReindexResponse:
    settings = RagSettings.from_env()
    store = PgvectorRagStore(settings.database_url)
    documents = [
        *load_markdown_documents(),
        *load_dataset_case_documents(limit=dataset_limit),
    ]
    store.initialize_schema()
    chunk_count = store.ingest_documents(documents, build_embedding_model(settings))
    return RagReindexResponse(
        document_count=len(documents),
        chunk_count=chunk_count,
        dataset_limit=dataset_limit,
        embedding_model=settings.embedding_model,
    )


def _offline_status(settings: RagSettings, error: str) -> RagStatusResponse:
    return RagStatusResponse(
        ready=False,
        database_name=_database_name(settings.database_url),
        embedding_model=settings.embedding_model,
        vector_dim=settings.vector_dim,
        top_k=settings.top_k,
        source_types=list(settings.source_types),
        document_count=0,
        chunk_count=0,
        query_log_count=0,
        error=error,
    )


def _database_name(database_url: str) -> str:
    return urlparse(database_url).path.lstrip("/") or "unknown"
