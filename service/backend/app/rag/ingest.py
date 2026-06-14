from __future__ import annotations

from service.backend.app.rag.embeddings import build_embedding_model
from service.backend.app.rag.settings import RagSettings
from service.backend.app.rag.sources import load_dataset_case_documents, load_markdown_documents
from service.backend.app.rag.vector_store import PgvectorRagStore


def initialize_rag_database(settings: RagSettings | None = None) -> None:
    resolved_settings = settings or RagSettings.from_env()
    PgvectorRagStore(resolved_settings.database_url).initialize_schema()


def ingest_rag_sources(settings: RagSettings | None = None, dataset_limit: int | None = None) -> int:
    resolved_settings = settings or RagSettings.from_env()
    embedding_model = build_embedding_model(resolved_settings)
    documents = [
        *load_markdown_documents(),
        *load_dataset_case_documents(limit=dataset_limit),
    ]
    return PgvectorRagStore(resolved_settings.database_url).ingest_documents(documents, embedding_model)
