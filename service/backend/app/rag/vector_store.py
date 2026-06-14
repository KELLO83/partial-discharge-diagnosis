from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from service.backend.app.rag.documents import RagChunk, RagSearchHit, RagSourceDocument, chunk_document
from service.backend.app.rag.embeddings import TextEmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "service" / "backend" / "db" / "rag_schema.sql"


class PgvectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PendingRagChunk:
    document_id: int
    chunk: RagChunk


@dataclass(frozen=True, slots=True)
class ChunkInsert:
    connection: Any
    pending_chunk: PendingRagChunk
    embedding: list[float]


class PgvectorRagStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = _normalize_database_url(database_url)

    def initialize_schema(self, schema_path: Path = DEFAULT_SCHEMA_PATH) -> None:
        sql = schema_path.read_text(encoding="utf-8")
        with self._connect() as connection:
            connection.execute(sql)
            connection.commit()

    def ingest_documents(self, documents: list[RagSourceDocument], embedding_model: TextEmbeddingModel) -> int:
        with self._connect() as connection:
            pending_chunks: list[PendingRagChunk] = []
            for document in documents:
                document_id = self._upsert_document(connection, document)
                connection.execute("DELETE FROM rag.chunks WHERE document_id = %s", (document_id,))
                for chunk in chunk_document(document):
                    pending_chunks.append(PendingRagChunk(document_id, chunk))
            embeddings = embedding_model.embed_passages([pending.chunk.text for pending in pending_chunks])
            for pending_chunk, embedding in zip(pending_chunks, embeddings):
                self._insert_chunk(ChunkInsert(connection, pending_chunk, embedding))
            connection.commit()
        return len(pending_chunks)

    def has_indexed_chunks(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT EXISTS (SELECT 1 FROM rag.chunks LIMIT 1)").fetchone()
        return bool(row and row[0])

    def status(self) -> dict[str, object]:
        with self._connect() as connection:
            extension = connection.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            document_count = connection.execute("SELECT COUNT(*) FROM rag.documents").fetchone()
            chunk_count = connection.execute("SELECT COUNT(*) FROM rag.chunks").fetchone()
            query_log_count = connection.execute("SELECT COUNT(*) FROM rag.query_logs").fetchone()
            source_rows = connection.execute(
                """
                SELECT d.source_type, COUNT(DISTINCT d.id), COUNT(c.id)
                FROM rag.documents d
                LEFT JOIN rag.chunks c ON c.document_id = d.id
                GROUP BY d.source_type
                ORDER BY d.source_type
                """
            ).fetchall()
        return {
            "database_name": _database_name(self.database_url),
            "vector_extension": str(extension[0]) if extension else None,
            "document_count": _count_value(document_count),
            "chunk_count": _count_value(chunk_count),
            "query_log_count": _count_value(query_log_count),
            "source_counts": {
                str(row[0]): {"documents": int(row[1]), "chunks": int(row[2])}
                for row in source_rows
            },
        }

    def list_documents(self, source_type: str | None = None, limit: int = 50) -> list[dict[str, object]]:
        safe_limit = max(1, min(limit, 200))
        where_clause = "WHERE d.source_type = %s" if source_type else ""
        params: list[Any] = [source_type] if source_type else []
        params.append(safe_limit)
        sql = f"""
            SELECT
                d.document_key,
                d.source_type,
                d.title,
                d.source_path,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM rag.documents d
            LEFT JOIN rag.chunks c ON c.document_id = d.id
            {where_clause}
            GROUP BY d.id
            ORDER BY d.updated_at DESC, d.document_key
            LIMIT %s
        """
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_document_row(row) for row in rows]

    def recent_query_logs(self, limit: int = 20) -> list[dict[str, object]]:
        safe_limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, diagnosis_id, query_text, query_metadata, retrieved_chunks, created_at
                FROM rag.query_logs
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (safe_limit,),
            ).fetchall()
        return [_query_log_row(row) for row in rows]

    def search(self, query_embedding: list[float], source_types: tuple[str, ...], top_k: int) -> list[RagSearchHit]:
        if top_k <= 0:
            return []
        placeholders = ", ".join(["%s"] * len(source_types))
        sql = f"""
            SELECT
                c.chunk_key,
                d.document_key,
                d.source_type,
                d.title,
                c.chunk_text,
                COALESCE(c.source_ref, d.source_path, d.document_key) AS source,
                1 - (c.embedding <=> %s::vector) AS relevance,
                c.label_id,
                c.sensor_type,
                c.equipment_type,
                c.insulator_type,
                c.metadata
            FROM rag.chunks c
            JOIN rag.documents d ON d.id = c.document_id
            WHERE d.source_type IN ({placeholders})
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        vector = _vector_literal(query_embedding)
        params: list[Any] = [vector, *source_types, vector, top_k]
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_row_to_hit(row) for row in rows]

    def log_query(
        self,
        query_text: str,
        hits: list[RagSearchHit],
        metadata: dict[str, object] | None = None,
        diagnosis_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rag.query_logs (diagnosis_id, query_text, query_metadata, retrieved_chunks)
                VALUES (%s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    diagnosis_id,
                    query_text,
                    _json_literal(metadata or {}),
                    _json_literal([_hit_metadata(hit) for hit in hits]),
                ),
            )
            connection.commit()

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise PgvectorStoreError("psycopg is not installed") from exc
        try:
            return psycopg.connect(self.database_url)
        except Exception as exc:
            raise PgvectorStoreError(str(exc)) from exc

    @staticmethod
    def _upsert_document(connection: Any, document: RagSourceDocument) -> int:
        row = connection.execute(
            """
            INSERT INTO rag.documents (document_key, source_type, title, source_path, metadata, updated_at)
            VALUES (%s, %s, %s, %s, %s::jsonb, now())
            ON CONFLICT (document_key) DO UPDATE
            SET source_type = EXCLUDED.source_type,
                title = EXCLUDED.title,
                source_path = EXCLUDED.source_path,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING id
            """,
            (
                document.document_key,
                document.source_type,
                document.title,
                document.source_path,
                _json_literal(document.metadata),
            ),
        ).fetchone()
        if row is None:
            raise PgvectorStoreError(f"document upsert failed: {document.document_key}")
        return int(row[0])

    @staticmethod
    def _insert_chunk(insert: ChunkInsert) -> None:
        chunk = insert.pending_chunk.chunk
        insert.connection.execute(
            """
            INSERT INTO rag.chunks (
                chunk_key, document_id, chunk_index, chunk_text, embedding,
                label_id, sensor_type, equipment_type, insulator_type, source_ref, metadata
            )
            VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (chunk_key) DO NOTHING
            """,
            (
                chunk.chunk_key,
                insert.pending_chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                _vector_literal(insert.embedding),
                chunk.label_id,
                chunk.sensor_type,
                chunk.equipment_type,
                chunk.insulator_type,
                chunk.source_ref,
                _json_literal(chunk.metadata),
            ),
        )


def _normalize_database_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _database_name(database_url: str) -> str:
    parsed = urlparse(database_url)
    return parsed.path.lstrip("/") or "unknown"


def _count_value(row: tuple[Any, ...] | None) -> int:
    return 0 if row is None else int(row[0])


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def _json_literal(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _row_to_hit(row: tuple[Any, ...]) -> RagSearchHit:
    metadata = row[11] if isinstance(row[11], dict) else {}
    return RagSearchHit(
        chunk_key=str(row[0]),
        document_key=str(row[1]),
        source_type=str(row[2]),
        title=str(row[3]),
        text=str(row[4]),
        source=str(row[5]),
        relevance=round(float(row[6]), 4),
        label_id=int(row[7]) if row[7] is not None else None,
        sensor_type=str(row[8]) if row[8] is not None else None,
        equipment_type=str(row[9]) if row[9] is not None else None,
        insulator_type=str(row[10]) if row[10] is not None else None,
        metadata=metadata,
    )


def _document_row(row: tuple[Any, ...]) -> dict[str, object]:
    updated_at = row[4].isoformat() if row[4] is not None else ""
    return {
        "document_key": str(row[0]),
        "source_type": str(row[1]),
        "title": str(row[2]),
        "source_path": str(row[3]) if row[3] is not None else None,
        "updated_at": updated_at,
        "chunk_count": int(row[5]),
    }


def _query_log_row(row: tuple[Any, ...]) -> dict[str, object]:
    created_at = row[5].isoformat() if row[5] is not None else ""
    return {
        "id": int(row[0]),
        "diagnosis_id": str(row[1]) if row[1] is not None else None,
        "query_text": str(row[2]),
        "query_metadata": row[3] if isinstance(row[3], dict) else {},
        "retrieved_chunks": row[4] if isinstance(row[4], list) else [],
        "created_at": created_at,
    }


def _hit_metadata(hit: RagSearchHit) -> dict[str, object]:
    return {
        "chunk_key": hit.chunk_key,
        "document_key": hit.document_key,
        "source_type": hit.source_type,
        "title": hit.title,
        "source": hit.source,
        "relevance": hit.relevance,
        "label_id": hit.label_id,
        "sensor_type": hit.sensor_type,
        "equipment_type": hit.equipment_type,
        "insulator_type": hit.insulator_type,
    }
