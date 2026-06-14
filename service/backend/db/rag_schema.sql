CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.documents (
    id BIGSERIAL PRIMARY KEY,
    document_key TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL CHECK (source_type IN ('rulebook', 'sop', 'dataset_case')),
    title TEXT NOT NULL,
    source_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag.chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_key TEXT NOT NULL UNIQUE,
    document_id BIGINT NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    label_id INTEGER,
    sensor_type TEXT,
    equipment_type TEXT,
    insulator_type TEXT,
    source_ref TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag.query_logs (
    id BIGSERIAL PRIMARY KEY,
    diagnosis_id TEXT,
    query_text TEXT NOT NULL,
    query_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieved_chunks JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_documents_source_type_idx
    ON rag.documents (source_type);

CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx
    ON rag.chunks (document_id);

CREATE INDEX IF NOT EXISTS rag_chunks_label_id_idx
    ON rag.chunks (label_id);

CREATE INDEX IF NOT EXISTS rag_chunks_sensor_type_idx
    ON rag.chunks (sensor_type);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
    ON rag.chunks USING hnsw (embedding vector_cosine_ops);
