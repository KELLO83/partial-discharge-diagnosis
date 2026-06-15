from __future__ import annotations

import os
from dataclasses import dataclass

from service.backend.app.config.env import load_project_env

DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5432/partial_discharge_diagnosis"
DEFAULT_EMBEDDING_MODEL = "dragonkue/multilingual-e5-small-ko-v2"
DEFAULT_VECTOR_DIM = 384
DEFAULT_TOP_K = 6
DEFAULT_SOURCE_TYPES = ("rulebook", "sop", "dataset_case")


@dataclass(frozen=True, slots=True)
class RagSettings:
    database_url: str
    embedding_model: str
    vector_dim: int
    top_k: int
    source_types: tuple[str, ...]
    allow_deterministic_fallback: bool
    force_deterministic_embeddings: bool = False

    @classmethod
    def from_env(cls) -> "RagSettings":
        load_project_env()
        return cls(
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            embedding_model=os.getenv("RAG_TEXT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            vector_dim=_env_int("RAG_VECTOR_DIM", DEFAULT_VECTOR_DIM),
            top_k=_env_int("RAG_TOP_K", DEFAULT_TOP_K),
            source_types=_env_tuple("RAG_SOURCE_TYPES", DEFAULT_SOURCE_TYPES),
            allow_deterministic_fallback=_env_bool("RAG_ALLOW_DETERMINISTIC_FALLBACK", True),
            force_deterministic_embeddings=_env_bool("RAG_FORCE_DETERMINISTIC_EMBEDDINGS", False),
        )


def _env_int(name: str, fallback: int) -> int:
    value = os.getenv(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def _env_tuple(name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return fallback
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or fallback


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}
