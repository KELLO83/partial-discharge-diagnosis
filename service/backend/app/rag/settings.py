from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5432/partial_discharge_diagnosis"
DEFAULT_EMBEDDING_MODEL = "dragonkue/multilingual-e5-small-ko-v2"
DEFAULT_VECTOR_DIM = 384
DEFAULT_TOP_K = 6
DEFAULT_SOURCE_TYPES = ("rulebook", "dataset_case")
PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOCAL_ENV_FILES = (
    PROJECT_ROOT / "service" / "backend" / ".env",
    PROJECT_ROOT / ".env",
)


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
        _load_local_env_files()
        return cls(
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            embedding_model=os.getenv("RAG_TEXT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            vector_dim=_env_int("RAG_VECTOR_DIM", DEFAULT_VECTOR_DIM),
            top_k=_env_int("RAG_TOP_K", DEFAULT_TOP_K),
            source_types=_env_tuple("RAG_SOURCE_TYPES", DEFAULT_SOURCE_TYPES),
            allow_deterministic_fallback=_env_bool("RAG_ALLOW_DETERMINISTIC_FALLBACK", True),
            force_deterministic_embeddings=_env_bool("RAG_FORCE_DETERMINISTIC_EMBEDDINGS", False),
        )


def _load_local_env_files() -> None:
    for env_path in LOCAL_ENV_FILES:
        _load_env_file(env_path)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    clean_key = key.strip()
    if not clean_key:
        return None
    return clean_key, _strip_env_quotes(value.strip())


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


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
