from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

from service.backend.app.rag.settings import RagSettings


class TextEmbeddingModel(Protocol):
    model_name: str
    vector_dim: int

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_passage(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DeterministicTextEmbeddingModel:
    model_name: str = "deterministic_hash_embedding"
    vector_dim: int = 384

    def embed_query(self, text: str) -> list[float]:
        return self._embed(f"query: {text}")

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(f"passage: {text}")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_passage(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.vector_dim)]
        tokens = [token for token in _normalize_text(text).split(" ") if token]
        for token in tokens or [text]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 2):
                index = int.from_bytes(digest[offset:offset + 2], "big") % self.vector_dim
                vector[index] += 1.0
        return _l2_normalize(vector)


class SentenceTransformerTextEmbeddingModel:
    def __init__(self, model_name: str, vector_dim: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model_name = model_name
        self.vector_dim = vector_dim
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(f"query: {text}")

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(f"passage: {text}")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {text}" for text in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True).tolist()
        return [self._validate_vector(vector) for vector in vectors]

    def _embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        return self._validate_vector(vector)

    def _validate_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.vector_dim:
            raise ValueError(f"embedding dimension mismatch: expected {self.vector_dim}, got {len(vector)}")
        return [float(value) for value in vector]


def build_embedding_model(settings: RagSettings) -> TextEmbeddingModel:
    if settings.force_deterministic_embeddings:
        return DeterministicTextEmbeddingModel(vector_dim=settings.vector_dim)
    try:
        return SentenceTransformerTextEmbeddingModel(settings.embedding_model, settings.vector_dim)
    except Exception:
        if not settings.allow_deterministic_fallback:
            raise
        return DeterministicTextEmbeddingModel(vector_dim=settings.vector_dim)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
