from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from service.backend.app.rag.openrouter_client import OpenRouterSettings
from service.backend.app.schemas import DiagnosisDetailResponse, RagChatMessage, RagDocument


class ChatCompleter(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


class RagTextRetriever(Protocol):
    def search_text(
        self,
        query: str,
        top_k: int,
        source_types: tuple[str, ...] | None = None,
    ) -> list[RagDocument]:
        ...


class DiagnosisHistoryStore(Protocol):
    def detail(self, diagnosis_id: str) -> DiagnosisDetailResponse | None:
        ...


@dataclass(frozen=True, slots=True)
class RagChatInput:
    question: str
    history: list[RagChatMessage]
    top_k: int


@dataclass(frozen=True, slots=True)
class RagChatDependencies:
    retriever: RagTextRetriever | None = None
    diagnosis_store: DiagnosisHistoryStore | None = None
    client: ChatCompleter | None = None
    settings: OpenRouterSettings | None = None
