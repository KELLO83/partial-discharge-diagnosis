from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from service.backend.app.rag.chat.constants import (
    ANSWER_MODE_GENERAL_DOMAIN,
    ANSWER_MODE_OUT_OF_SCOPE,
    ANSWER_MODE_RAG_EVIDENCE,
    DOMAIN_TERMS,
    GREETING_QUERIES,
)
from service.backend.app.rag.query_constraints import extract_query_constraints


CONCEPT_EVIDENCE_SOURCE_TYPES = ("rulebook", "sop")
CONCEPT_QUERY_MARKERS = (
    "이란",
    "란",
    "무엇",
    "뭐",
    "뜻",
    "정의",
    "개념",
    "설명",
)
EVIDENCE_QUERY_MARKERS = (
    "사례",
    "데이터",
    "샘플",
    "근거",
    "검색",
    "찾아",
    "비교",
    "판정",
    "기준",
    "위험",
    "조치",
)


class QueryIntentKind(StrEnum):
    concept_explanation = "concept_explanation"
    evidence_search = "evidence_search"
    out_of_scope = "out_of_scope"


@dataclass(frozen=True, slots=True)
class QueryIntent:
    kind: QueryIntentKind
    answer_mode: str
    source_types: tuple[str, ...] | None = None

    @property
    def should_retrieve(self) -> bool:
        return self.kind != QueryIntentKind.out_of_scope


def classify_question(question: str) -> QueryIntent:
    normalized_question = compact_text(question)
    if is_out_of_scope(normalized_question) and not extract_query_constraints(question).has_constraints:
        return QueryIntent(kind=QueryIntentKind.out_of_scope, answer_mode=ANSWER_MODE_OUT_OF_SCOPE)
    if is_concept_question(normalized_question):
        return QueryIntent(
            kind=QueryIntentKind.concept_explanation,
            answer_mode=ANSWER_MODE_GENERAL_DOMAIN,
            source_types=CONCEPT_EVIDENCE_SOURCE_TYPES,
        )
    return QueryIntent(kind=QueryIntentKind.evidence_search, answer_mode=ANSWER_MODE_RAG_EVIDENCE)


def is_out_of_scope(normalized_question: str) -> bool:
    if normalized_question in GREETING_QUERIES:
        return True
    return not any(compact_text(term) in normalized_question for term in DOMAIN_TERMS)


def is_concept_question(normalized_question: str) -> bool:
    if any(marker in normalized_question for marker in EVIDENCE_QUERY_MARKERS):
        return False
    return any(marker in normalized_question for marker in CONCEPT_QUERY_MARKERS)


def compact_text(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
