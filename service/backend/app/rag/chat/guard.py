from __future__ import annotations

from service.backend.app.rag.chat.constants import (
    ANSWER_MODE_OUT_OF_SCOPE,
    LOCAL_CHAT_MODEL,
    NON_DOMAIN_CHAT_ANSWER,
)
from service.backend.app.rag.chat.intent import classify_question
from service.backend.app.schemas import RagChatResponse


def should_skip_retrieval(question: str) -> bool:
    return not classify_question(question).should_retrieve


def build_non_domain_response() -> RagChatResponse:
    return RagChatResponse(
        answer=NON_DOMAIN_CHAT_ANSWER,
        documents=[],
        model=LOCAL_CHAT_MODEL,
        ready=True,
        answer_mode=ANSWER_MODE_OUT_OF_SCOPE,
    )
