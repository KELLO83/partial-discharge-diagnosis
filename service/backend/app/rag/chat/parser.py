from __future__ import annotations

import json

from pydantic import BaseModel, Field


class RagChatPayload(BaseModel):
    answer: str = Field(min_length=1)


def parse_rag_chat_payload(content: str) -> RagChatPayload:
    return RagChatPayload.model_validate(json.loads(extract_json_object(content)))


def extract_json_object(content: str) -> str:
    clean_content = content.strip()
    if clean_content.startswith("```"):
        clean_content = clean_content.strip("`").strip()
        if clean_content.lower().startswith("json"):
            clean_content = clean_content[4:].strip()

    json_start = clean_content.find("{")
    json_end = clean_content.rfind("}")
    if json_start < 0 or json_end <= json_start:
        raise ValueError("RAG chat response did not contain a JSON object")
    return clean_content[json_start : json_end + 1]
