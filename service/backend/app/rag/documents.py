from __future__ import annotations

from dataclasses import dataclass, field


SourceType = str


@dataclass(frozen=True, slots=True)
class RagSourceDocument:
    document_key: str
    source_type: SourceType
    title: str
    text: str
    source_path: str | None = None
    label_id: int | None = None
    sensor_type: str | None = None
    equipment_type: str | None = None
    insulator_type: str | None = None
    source_ref: str | None = None
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RagChunk:
    chunk_key: str
    document_key: str
    source_type: SourceType
    title: str
    text: str
    chunk_index: int
    source_path: str | None
    label_id: int | None
    sensor_type: str | None
    equipment_type: str | None
    insulator_type: str | None
    source_ref: str | None
    metadata: dict[str, str | int | float | None]


@dataclass(frozen=True, slots=True)
class RagSearchHit:
    chunk_key: str
    document_key: str
    source_type: SourceType
    title: str
    text: str
    source: str
    relevance: float
    label_id: int | None
    sensor_type: str | None
    equipment_type: str | None
    insulator_type: str | None
    metadata: dict[str, object]


def chunk_document(document: RagSourceDocument, max_chars: int = 900) -> list[RagChunk]:
    paragraphs = [paragraph.strip() for paragraph in document.text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return [
        RagChunk(
            chunk_key=f"{document.document_key}#{index}",
            document_key=document.document_key,
            source_type=document.source_type,
            title=document.title,
            text=text,
            chunk_index=index,
            source_path=document.source_path,
            label_id=document.label_id,
            sensor_type=document.sensor_type,
            equipment_type=document.equipment_type,
            insulator_type=document.insulator_type,
            source_ref=document.source_ref,
            metadata=document.metadata,
        )
        for index, text in enumerate(chunks)
    ]
