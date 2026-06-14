from __future__ import annotations

from dataclasses import dataclass

from service.backend.app.schemas import MetadataInput, RagDocument, TimeSeriesResult, VisionResult


@dataclass(frozen=True, slots=True)
class KnowledgeSnippet:
    document_id: str
    title: str
    source: str
    excerpt: str
    label_ids: tuple[int, ...]
    sensor_terms: tuple[str, ...] = ()


KNOWLEDGE_SNIPPETS: tuple[KnowledgeSnippet, ...] = (
    KnowledgeSnippet(
        document_id="pd-normal-baseline",
        title="정상 PRPD 기준 패턴",
        source="pd_rulebook:v0",
        excerpt="정상 감시 상태에서는 위상에 연동된 군집이 뚜렷하지 않고 펄스 크기가 노이즈 플로어 근처에 머무릅니다.",
        label_ids=(0,),
    ),
    KnowledgeSnippet(
        document_id="pd-noise-phase-uniformity",
        title="노이즈성 PRPD 선별",
        source="pd_rulebook:v0",
        excerpt="전 위상에 균일한 분포, 넓은 수평 밴드, 약한 반주기 대칭성은 PD 분류 전에 노이즈 또는 취득 아티팩트로 우선 검토해야 합니다.",
        label_ids=(1,),
        sensor_terms=("hfct", "uhf", "tev"),
    ),
    KnowledgeSnippet(
        document_id="pd-surface-discharge",
        title="표면 방전 근거",
        source="pd_rulebook:v0",
        excerpt="표면 방전은 위상 집중과 극성 비대칭을 보이는 경우가 많으며 절연 오염 또는 트래킹 흔적과 교차 확인해야 합니다.",
        label_ids=(2,),
    ),
    KnowledgeSnippet(
        document_id="pd-corona-discharge",
        title="코로나 방전 근거",
        source="pd_rulebook:v0",
        excerpt="코로나 방전은 전압 피크 부근의 위상 국부 로브로 나타나는 경우가 많아 날카로운 모서리와 고전계 접속부 점검이 필요합니다.",
        label_ids=(3,),
        sensor_terms=("hfct", "uhf"),
    ),
    KnowledgeSnippet(
        document_id="pd-internal-void",
        title="내부 보이드 방전 상향 조치",
        source="pd_rulebook:v0",
        excerpt="내부 보이드 방전은 고위험 절연 결함 패턴이므로 높은 신뢰도의 근거가 지속되면 확인 시험으로 상향 조치해야 합니다.",
        label_ids=(4,),
    ),
    KnowledgeSnippet(
        document_id="pd-human-review-disagreement",
        title="모델 판단 불일치 처리",
        source="operator_sop:v0",
        excerpt="신호, 비전, 리포트 경로가 서로 다른 판단을 내리면 자동 확정을 보류하고 재측정 또는 엔지니어 검토를 요청합니다.",
        label_ids=(),
    ),
)
DEFAULT_RAG_LIMIT = 3


def retrieve_pd_knowledge(
    metadata: MetadataInput | None,
    ts_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
) -> tuple[str, list[RagDocument]]:
    label_ids = _candidate_label_ids(ts_result, vision_result)
    sensor = metadata.sensor_type.lower() if metadata is not None else ""
    equipment = metadata.equipment_name if metadata is not None else "unknown equipment"
    query = _build_query(equipment, sensor, label_ids)
    scored = [
        _to_document(snippet, _score_snippet(snippet, label_ids, sensor))
        for snippet in KNOWLEDGE_SNIPPETS
    ]
    scored.sort(key=lambda document: document.relevance, reverse=True)
    return query, scored[:DEFAULT_RAG_LIMIT]


def _candidate_label_ids(ts_result: TimeSeriesResult | None, vision_result: VisionResult | None) -> tuple[int, ...]:
    labels = []
    if ts_result is not None:
        labels.append(ts_result.label_id)
    if vision_result is not None:
        labels.append(vision_result.label_id)
    return tuple(dict.fromkeys(labels))


def _build_query(equipment: str, sensor: str, label_ids: tuple[int, ...]) -> str:
    labels = ",".join(str(label_id) for label_id in label_ids) if label_ids else "unknown"
    sensor_term = sensor or "unknown_sensor"
    return f"equipment={equipment}; sensor={sensor_term}; candidate_labels={labels}"


def _score_snippet(snippet: KnowledgeSnippet, label_ids: tuple[int, ...], sensor: str) -> float:
    score = 0.35
    if snippet.label_ids and any(label_id in snippet.label_ids for label_id in label_ids):
        score += 0.42
    if sensor and sensor in snippet.sensor_terms:
        score += 0.12
    if snippet.document_id == "pd-human-review-disagreement":
        score += 0.42 if len(label_ids) > 1 else 0.05
    return min(score, 0.99)


def _to_document(snippet: KnowledgeSnippet, relevance: float) -> RagDocument:
    return RagDocument(
        document_id=snippet.document_id,
        title=snippet.title,
        source=snippet.source,
        excerpt=snippet.excerpt,
        relevance=round(relevance, 3),
    )
