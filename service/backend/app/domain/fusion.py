from __future__ import annotations

from collections import Counter

from service.backend.app.schemas import (
    EvidenceFactor,
    FusionSummary,
    RagResult,
    SimilarCaseResult,
    StandardModelEvidence,
    TimeSeriesResult,
    VisionResult,
    VlmResult,
)


def time_series_evidence(result: TimeSeriesResult) -> StandardModelEvidence:
    return StandardModelEvidence(
        source="time_series",
        model_name=result.model_name,
        model_version=result.model_version,
        label_id=result.label_id,
        label_name=result.label_name,
        confidence=result.confidence,
        uncertainty=round(1.0 - result.confidence, 6),
        ood_score=None,
        top_factors=[
            EvidenceFactor(
                name="abs_p99",
                value=result.features.get("abs_p99"),
                weight=0.36,
                explanation="상위 진폭 분위값이 방전 강도 판단에 사용되었습니다.",
            ),
            EvidenceFactor(
                name="pulse_rate",
                value=result.features.get("pulse_rate"),
                weight=0.30,
                explanation="펄스 반복률이 방전 활동성 판단에 사용되었습니다.",
            ),
            EvidenceFactor(
                name="spectral_energy",
                value=result.features.get("spectral_energy"),
                weight=0.22,
                explanation="주파수 에너지 요약값이 시계열 패턴 근거로 사용되었습니다.",
            ),
        ],
        explanation=f"시계열 모델은 {result.label_name} 가능성을 {result.confidence:.0%}로 판단했습니다.",
    )


def vision_evidence(result: VisionResult) -> StandardModelEvidence:
    ood_score = _float_or_none(result.evidence.get("ood_score"))
    return StandardModelEvidence(
        source="vision",
        model_name=result.model_name,
        model_version=result.model_version,
        label_id=result.label_id,
        label_name=result.label_name,
        confidence=result.confidence,
        uncertainty=round(1.0 - result.confidence, 6),
        ood_score=ood_score,
        top_factors=[
            EvidenceFactor(
                name="phase_localization_score",
                value=result.evidence.get("phase_localization_score"),
                weight=0.40,
                explanation="특정 위상 구간 집중도가 PRPD 형태 판단에 사용되었습니다.",
            ),
            EvidenceFactor(
                name="band_like_noise_score",
                value=result.evidence.get("band_like_noise_score"),
                weight=0.24,
                explanation="대역형 노이즈 가능성이 실제 방전 여부 판단에 반영되었습니다.",
            ),
            EvidenceFactor(
                name="ood_score",
                value=ood_score,
                weight=0.18,
                explanation="학습 분포 밖 패턴 가능성을 보류 판단 근거로 사용합니다.",
            ),
        ],
        explanation=str(
            result.evidence.get(
                "visual_evidence_summary",
                f"비전 모델은 {result.label_name} 가능성을 {result.confidence:.0%}로 판단했습니다.",
            )
        ),
    )


def vlm_evidence(result: VlmResult) -> StandardModelEvidence:
    return StandardModelEvidence(
        source="vlm",
        model_name=result.model_name,
        model_version=result.model_version,
        label_id=result.label_id,
        label_name=result.diagnosis,
        confidence=result.confidence,
        uncertainty=round(1.0 - result.confidence, 6),
        ood_score=None,
        top_factors=[
            EvidenceFactor(
                name="reason",
                value=result.reason,
                weight=0.42,
                explanation="모델 근거와 검색 근거를 종합한 설명입니다.",
            ),
            EvidenceFactor(
                name="recommended_action",
                value=result.recommended_action,
                weight=0.24,
                explanation="현장 조치 권고 문장입니다.",
            ),
        ],
        explanation=result.reason,
    )


def similar_case_evidence(result: SimilarCaseResult | None) -> StandardModelEvidence | None:
    if result is None or not result.cases:
        return None
    top_case = result.cases[0]
    return StandardModelEvidence(
        source="similar_case",
        model_name=result.retriever_name,
        model_version=result.retriever_version,
        label_id=top_case.label_id,
        label_name=top_case.label_name,
        confidence=top_case.similarity,
        uncertainty=round(1.0 - top_case.similarity, 6),
        ood_score=None,
        top_factors=[
            EvidenceFactor(
                name="top_similarity",
                value=top_case.similarity,
                weight=0.45,
                explanation="가장 가까운 과거 사례와의 유사도입니다.",
            ),
            EvidenceFactor(
                name="case_count",
                value=len(result.cases),
                weight=0.20,
                explanation="현재 조건에서 검색된 과거 참조 사례 수입니다.",
            ),
        ],
        explanation=f"가장 유사한 과거 사례는 {top_case.sample_id}이며 라벨은 {top_case.label_name}입니다.",
    )


def rag_evidence(result: RagResult | None) -> StandardModelEvidence | None:
    if result is None:
        return None
    top_document = result.documents[0] if result.documents else None
    return StandardModelEvidence(
        source="rag",
        model_name=result.retriever_name,
        model_version=result.retriever_version,
        label_id=None,
        label_name=None,
        confidence=top_document.relevance if top_document is not None else None,
        uncertainty=None,
        ood_score=None,
        top_factors=[
            EvidenceFactor(
                name="document_count",
                value=len(result.documents),
                weight=0.25,
                explanation="진단 근거로 검색된 규정/절차 문서 수입니다.",
            ),
            EvidenceFactor(
                name="reference_case_count",
                value=len(result.similar_cases),
                weight=0.25,
                explanation="RAG 컨텍스트에 포함된 과거 사례 수입니다.",
            ),
            EvidenceFactor(
                name="top_document",
                value=top_document.title if top_document is not None else None,
                weight=0.20,
                explanation="가장 관련도가 높은 문서 근거입니다.",
            ),
        ],
        explanation="규정/절차 문서와 유사 사례를 함께 검색해 리포트 근거로 사용했습니다.",
    )


def build_fusion_summary(
    ts_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
    similar_case_result: SimilarCaseResult | None,
    rag_result: RagResult | None,
    vlm_result: VlmResult | None,
) -> FusionSummary:
    evidence = _collect_evidence(ts_result, vision_result, similar_case_result, rag_result, vlm_result)
    label_votes = [item.label_id for item in evidence if item.label_id is not None]
    agreement_level = _agreement_level(label_votes)
    final_label_id = _final_label_id(label_votes, vlm_result)
    final_label_name = _label_name_for(final_label_id, evidence)
    confidence = _average_confidence(item.confidence for item in evidence if item.confidence is not None)
    return FusionSummary(
        strategy="rule_based_late_fusion_v1",
        final_label_id=final_label_id,
        final_label_name=final_label_name,
        confidence=confidence,
        agreement_level=agreement_level,
        contributing_sources=[item.source for item in evidence],
        rationale=_fusion_rationale(agreement_level, final_label_name, evidence),
        evidence=evidence,
    )


def _collect_evidence(
    ts_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
    similar_case_result: SimilarCaseResult | None,
    rag_result: RagResult | None,
    vlm_result: VlmResult | None,
) -> list[StandardModelEvidence]:
    evidence: list[StandardModelEvidence] = []
    if ts_result is not None:
        evidence.append(ts_result.standard_evidence or time_series_evidence(ts_result))
    if vision_result is not None:
        evidence.append(vision_result.standard_evidence or vision_evidence(vision_result))
    case_evidence = similar_case_evidence(similar_case_result)
    if case_evidence is not None:
        evidence.append(case_evidence)
    rag_item = rag_evidence(rag_result)
    if rag_item is not None:
        evidence.append(rag_item)
    if vlm_result is not None:
        evidence.append(vlm_result.standard_evidence or vlm_evidence(vlm_result))
    return evidence


def _agreement_level(label_votes: list[int]) -> str:
    if not label_votes:
        return "none"
    if len(label_votes) == 1:
        return "single_source"
    counts = Counter(label_votes)
    if len(counts) == 1:
        return "agreement"
    if counts.most_common(1)[0][1] >= 2:
        return "partial_agreement"
    return "conflict"


def _final_label_id(label_votes: list[int], vlm_result: VlmResult | None) -> int | None:
    if vlm_result is not None:
        return vlm_result.label_id
    if not label_votes:
        return None
    return Counter(label_votes).most_common(1)[0][0]


def _label_name_for(label_id: int | None, evidence: list[StandardModelEvidence]) -> str | None:
    if label_id is None:
        return None
    for item in evidence:
        if item.label_id == label_id:
            return item.label_name
    return str(label_id)


def _average_confidence(values: object) -> float | None:
    numbers = [value for value in values if isinstance(value, float | int)]
    if not numbers:
        return None
    return round(sum(float(value) for value in numbers) / len(numbers), 6)


def _fusion_rationale(
    agreement_level: str,
    final_label_name: str | None,
    evidence: list[StandardModelEvidence],
) -> str:
    source_count = len(evidence)
    label_text = final_label_name or "미정"
    if agreement_level == "agreement":
        return f"{source_count}개 근거가 {label_text} 판단에 일치합니다."
    if agreement_level == "partial_agreement":
        return f"일부 근거가 {label_text} 판단에 일치하지만 추가 검토가 필요합니다."
    if agreement_level == "conflict":
        return "모델/검색 근거 간 라벨 충돌이 있어 운영자 검토가 필요합니다."
    return f"{source_count}개 근거를 기반으로 {label_text} 판단을 구성했습니다."


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
