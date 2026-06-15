from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from prpd_similarity_retrieval import FEATURE_SCHEMA_VERSION
from prpd_similarity_retrieval.features import extract_case_features
from prpd_similarity_retrieval.models import CaseFeatures, CaseRecord, SearchResult


IMAGE_WEIGHT = 0.55
TIMESERIES_WEIGHT = 0.45
METADATA_WEIGHT = 0.0
LABEL_WEIGHT = 0.0
METADATA_BASELINE_WEIGHT = 0.75
LABEL_BASELINE_WEIGHT = 0.25
METADATA_WEIGHTS = {
    "equipment_name": 0.25,
    "sensor_type": 0.20,
    "insulator_type": 0.18,
    "clearance_distance": 0.14,
    "equipment_rated_voltage": 0.14,
    "equipment_rated_current": 0.09,
}


def build_feature_index(cases: list[CaseRecord]) -> list[CaseFeatures]:
    return [extract_case_features(case) for case in cases]


def save_feature_index(path: Path, cases: list[CaseFeatures]) -> None:
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "case_count": len(cases),
        "cases": [case.to_dict() for case in cases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_index(path: Path) -> list[CaseFeatures]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature schema: {payload.get('schema_version')}")
    return [CaseFeatures.from_dict(item) for item in payload.get("cases", [])]


def find_case(cases: list[CaseFeatures], sample_id: str) -> CaseFeatures:
    for case in cases:
        if case.sample_id == sample_id:
            return case
    raise ValueError(f"sample_id not found in index: {sample_id}")


def search_similar_cases(
    query: CaseFeatures,
    candidates: list[CaseFeatures],
    top_k: int = 5,
    exclude_self: bool = True,
) -> list[SearchResult]:
    return _search_with_scorer(query, candidates, _score_candidate, top_k, exclude_self)


def search_metadata_baseline(
    query: CaseFeatures,
    candidates: list[CaseFeatures],
    top_k: int = 5,
    exclude_self: bool = True,
) -> list[SearchResult]:
    return _search_with_scorer(query, candidates, _score_metadata_candidate, top_k, exclude_self)


def _search_with_scorer(
    query: CaseFeatures,
    candidates: list[CaseFeatures],
    scorer: Callable[[CaseFeatures, CaseFeatures], SearchResult],
    top_k: int,
    exclude_self: bool,
) -> list[SearchResult]:
    results = [
        scorer(query, candidate)
        for candidate in candidates
        if not (exclude_self and candidate.sample_id == query.sample_id)
    ]
    results.sort(key=lambda item: (item.score, item.case.sample_id), reverse=True)
    return results[:top_k]


def _score_candidate(query: CaseFeatures, candidate: CaseFeatures) -> SearchResult:
    image_score = _cosine_similarity(query.image_vector, candidate.image_vector)
    timeseries_score = _cosine_similarity(query.timeseries_vector, candidate.timeseries_vector)
    weighted_scores = [
        (image_score, IMAGE_WEIGHT),
        (timeseries_score, TIMESERIES_WEIGHT),
    ]
    score = _weighted_average(weighted_scores)
    return SearchResult(
        case=candidate,
        score=score,
        image_score=image_score,
        timeseries_score=timeseries_score,
        metadata_score=None,
        label_score=None,
        reason=_reason(image_score, timeseries_score, None, None),
    )


def _score_metadata_candidate(query: CaseFeatures, candidate: CaseFeatures) -> SearchResult:
    metadata_score = _metadata_similarity(query.metadata, candidate.metadata)
    label_score = _label_similarity(query.label_id, candidate.label_id)
    score = _weighted_average(
        [
            (metadata_score, METADATA_BASELINE_WEIGHT),
            (label_score, LABEL_BASELINE_WEIGHT),
        ]
    )
    return SearchResult(
        case=candidate,
        score=score,
        image_score=None,
        timeseries_score=None,
        metadata_score=metadata_score,
        label_score=label_score,
        reason=_reason(None, None, metadata_score, label_score),
    )


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 0.0:
        return None
    return _clip_01(float(np.dot(left_array, right_array) / denominator))


def _metadata_similarity(left: dict[str, str], right: dict[str, str]) -> float | None:
    total_weight = 0.0
    score = 0.0
    for field, weight in METADATA_WEIGHTS.items():
        left_value = _normalize_metadata(left.get(field, ""))
        right_value = _normalize_metadata(right.get(field, ""))
        if left_value == "" or right_value == "":
            continue
        total_weight += weight
        if left_value == right_value:
            score += weight
    if total_weight == 0.0:
        return None
    return score / total_weight


def _label_similarity(left: int | None, right: int | None) -> float | None:
    if left is None or right is None:
        return None
    return 1.0 if left == right else 0.0


def _weighted_average(weighted_scores: list[tuple[float | None, float]]) -> float:
    available = [(score, weight) for score, weight in weighted_scores if score is not None]
    if not available:
        return 0.0
    total_weight = sum(weight for _, weight in available)
    return sum(score * weight for score, weight in available) / total_weight


def _reason(
    image_score: float | None,
    timeseries_score: float | None,
    metadata_score: float | None,
    label_score: float | None,
) -> str:
    reasons: list[str] = []
    if image_score is not None and image_score >= 0.82:
        reasons.append("PRPD 패턴 유사")
    if timeseries_score is not None and timeseries_score >= 0.82:
        reasons.append("시계열 파형 유사")
    return ", ".join(reasons) if reasons else "PRPD/시계열 feature 기준 근접 사례"


def _normalize_metadata(value: str) -> str:
    return value.lower().replace(" ", "").replace("['", "").replace("']", "").replace("mm", "")


def _clip_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def results_to_json(results: list[SearchResult]) -> str:
    payload: dict[str, Any] = {"results": [result.to_dict() for result in results]}
    return json.dumps(payload, ensure_ascii=False, indent=2)
