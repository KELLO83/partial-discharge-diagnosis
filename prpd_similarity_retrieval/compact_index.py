from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from prpd_similarity_retrieval import FEATURE_SCHEMA_VERSION
from prpd_similarity_retrieval.models import CaseFeatures, SearchResult
from prpd_similarity_retrieval.retrieval import (
    IMAGE_WEIGHT,
    LABEL_WEIGHT,
    METADATA_WEIGHTS,
    METADATA_WEIGHT,
    TIMESERIES_WEIGHT,
    _normalize_metadata,
    _reason,
)


COMPACT_INDEX_SUFFIX = ".npz"
MISSING_SCORE = np.nan


@dataclass(frozen=True, slots=True)
class CompactFeatureIndex:
    cases: list[CaseFeatures]
    image_vectors: np.ndarray
    image_available: np.ndarray
    image_norms: np.ndarray
    timeseries_vectors: np.ndarray
    timeseries_available: np.ndarray
    timeseries_norms: np.ndarray
    metadata_values: dict[str, np.ndarray]
    label_ids: np.ndarray
    label_available: np.ndarray
    sample_tie_ranks: np.ndarray

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def case_at(self, index: int) -> CaseFeatures:
        case = self.cases[index]
        image_vector = _row_to_vector(self.image_vectors, self.image_available, index)
        timeseries_vector = _row_to_vector(self.timeseries_vectors, self.timeseries_available, index)
        return replace(case, image_vector=image_vector, timeseries_vector=timeseries_vector)

    def find_case(self, sample_id: str) -> CaseFeatures:
        for index, case in enumerate(self.cases):
            if case.sample_id == sample_id:
                return self.case_at(index)
        raise ValueError(f"sample_id not found in index: {sample_id}")

    def search_similar_cases(
        self,
        query: CaseFeatures,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        return self._search(query, top_k, exclude_self, include_feature_scores=True)

    def search_metadata_baseline(
        self,
        query: CaseFeatures,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        return self._search(query, top_k, exclude_self, include_feature_scores=False)

    def _search(
        self,
        query: CaseFeatures,
        top_k: int,
        exclude_self: bool,
        include_feature_scores: bool,
    ) -> list[SearchResult]:
        image_scores = _cosine_scores(query.image_vector, self.image_vectors, self.image_available, self.image_norms)
        timeseries_scores = _cosine_scores(
            query.timeseries_vector,
            self.timeseries_vectors,
            self.timeseries_available,
            self.timeseries_norms,
        )
        if not include_feature_scores:
            image_scores = _missing_scores(self.case_count)
            timeseries_scores = _missing_scores(self.case_count)

        metadata_scores = _metadata_scores(query.metadata, self.metadata_values, self.case_count)
        label_scores = _label_scores(query.label_id, self.label_ids, self.label_available)
        total_scores = _weighted_scores(
            [
                (image_scores, IMAGE_WEIGHT),
                (timeseries_scores, TIMESERIES_WEIGHT),
                (metadata_scores, METADATA_WEIGHT),
                (label_scores, LABEL_WEIGHT),
            ]
        )
        if exclude_self:
            for index, case in enumerate(self.cases):
                if case.sample_id == query.sample_id:
                    total_scores[index] = -np.inf
        return [
            self._result_at(index, total_scores, image_scores, timeseries_scores, metadata_scores, label_scores)
            for index in _top_indices(total_scores, self.sample_tie_ranks, top_k)
        ]

    def _result_at(
        self,
        index: int,
        total_scores: np.ndarray,
        image_scores: np.ndarray,
        timeseries_scores: np.ndarray,
        metadata_scores: np.ndarray,
        label_scores: np.ndarray,
    ) -> SearchResult:
        image_score = _float_or_none(image_scores[index])
        timeseries_score = _float_or_none(timeseries_scores[index])
        metadata_score = _float_or_none(metadata_scores[index])
        label_score = _float_or_none(label_scores[index])
        return SearchResult(
            case=self.cases[index],
            score=float(total_scores[index]),
            image_score=image_score,
            timeseries_score=timeseries_score,
            metadata_score=metadata_score,
            label_score=label_score,
            reason=_reason(image_score, timeseries_score, metadata_score, label_score),
        )


def save_compact_feature_index(path: Path, cases: list[CaseFeatures]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_vectors, image_available = _stack_vectors(cases, "image_vector")
    timeseries_vectors, timeseries_available = _stack_vectors(cases, "timeseries_vector")
    cases_json = json.dumps([_case_without_vectors(case).to_dict() for case in cases], ensure_ascii=False)
    np.savez_compressed(
        path,
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
        cases_json=np.asarray(cases_json),
        image_vectors=image_vectors,
        image_available=image_available,
        image_norms=_matrix_norms(image_vectors),
        timeseries_vectors=timeseries_vectors,
        timeseries_available=timeseries_available,
        timeseries_norms=_matrix_norms(timeseries_vectors),
    )


def load_compact_feature_index(path: Path) -> CompactFeatureIndex:
    with np.load(path, allow_pickle=False) as archive:
        schema_version = str(archive["schema_version"].item())
        if schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported feature schema: {schema_version}")
        cases = [CaseFeatures.from_dict(item) for item in json.loads(str(archive["cases_json"].item()))]
        image_vectors = np.asarray(archive["image_vectors"], dtype=np.float32)
        timeseries_vectors = np.asarray(archive["timeseries_vectors"], dtype=np.float32)
        return CompactFeatureIndex(
            cases=cases,
            image_vectors=image_vectors,
            image_available=np.asarray(archive["image_available"], dtype=bool),
            image_norms=_archive_array_or_norms(archive, "image_norms", image_vectors),
            timeseries_vectors=timeseries_vectors,
            timeseries_available=np.asarray(archive["timeseries_available"], dtype=bool),
            timeseries_norms=_archive_array_or_norms(archive, "timeseries_norms", timeseries_vectors),
            metadata_values=_metadata_values(cases),
            label_ids=_label_ids(cases),
            label_available=_label_available(cases),
            sample_tie_ranks=_sample_tie_ranks(cases),
        )


def is_compact_index_path(path: Path) -> bool:
    return path.suffix.lower() == COMPACT_INDEX_SUFFIX


def _case_without_vectors(case: CaseFeatures) -> CaseFeatures:
    return replace(case, image_vector=None, timeseries_vector=None)


def _stack_vectors(cases: list[CaseFeatures], field_name: str) -> tuple[np.ndarray, np.ndarray]:
    vectors = [getattr(case, field_name) for case in cases]
    dimension = _vector_dimension(vectors, field_name)
    matrix = np.zeros((len(cases), dimension), dtype=np.float32)
    available = np.zeros(len(cases), dtype=bool)
    for index, vector in enumerate(vectors):
        if vector is None:
            continue
        matrix[index] = np.asarray(vector, dtype=np.float32)
        available[index] = True
    return matrix, available


def _vector_dimension(vectors: list[list[float] | None], field_name: str) -> int:
    dimensions = {len(vector) for vector in vectors if vector is not None}
    if len(dimensions) > 1:
        raise ValueError(f"{field_name} has inconsistent dimensions: {sorted(dimensions)}")
    return next(iter(dimensions), 0)


def _row_to_vector(matrix: np.ndarray, available: np.ndarray, index: int) -> list[float] | None:
    if not available[index]:
        return None
    return [float(item) for item in matrix[index]]


def _cosine_scores(
    query_vector: list[float] | None,
    matrix: np.ndarray,
    available: np.ndarray,
    matrix_norms: np.ndarray,
) -> np.ndarray:
    scores = _missing_scores(matrix.shape[0])
    if query_vector is None or matrix.shape[1] == 0 or len(query_vector) != matrix.shape[1]:
        return scores

    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 0.0:
        return scores

    valid = available & (matrix_norms > 0.0)
    if not np.any(valid):
        return scores

    raw_scores = np.matmul(matrix[valid], query) / (matrix_norms[valid] * query_norm)
    scores[valid] = np.clip(raw_scores, 0.0, 1.0)
    return scores


def _metadata_scores(
    query_metadata: dict[str, str],
    metadata_values: dict[str, np.ndarray],
    case_count: int,
) -> np.ndarray:
    scores = np.zeros(case_count, dtype=np.float64)
    total_weights = np.zeros(case_count, dtype=np.float64)
    for field, weight in METADATA_WEIGHTS.items():
        query_value = _normalize_metadata(query_metadata.get(field, ""))
        if query_value == "":
            continue
        candidate_values = metadata_values[field]
        available = candidate_values != ""
        if not np.any(available):
            continue
        total_weights[available] += weight
        scores[available & (candidate_values == query_value)] += weight
    return np.divide(scores, total_weights, out=_missing_scores(case_count), where=total_weights > 0.0)


def _label_scores(
    query_label_id: int | None,
    label_ids: np.ndarray,
    label_available: np.ndarray,
) -> np.ndarray:
    scores = _missing_scores(label_ids.shape[0])
    if query_label_id is None:
        return scores
    scores[label_available] = (label_ids[label_available] == query_label_id).astype(np.float64)
    return scores


def _metadata_values(cases: list[CaseFeatures]) -> dict[str, np.ndarray]:
    return {
        field: np.asarray([_normalize_metadata(case.metadata.get(field, "")) for case in cases], dtype=object)
        for field in METADATA_WEIGHTS
    }


def _label_ids(cases: list[CaseFeatures]) -> np.ndarray:
    return np.asarray([case.label_id if case.label_id is not None else -1 for case in cases], dtype=np.int32)


def _label_available(cases: list[CaseFeatures]) -> np.ndarray:
    return np.asarray([case.label_id is not None for case in cases], dtype=bool)


def _archive_array_or_norms(archive, key: str, matrix: np.ndarray) -> np.ndarray:
    if key in archive.files:
        return np.asarray(archive[key], dtype=np.float32)
    return _matrix_norms(matrix)


def _matrix_norms(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.linalg.norm(matrix, axis=1).astype(np.float32)


def _weighted_scores(weighted_scores: list[tuple[np.ndarray, float]]) -> np.ndarray:
    if not weighted_scores:
        return np.asarray([], dtype=np.float64)
    total = np.zeros(weighted_scores[0][0].shape[0], dtype=np.float64)
    weights = np.zeros(weighted_scores[0][0].shape[0], dtype=np.float64)
    for scores, weight in weighted_scores:
        available = np.isfinite(scores)
        total[available] += scores[available] * weight
        weights[available] += weight
    return np.divide(total, weights, out=np.zeros_like(total), where=weights > 0.0)


def _top_indices(scores: np.ndarray, sample_tie_ranks: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    valid_indices = np.flatnonzero(np.isfinite(scores))
    if valid_indices.size <= top_k:
        return _sort_ranked_indices(valid_indices, scores, sample_tie_ranks).tolist()

    valid_scores = scores[valid_indices]
    threshold_position = valid_scores.size - top_k
    threshold = np.partition(valid_scores, threshold_position)[threshold_position]
    candidate_indices = valid_indices[valid_scores >= threshold]
    return _sort_ranked_indices(candidate_indices, scores, sample_tie_ranks)[:top_k].tolist()


def _sort_ranked_indices(indices: np.ndarray, scores: np.ndarray, sample_tie_ranks: np.ndarray) -> np.ndarray:
    sort_order = np.lexsort((sample_tie_ranks[indices], scores[indices]))
    return indices[sort_order[::-1]]


def _sample_tie_ranks(cases: list[CaseFeatures]) -> np.ndarray:
    sorted_indices = sorted(range(len(cases)), key=lambda index: cases[index].sample_id)
    ranks = np.zeros(len(cases), dtype=np.int32)
    for rank, index in enumerate(sorted_indices):
        ranks[index] = rank
    return ranks


def _optional_score(value: float | None) -> float:
    return MISSING_SCORE if value is None else float(value)


def _float_or_none(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _missing_scores(size: int) -> np.ndarray:
    return np.full(size, MISSING_SCORE, dtype=np.float64)
