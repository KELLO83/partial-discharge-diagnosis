from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prpd_similarity_retrieval import FEATURE_SCHEMA_VERSION
from prpd_similarity_retrieval.compact_index import CompactFeatureIndex
from prpd_similarity_retrieval.models import CaseFeatures


PROTOTYPE_ENCODER_VERSION = "prototype_centroid_encoder_v1"
PROTOTYPE_INDEX_SUFFIX = ".npz"


@dataclass(frozen=True, slots=True)
class PrototypeEncoderConfig:
    image_dim: int = 128
    timeseries_dim: int = 64
    centroid_weight: float = 0.30
    image_weight: float = 0.55
    timeseries_weight: float = 0.45
    random_seed: int = 42


@dataclass(frozen=True, slots=True)
class PrototypeEncoderState:
    config: PrototypeEncoderConfig
    image_projection: np.ndarray
    timeseries_projection: np.ndarray
    image_centroids: np.ndarray
    timeseries_centroids: np.ndarray


@dataclass(frozen=True, slots=True)
class PrototypeSearchResult:
    case: CaseFeatures
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.case.sample_id,
            "label_id": self.case.label_id,
            "label_name": self.case.label_name,
            "similarity": round(self.score, 6),
            "image_path": self.case.image_path,
            "timeseries_path": self.case.timeseries_path,
            "metadata": self.case.metadata,
        }


@dataclass(frozen=True, slots=True)
class PrototypeEmbeddingIndex:
    cases: list[CaseFeatures]
    embeddings: np.ndarray
    embedding_norms: np.ndarray
    label_ids: np.ndarray
    label_available: np.ndarray
    sample_tie_ranks: np.ndarray
    config: PrototypeEncoderConfig

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def find_index(self, sample_id: str) -> int:
        for index, case in enumerate(self.cases):
            if case.sample_id == sample_id:
                return index
        raise ValueError(f"sample_id not found in prototype index: {sample_id}")

    def search_sample(self, sample_id: str, top_k: int = 5, exclude_self: bool = True) -> list[PrototypeSearchResult]:
        query_index = self.find_index(sample_id)
        scores = _cosine_scores(self.embeddings[query_index], self.embeddings, self.embedding_norms)
        if exclude_self:
            scores[query_index] = -np.inf
        return [PrototypeSearchResult(self.cases[index], float(scores[index])) for index in _top_indices(scores, self.sample_tie_ranks, top_k)]


@dataclass(frozen=True, slots=True)
class PrototypeEvaluationMetrics:
    evaluated: int
    top1_label_match_rate: float
    topk_label_match_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "evaluated": self.evaluated,
            "top1_label_match_rate": self.top1_label_match_rate,
            "topk_label_match_rate": self.topk_label_match_rate,
        }


def build_prototype_embedding_index(
    feature_index: CompactFeatureIndex,
    config: PrototypeEncoderConfig,
) -> PrototypeEmbeddingIndex:
    training_indices = np.arange(feature_index.case_count, dtype=np.int32)
    state = fit_prototype_encoder_state(feature_index, training_indices, config)
    embeddings = transform_prototype_embeddings(feature_index, state)
    return PrototypeEmbeddingIndex(
        cases=feature_index.cases,
        embeddings=embeddings,
        embedding_norms=_row_norms(embeddings),
        label_ids=feature_index.label_ids,
        label_available=feature_index.label_available,
        sample_tie_ranks=feature_index.sample_tie_ranks,
        config=config,
    )


def fit_prototype_encoder_state(
    feature_index: CompactFeatureIndex,
    training_indices: np.ndarray,
    config: PrototypeEncoderConfig,
) -> PrototypeEncoderState:
    normalized_training_indices = _normalize_indices(training_indices, feature_index.case_count)
    image_projection = _projection_matrix(feature_index.image_vectors.shape[1], config.image_dim, config.random_seed)
    timeseries_projection = _projection_matrix(
        feature_index.timeseries_vectors.shape[1],
        config.timeseries_dim,
        config.random_seed + 17,
    )
    image_projected = _project_modality(feature_index.image_vectors, feature_index.image_available, image_projection)
    timeseries_projected = _project_modality(
        feature_index.timeseries_vectors,
        feature_index.timeseries_available,
        timeseries_projection,
    )
    return PrototypeEncoderState(
        config=config,
        image_projection=image_projection,
        timeseries_projection=timeseries_projection,
        image_centroids=_fit_centroids(
            image_projected,
            feature_index.image_available,
            feature_index.label_ids,
            feature_index.label_available,
            normalized_training_indices,
        ),
        timeseries_centroids=_fit_centroids(
            timeseries_projected,
            feature_index.timeseries_available,
            feature_index.label_ids,
            feature_index.label_available,
            normalized_training_indices,
        ),
    )


def transform_prototype_embeddings(
    feature_index: CompactFeatureIndex,
    state: PrototypeEncoderState,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    row_indices = _all_indices(feature_index.case_count) if indices is None else _normalize_indices(indices, feature_index.case_count)
    image_embeddings = _transform_modality_embeddings(
        feature_index.image_vectors[row_indices],
        feature_index.image_available[row_indices],
        state.image_projection,
        state.image_centroids,
        state.config.centroid_weight,
    )
    timeseries_embeddings = _transform_modality_embeddings(
        feature_index.timeseries_vectors[row_indices],
        feature_index.timeseries_available[row_indices],
        state.timeseries_projection,
        state.timeseries_centroids,
        state.config.centroid_weight,
    )
    return _normalize_rows(
        np.concatenate(
            [
                image_embeddings * state.config.image_weight,
                timeseries_embeddings * state.config.timeseries_weight,
            ],
            axis=1,
        )
    )


def save_prototype_embedding_index(path: Path, index: PrototypeEmbeddingIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases_json = json.dumps([case.to_dict() for case in index.cases], ensure_ascii=False)
    np.savez_compressed(
        path,
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
        encoder_version=np.asarray(PROTOTYPE_ENCODER_VERSION),
        config_json=np.asarray(json.dumps(_config_to_dict(index.config), ensure_ascii=False)),
        cases_json=np.asarray(cases_json),
        embeddings=index.embeddings.astype(np.float32),
        embedding_norms=index.embedding_norms.astype(np.float32),
        label_ids=index.label_ids.astype(np.int32),
        label_available=index.label_available.astype(bool),
        sample_tie_ranks=index.sample_tie_ranks.astype(np.int32),
    )


def load_prototype_embedding_index(path: Path) -> PrototypeEmbeddingIndex:
    with np.load(path, allow_pickle=False) as archive:
        schema_version = str(archive["schema_version"].item())
        if schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported feature schema: {schema_version}")
        encoder_version = str(archive["encoder_version"].item())
        if encoder_version != PROTOTYPE_ENCODER_VERSION:
            raise ValueError(f"Unsupported prototype encoder: {encoder_version}")
        cases = [CaseFeatures.from_dict(item) for item in json.loads(str(archive["cases_json"].item()))]
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        return PrototypeEmbeddingIndex(
            cases=cases,
            embeddings=embeddings,
            embedding_norms=np.asarray(archive["embedding_norms"], dtype=np.float32),
            label_ids=np.asarray(archive["label_ids"], dtype=np.int32),
            label_available=np.asarray(archive["label_available"], dtype=bool),
            sample_tie_ranks=np.asarray(archive["sample_tie_ranks"], dtype=np.int32),
            config=PrototypeEncoderConfig(**json.loads(str(archive["config_json"].item()))),
        )


def evaluate_prototype_index(
    index: PrototypeEmbeddingIndex,
    limit: int | None,
    top_k: int,
    batch_size: int = 256,
) -> PrototypeEvaluationMetrics:
    if top_k <= 0:
        return PrototypeEvaluationMetrics(evaluated=0, top1_label_match_rate=0.0, topk_label_match_rate=0.0)
    count = index.case_count if limit is None else min(limit, index.case_count)
    query_indices = np.arange(count, dtype=np.int32)
    query_indices = query_indices[index.label_available[:count]]
    if query_indices.size == 0:
        return PrototypeEvaluationMetrics(evaluated=0, top1_label_match_rate=0.0, topk_label_match_rate=0.0)

    evaluated = 0
    top1_matches = 0
    topk_matches = 0
    for batch_indices in _batches(query_indices, batch_size):
        score_matrix = _cosine_score_matrix(index.embeddings, index.embedding_norms, batch_indices)
        score_matrix[np.arange(batch_indices.size), batch_indices] = -np.inf
        top_indices_matrix = _top_indices_matrix(score_matrix, index.sample_tie_ranks, top_k)
        has_results = top_indices_matrix[:, 0] >= 0
        if not np.any(has_results):
            continue
        query_labels = index.label_ids[batch_indices]
        top_labels = _labels_for_top_indices(index, top_indices_matrix)
        evaluated += int(np.sum(has_results))
        top1_matches += int(np.sum(has_results & (top_labels[:, 0] == query_labels)))
        topk_matches += int(np.sum(has_results & np.any(top_labels == query_labels[:, None], axis=1)))
    return PrototypeEvaluationMetrics(
        evaluated=evaluated,
        top1_label_match_rate=_safe_rate(top1_matches, evaluated),
        topk_label_match_rate=_safe_rate(topk_matches, evaluated),
    )


def prototype_results_to_json(results: list[PrototypeSearchResult]) -> str:
    return json.dumps({"results": [result.to_dict() for result in results]}, ensure_ascii=False, indent=2)


def _transform_modality_embeddings(
    vectors: np.ndarray,
    available: np.ndarray,
    projection: np.ndarray,
    centroids: np.ndarray,
    centroid_weight: float,
) -> np.ndarray:
    if vectors.shape[1] == 0:
        return np.zeros((vectors.shape[0], 0), dtype=np.float32)
    projected = _project_modality(vectors, available, projection)
    centroid_affinity = _centroid_affinity_from_centroids(projected, centroids)
    return _normalize_rows(
        np.concatenate(
            [
                projected * (1.0 - centroid_weight),
                centroid_affinity * centroid_weight,
            ],
            axis=1,
        )
    )


def _project_modality(vectors: np.ndarray, available: np.ndarray, projection: np.ndarray) -> np.ndarray:
    if vectors.shape[1] == 0 or projection.shape[1] == 0:
        return np.zeros((vectors.shape[0], projection.shape[1]), dtype=np.float32)
    projected = _normalize_rows(vectors @ projection)
    projected[~available] = 0.0
    return projected


def _projection_matrix(input_dim: int, output_dim: int, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    return (rng.standard_normal((input_dim, output_dim)).astype(np.float32) / max(np.sqrt(input_dim), 1.0)).astype(np.float32)


def _fit_centroids(
    vectors: np.ndarray,
    available: np.ndarray,
    label_ids: np.ndarray,
    label_available: np.ndarray,
    training_indices: np.ndarray,
) -> np.ndarray:
    training_mask = np.zeros(label_ids.shape[0], dtype=bool)
    training_mask[training_indices] = True
    labels = sorted(int(label_id) for label_id in np.unique(label_ids[training_mask & available & label_available]) if label_id >= 0)
    if not labels:
        return np.zeros((0, vectors.shape[1]), dtype=np.float32)
    centroids = []
    for label_id in labels:
        mask = training_mask & available & label_available & (label_ids == label_id)
        if not np.any(mask):
            centroids.append(np.zeros(vectors.shape[1], dtype=np.float32))
            continue
        centroids.append(np.mean(vectors[mask], axis=0).astype(np.float32))
    return _normalize_rows(np.vstack(centroids).astype(np.float32))


def _centroid_affinity_from_centroids(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if centroids.shape[0] == 0:
        return np.zeros((vectors.shape[0], 0), dtype=np.float32)
    if vectors.shape[1] == 0:
        return np.zeros((vectors.shape[0], centroids.shape[0]), dtype=np.float32)
    return np.clip(vectors @ centroids.T, 0.0, 1.0).astype(np.float32)


def _cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray, embedding_norms: np.ndarray) -> np.ndarray:
    scores = np.full(embeddings.shape[0], np.nan, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_embedding))
    valid = embedding_norms > 0.0
    if query_norm <= 0.0 or not np.any(valid):
        return scores
    raw_scores = embeddings[valid] @ query_embedding / (embedding_norms[valid] * query_norm)
    scores[valid] = np.clip(raw_scores, 0.0, 1.0)
    return scores


def _cosine_score_matrix(embeddings: np.ndarray, embedding_norms: np.ndarray, query_indices: np.ndarray) -> np.ndarray:
    query_embeddings = embeddings[query_indices]
    query_norms = embedding_norms[query_indices]
    raw_scores = query_embeddings @ embeddings.T
    denominators = query_norms[:, None] * embedding_norms[None, :]
    valid = (query_norms[:, None] > 0.0) & (embedding_norms[None, :] > 0.0)
    scores = np.divide(raw_scores, denominators, out=np.full(raw_scores.shape, np.nan, dtype=np.float32), where=valid)
    return np.clip(scores, 0.0, 1.0, out=scores)


def _batches(indices: np.ndarray, batch_size: int):
    normalized_batch_size = max(1, batch_size)
    for start in range(0, indices.size, normalized_batch_size):
        yield indices[start : start + normalized_batch_size]


def _top_indices_matrix(scores: np.ndarray, sample_tie_ranks: np.ndarray, top_k: int) -> np.ndarray:
    rows = np.full((scores.shape[0], top_k), -1, dtype=np.int32)
    for row_index, row_scores in enumerate(scores):
        top_indices = _top_indices(row_scores, sample_tie_ranks, top_k)
        rows[row_index, : len(top_indices)] = top_indices
    return rows


def _labels_for_top_indices(index: PrototypeEmbeddingIndex, top_indices: np.ndarray) -> np.ndarray:
    labels = np.full(top_indices.shape, -1, dtype=np.int32)
    valid = top_indices >= 0
    labels[valid] = index.label_ids[top_indices[valid]]
    return labels


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


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return matrix.astype(np.float32)
    norms = _row_norms(matrix)
    return np.divide(matrix, norms[:, None], out=np.zeros_like(matrix, dtype=np.float32), where=norms[:, None] > 0.0).astype(np.float32)


def _row_norms(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.linalg.norm(matrix, axis=1).astype(np.float32)


def _all_indices(count: int) -> np.ndarray:
    return np.arange(count, dtype=np.int32)


def _normalize_indices(indices: np.ndarray, count: int) -> np.ndarray:
    normalized = np.asarray(indices, dtype=np.int32)
    if normalized.ndim != 1:
        raise ValueError("indices must be a one-dimensional array")
    if normalized.size == 0:
        return normalized
    if int(np.min(normalized)) < 0 or int(np.max(normalized)) >= count:
        raise ValueError("indices contain out-of-range values")
    return normalized


def _config_to_dict(config: PrototypeEncoderConfig) -> dict[str, int | float]:
    return {
        "image_dim": config.image_dim,
        "timeseries_dim": config.timeseries_dim,
        "centroid_weight": config.centroid_weight,
        "image_weight": config.image_weight,
        "timeseries_weight": config.timeseries_weight,
        "random_seed": config.random_seed,
    }


def _safe_rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 6)
