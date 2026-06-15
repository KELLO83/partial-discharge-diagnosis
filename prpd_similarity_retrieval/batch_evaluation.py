from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prpd_similarity_retrieval.compact_index import CompactFeatureIndex
from prpd_similarity_retrieval.retrieval import (
    IMAGE_WEIGHT,
    LABEL_BASELINE_WEIGHT,
    LABEL_WEIGHT,
    METADATA_BASELINE_WEIGHT,
    METADATA_WEIGHTS,
    METADATA_WEIGHT,
    TIMESERIES_WEIGHT,
)


ProgressWriter = Callable[[dict[str, int | float | str]], None]
BREAKDOWN_FIELDS = (
    "label_id",
    "label_name",
    "equipment_name",
    "sensor_type",
    "insulator_type",
    "clearance_distance",
    "equipment_rated_voltage",
)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    evaluated: int
    top1_label_match_rate: float
    topk_label_match_rate: float
    breakdowns: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evaluated": self.evaluated,
            "top1_label_match_rate": self.top1_label_match_rate,
            "topk_label_match_rate": self.topk_label_match_rate,
        }
        if self.breakdowns:
            payload["breakdowns"] = self.breakdowns
        return payload


def evaluate_compact_index(
    index: CompactFeatureIndex,
    limit: int | None,
    top_k: int,
    use_query_label: bool,
    metadata_baseline: bool,
    batch_size: int = 64,
    progress_every: int = 0,
    progress_writer: ProgressWriter | None = None,
    breakdown_fields: tuple[str, ...] = (),
) -> EvaluationMetrics:
    if top_k <= 0:
        return EvaluationMetrics(evaluated=0, top1_label_match_rate=0.0, topk_label_match_rate=0.0)

    query_indices = _query_indices(index, limit)
    if query_indices.size == 0:
        return EvaluationMetrics(evaluated=0, top1_label_match_rate=0.0, topk_label_match_rate=0.0)

    started_at = time.perf_counter()
    top1_matches = 0
    topk_matches = 0
    evaluated = 0
    completed = 0
    breakdowns = _new_breakdowns(breakdown_fields)
    for batch_indices in _batches(query_indices, batch_size):
        top_indices = _top_indices_for_batch(
            index=index,
            query_indices=batch_indices,
            top_k=top_k,
            use_query_label=use_query_label,
            metadata_baseline=metadata_baseline,
        )
        has_results = top_indices[:, 0] >= 0
        if not np.any(has_results):
            completed += int(batch_indices.size)
            _maybe_write_progress(completed, int(query_indices.size), started_at, progress_every, progress_writer)
            continue
        batch_labels = index.label_ids[batch_indices]
        top_labels = _labels_for_top_indices(index, top_indices)
        top1_match_mask = has_results & (top_labels[:, 0] == batch_labels)
        topk_match_mask = has_results & np.any(top_labels == batch_labels[:, None], axis=1)
        top1_matches += int(np.sum(top1_match_mask))
        topk_matches += int(np.sum(topk_match_mask))
        evaluated += int(np.sum(has_results))
        _update_breakdowns(breakdowns, index, batch_indices, has_results, top1_match_mask, topk_match_mask)
        completed += int(batch_indices.size)
        _maybe_write_progress(completed, int(query_indices.size), started_at, progress_every, progress_writer)

    return EvaluationMetrics(
        evaluated=evaluated,
        top1_label_match_rate=_safe_rate(top1_matches, evaluated),
        topk_label_match_rate=_safe_rate(topk_matches, evaluated),
        breakdowns=_finalize_breakdowns(breakdowns),
    )


def default_progress_writer(payload: dict[str, int | float | str]) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def _top_indices_for_batch(
    index: CompactFeatureIndex,
    query_indices: np.ndarray,
    top_k: int,
    use_query_label: bool,
    metadata_baseline: bool,
) -> np.ndarray:
    metadata_scores = _metadata_score_matrix(index, query_indices)
    label_scores = _label_score_matrix(index, query_indices) if use_query_label else _missing_matrix(query_indices.size, index.case_count)
    if metadata_baseline:
        total_scores = _weighted_score_matrix([(metadata_scores, METADATA_BASELINE_WEIGHT), (label_scores, LABEL_BASELINE_WEIGHT)])
    else:
        total_scores = _weighted_score_matrix(
            [
                (_cosine_score_matrix(index.image_vectors, index.image_norms, index.image_available, query_indices), IMAGE_WEIGHT),
                (
                    _cosine_score_matrix(
                        index.timeseries_vectors,
                        index.timeseries_norms,
                        index.timeseries_available,
                        query_indices,
                    ),
                    TIMESERIES_WEIGHT,
                ),
                (metadata_scores, METADATA_WEIGHT),
                (label_scores, LABEL_WEIGHT),
            ]
        )
    total_scores[np.arange(query_indices.size), query_indices] = -np.inf
    return _top_indices_matrix(total_scores, index.sample_tie_ranks, top_k)


def _query_indices(index: CompactFeatureIndex, limit: int | None) -> np.ndarray:
    count = index.case_count if limit is None else min(limit, index.case_count)
    indices = np.arange(count, dtype=np.int32)
    return indices[index.label_available[:count]]


def _new_breakdowns(fields: tuple[str, ...]) -> dict[str, dict[str, dict[str, int]]]:
    return {field: {} for field in fields}


def _update_breakdowns(
    breakdowns: dict[str, dict[str, dict[str, int]]],
    index: CompactFeatureIndex,
    query_indices: np.ndarray,
    has_results: np.ndarray,
    top1_match_mask: np.ndarray,
    topk_match_mask: np.ndarray,
) -> None:
    if not breakdowns:
        return
    for position, query_index in enumerate(query_indices):
        if not has_results[position]:
            continue
        for field, field_groups in breakdowns.items():
            group_value = _group_value(index, int(query_index), field)
            stats = field_groups.setdefault(group_value, {"evaluated": 0, "top1_matches": 0, "topk_matches": 0})
            stats["evaluated"] += 1
            stats["top1_matches"] += int(top1_match_mask[position])
            stats["topk_matches"] += int(topk_match_mask[position])


def _finalize_breakdowns(breakdowns: dict[str, dict[str, dict[str, int]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        field: [
            {
                "value": value,
                "evaluated": stats["evaluated"],
                "top1_label_match_rate": _safe_rate(stats["top1_matches"], stats["evaluated"]),
                "topk_label_match_rate": _safe_rate(stats["topk_matches"], stats["evaluated"]),
            }
            for value, stats in sorted(field_groups.items(), key=lambda item: (-item[1]["evaluated"], item[0]))
        ]
        for field, field_groups in breakdowns.items()
    }


def _group_value(index: CompactFeatureIndex, query_index: int, field: str) -> str:
    case = index.cases[query_index]
    if field == "label_id":
        return str(case.label_id) if case.label_id is not None else "unknown"
    if field == "label_name":
        return case.label_name or "unknown"
    value = case.metadata.get(field, "")
    return value if value != "" else "unknown"


def _batches(indices: np.ndarray, batch_size: int):
    normalized_batch_size = max(1, batch_size)
    for start in range(0, indices.size, normalized_batch_size):
        yield indices[start : start + normalized_batch_size]


def _cosine_score_matrix(
    vectors: np.ndarray,
    norms: np.ndarray,
    available: np.ndarray,
    query_indices: np.ndarray,
) -> np.ndarray:
    query_vectors = vectors[query_indices]
    query_norms = norms[query_indices]
    raw_scores = np.matmul(query_vectors, vectors.T).astype(np.float32, copy=False)
    denominators = query_norms[:, None] * norms[None, :]
    valid = (query_norms[:, None] > 0.0) & (norms[None, :] > 0.0) & available[query_indices, None] & available[None, :]
    scores = np.divide(raw_scores, denominators, out=_missing_matrix(query_indices.size, vectors.shape[0]), where=valid)
    return np.clip(scores, 0.0, 1.0, out=scores)


def _metadata_score_matrix(index: CompactFeatureIndex, query_indices: np.ndarray) -> np.ndarray:
    scores = np.zeros((query_indices.size, index.case_count), dtype=np.float32)
    total_weights = np.zeros((query_indices.size, index.case_count), dtype=np.float32)
    for field, weight in METADATA_WEIGHTS.items():
        values = index.metadata_values[field]
        query_values = values[query_indices]
        candidate_available = values != ""
        query_available = query_values != ""
        available = query_available[:, None] & candidate_available[None, :]
        if not np.any(available):
            continue
        total_weights[available] += weight
        scores[available & (query_values[:, None] == values[None, :])] += weight
    return np.divide(scores, total_weights, out=_missing_matrix(query_indices.size, index.case_count), where=total_weights > 0.0)


def _label_score_matrix(index: CompactFeatureIndex, query_indices: np.ndarray) -> np.ndarray:
    scores = _missing_matrix(query_indices.size, index.case_count)
    available = index.label_available[None, :]
    scores[available.repeat(query_indices.size, axis=0)] = 0.0
    matches = index.label_ids[query_indices, None] == index.label_ids[None, :]
    scores[available & matches] = 1.0
    return scores


def _weighted_score_matrix(weighted_scores: list[tuple[np.ndarray, float]]) -> np.ndarray:
    total = np.zeros(weighted_scores[0][0].shape, dtype=np.float32)
    weights = np.zeros(weighted_scores[0][0].shape, dtype=np.float32)
    for scores, weight in weighted_scores:
        available = np.isfinite(scores)
        total[available] += scores[available] * weight
        weights[available] += weight
    return np.divide(total, weights, out=np.zeros_like(total), where=weights > 0.0)


def _top_indices_matrix(scores: np.ndarray, sample_tie_ranks: np.ndarray, top_k: int) -> np.ndarray:
    rows = np.full((scores.shape[0], top_k), -1, dtype=np.int32)
    for row_index, row_scores in enumerate(scores):
        top_indices = _top_indices_for_row(row_scores, sample_tie_ranks, top_k)
        rows[row_index, : top_indices.size] = top_indices
    return rows


def _top_indices_for_row(scores: np.ndarray, sample_tie_ranks: np.ndarray, top_k: int) -> np.ndarray:
    valid_indices = np.flatnonzero(np.isfinite(scores))
    if valid_indices.size <= top_k:
        return _sort_ranked_indices(valid_indices, scores, sample_tie_ranks)
    valid_scores = scores[valid_indices]
    threshold_position = valid_scores.size - top_k
    threshold = np.partition(valid_scores, threshold_position)[threshold_position]
    candidate_indices = valid_indices[valid_scores >= threshold]
    return _sort_ranked_indices(candidate_indices, scores, sample_tie_ranks)[:top_k]


def _sort_ranked_indices(indices: np.ndarray, scores: np.ndarray, sample_tie_ranks: np.ndarray) -> np.ndarray:
    sort_order = np.lexsort((sample_tie_ranks[indices], scores[indices]))
    return indices[sort_order[::-1]]


def _labels_for_top_indices(index: CompactFeatureIndex, top_indices: np.ndarray) -> np.ndarray:
    labels = np.full(top_indices.shape, -1, dtype=np.int32)
    valid = top_indices >= 0
    labels[valid] = index.label_ids[top_indices[valid]]
    return labels


def _missing_matrix(row_count: int, column_count: int) -> np.ndarray:
    return np.full((row_count, column_count), np.nan, dtype=np.float32)


def _maybe_write_progress(
    completed: int,
    total: int,
    started_at: float,
    progress_every: int,
    progress_writer: ProgressWriter | None,
) -> None:
    if progress_every <= 0 or progress_writer is None:
        return
    if completed % progress_every != 0 and completed != total:
        return
    elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
    queries_per_second = completed / elapsed_seconds
    remaining_seconds = (total - completed) / queries_per_second if queries_per_second > 0 else 0.0
    progress_writer(
        {
            "event": "evaluate_progress",
            "done": completed,
            "total": total,
            "queries_per_second": round(queries_per_second, 2),
            "eta_seconds": round(remaining_seconds, 1),
        }
    )


def _safe_rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 6)
