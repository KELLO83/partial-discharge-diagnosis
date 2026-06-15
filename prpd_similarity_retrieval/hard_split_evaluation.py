from __future__ import annotations

import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from prpd_similarity_retrieval.batch_evaluation import BREAKDOWN_FIELDS
from prpd_similarity_retrieval.compact_index import CompactFeatureIndex
from prpd_similarity_retrieval.learned_encoder import (
    LEARNED_ENCODER_VERSION,
    LearnedEncoderConfig,
    fit_learned_encoder_state,
    transform_learned_embeddings,
)
from prpd_similarity_retrieval.prototype_encoder import (
    PROTOTYPE_ENCODER_VERSION,
    PrototypeEncoderConfig,
    fit_prototype_encoder_state,
    transform_prototype_embeddings,
)
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
DEFAULT_SPLIT_FIELD = "equipment_name"
SPLIT_FIELDS = BREAKDOWN_FIELDS


@dataclass(frozen=True, slots=True)
class HardSplitSelection:
    split_field: str
    holdout_values: tuple[str, ...]
    train_indices: np.ndarray
    query_indices: np.ndarray

    @property
    def train_count(self) -> int:
        return int(self.train_indices.size)

    @property
    def query_count(self) -> int:
        return int(self.query_indices.size)


@dataclass(frozen=True, slots=True)
class HardSplitMetrics:
    retrieval_mode: str
    split_field: str
    holdout_values: tuple[str, ...]
    train_count: int
    query_count: int
    evaluated: int
    top1_label_match_rate: float
    topk_label_match_rate: float
    encoder_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "retrieval_mode": self.retrieval_mode,
            "split_field": self.split_field,
            "holdout_values": list(self.holdout_values),
            "train_count": self.train_count,
            "query_count": self.query_count,
            "evaluated": self.evaluated,
            "top1_label_match_rate": self.top1_label_match_rate,
            "topk_label_match_rate": self.topk_label_match_rate,
        }
        if self.encoder_version is not None:
            payload["encoder_version"] = self.encoder_version
        return payload


@dataclass(frozen=True, slots=True)
class HardSplitComparison:
    holdout_value: str
    feature_retrieval: HardSplitMetrics
    metadata_baseline: HardSplitMetrics
    prototype_encoder: HardSplitMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "holdout_value": self.holdout_value,
            "feature_retrieval": self.feature_retrieval.to_dict(),
            "metadata_baseline": self.metadata_baseline.to_dict(),
            "delta": {
                "top1_label_match_rate": round(
                    self.feature_retrieval.top1_label_match_rate - self.metadata_baseline.top1_label_match_rate,
                    6,
                ),
                "topk_label_match_rate": round(
                    self.feature_retrieval.topk_label_match_rate - self.metadata_baseline.topk_label_match_rate,
                    6,
                ),
            },
        }
        if self.prototype_encoder is not None:
            payload["prototype_encoder"] = self.prototype_encoder.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class HardSplitReport:
    split_field: str
    top_k: int
    use_query_label: bool
    min_query_count: int
    limit_per_holdout: int | None
    comparisons: list[HardSplitComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_field": self.split_field,
            "top_k": self.top_k,
            "use_query_label": self.use_query_label,
            "min_query_count": self.min_query_count,
            "limit_per_holdout": self.limit_per_holdout,
            "holdout_count": len(self.comparisons),
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }


@dataclass(frozen=True, slots=True)
class HardSplitModeReport:
    retrieval_mode: str
    split_field: str
    top_k: int
    min_query_count: int
    limit_per_holdout: int | None
    metrics: list[HardSplitMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "split_field": self.split_field,
            "top_k": self.top_k,
            "min_query_count": self.min_query_count,
            "limit_per_holdout": self.limit_per_holdout,
            "holdout_count": len(self.metrics),
            "metrics": [metrics.to_dict() for metrics in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class HardSplitNeighbor:
    rank: int
    sample_id: str
    label_id: int | None
    label_name: str
    score: float
    image_path: str | None
    timeseries_path: str | None
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "sample_id": self.sample_id,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "score": round(self.score, 6),
            "image_path": self.image_path,
            "timeseries_path": self.timeseries_path,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class HardSplitFailureCase:
    query_sample_id: str
    query_label_id: int | None
    query_label_name: str
    query_image_path: str | None
    query_timeseries_path: str | None
    query_metadata: dict[str, str]
    neighbors: list[HardSplitNeighbor]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": {
                "sample_id": self.query_sample_id,
                "label_id": self.query_label_id,
                "label_name": self.query_label_name,
                "image_path": self.query_image_path,
                "timeseries_path": self.query_timeseries_path,
                "metadata": self.query_metadata,
            },
            "neighbors": [neighbor.to_dict() for neighbor in self.neighbors],
        }


@dataclass(frozen=True, slots=True)
class HardSplitFailureSample:
    retrieval_mode: str
    split_field: str
    holdout_values: tuple[str, ...]
    train_count: int
    query_count: int
    inspected: int
    top_k: int
    failures: list[HardSplitFailureCase]

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "split_field": self.split_field,
            "holdout_values": list(self.holdout_values),
            "train_count": self.train_count,
            "query_count": self.query_count,
            "inspected": self.inspected,
            "top_k": self.top_k,
            "failure_count": len(self.failures),
            "failures": [failure.to_dict() for failure in self.failures],
        }


def evaluate_feature_hard_split(
    index: CompactFeatureIndex,
    split_field: str = DEFAULT_SPLIT_FIELD,
    holdout_values: tuple[str, ...] = (),
    limit: int | None = None,
    top_k: int = 3,
    use_query_label: bool = False,
    metadata_baseline: bool = False,
    batch_size: int = 256,
    progress_every: int = 0,
    progress_writer: ProgressWriter | None = None,
) -> HardSplitMetrics:
    selection = select_hard_split(index, split_field, holdout_values, limit)
    if top_k <= 0 or selection.train_count == 0 or selection.query_count == 0:
        return _empty_metrics("metadata_baseline" if metadata_baseline else "feature_retrieval", selection)

    started_at = time.perf_counter()
    evaluated = 0
    top1_matches = 0
    topk_matches = 0
    completed = 0
    for batch_indices in _batches(selection.query_indices, batch_size):
        score_matrix = _feature_score_matrix_for_batch(
            index=index,
            query_indices=batch_indices,
            candidate_indices=selection.train_indices,
            use_query_label=use_query_label,
            metadata_baseline=metadata_baseline,
        )
        top_positions = _top_positions_matrix(score_matrix, index.sample_tie_ranks[selection.train_indices], top_k)
        has_results, top1_mask, topk_mask = _label_match_masks(index, batch_indices, selection.train_indices, top_positions)
        evaluated += int(np.sum(has_results))
        top1_matches += int(np.sum(top1_mask))
        topk_matches += int(np.sum(topk_mask))
        completed += int(batch_indices.size)
        _maybe_write_progress(
            mode="metadata_baseline" if metadata_baseline else "feature_retrieval",
            completed=completed,
            total=selection.query_count,
            started_at=started_at,
            progress_every=progress_every,
            progress_writer=progress_writer,
        )

    return HardSplitMetrics(
        retrieval_mode="metadata_baseline" if metadata_baseline else "feature_retrieval",
        split_field=selection.split_field,
        holdout_values=selection.holdout_values,
        train_count=selection.train_count,
        query_count=selection.query_count,
        evaluated=evaluated,
        top1_label_match_rate=_safe_rate(top1_matches, evaluated),
        topk_label_match_rate=_safe_rate(topk_matches, evaluated),
    )


def sample_feature_hard_split_failures(
    index: CompactFeatureIndex,
    split_field: str = DEFAULT_SPLIT_FIELD,
    holdout_values: tuple[str, ...] = (),
    limit: int | None = None,
    top_k: int = 3,
    max_failures: int = 10,
    use_query_label: bool = False,
    metadata_baseline: bool = False,
    batch_size: int = 128,
) -> HardSplitFailureSample:
    selection = select_hard_split(index, split_field, holdout_values, limit)
    retrieval_mode = "metadata_baseline" if metadata_baseline else "feature_retrieval"
    if top_k <= 0 or max_failures <= 0 or selection.train_count == 0 or selection.query_count == 0:
        return _empty_failure_sample(retrieval_mode, selection, top_k)

    failures: list[HardSplitFailureCase] = []
    inspected = 0
    for batch_indices in _batches(selection.query_indices, batch_size):
        score_matrix = _feature_score_matrix_for_batch(
            index=index,
            query_indices=batch_indices,
            candidate_indices=selection.train_indices,
            use_query_label=use_query_label,
            metadata_baseline=metadata_baseline,
        )
        top_positions = _top_positions_matrix(score_matrix, index.sample_tie_ranks[selection.train_indices], top_k)
        has_results, _, topk_match_mask = _label_match_masks(index, batch_indices, selection.train_indices, top_positions)
        for row_index, query_index in enumerate(batch_indices):
            inspected += 1
            if has_results[row_index] and not topk_match_mask[row_index]:
                failures.append(
                    _failure_case(
                        index=index,
                        query_index=int(query_index),
                        candidate_indices=selection.train_indices,
                        top_positions=top_positions[row_index],
                        scores=score_matrix[row_index],
                    )
                )
                if len(failures) >= max_failures:
                    return _failure_sample(retrieval_mode, selection, inspected, top_k, failures)
    return _failure_sample(retrieval_mode, selection, inspected, top_k, failures)


def evaluate_prototype_hard_split(
    index: CompactFeatureIndex,
    config: PrototypeEncoderConfig,
    split_field: str = DEFAULT_SPLIT_FIELD,
    holdout_values: tuple[str, ...] = (),
    limit: int | None = None,
    top_k: int = 3,
    batch_size: int = 256,
    progress_every: int = 0,
    progress_writer: ProgressWriter | None = None,
) -> HardSplitMetrics:
    selection = select_hard_split(index, split_field, holdout_values, limit)
    if top_k <= 0 or selection.train_count == 0 or selection.query_count == 0:
        return _empty_metrics("prototype_encoder", selection, encoder_version=PROTOTYPE_ENCODER_VERSION)

    state = fit_prototype_encoder_state(index, selection.train_indices, config)
    candidate_embeddings = transform_prototype_embeddings(index, state, selection.train_indices)
    candidate_norms = _row_norms(candidate_embeddings)
    started_at = time.perf_counter()
    evaluated = 0
    top1_matches = 0
    topk_matches = 0
    completed = 0
    for batch_indices in _batches(selection.query_indices, batch_size):
        query_embeddings = transform_prototype_embeddings(index, state, batch_indices)
        score_matrix = _embedding_score_matrix(query_embeddings, candidate_embeddings, candidate_norms)
        top_positions = _top_positions_matrix(score_matrix, index.sample_tie_ranks[selection.train_indices], top_k)
        has_results, top1_mask, topk_mask = _label_match_masks(index, batch_indices, selection.train_indices, top_positions)
        evaluated += int(np.sum(has_results))
        top1_matches += int(np.sum(top1_mask))
        topk_matches += int(np.sum(topk_mask))
        completed += int(batch_indices.size)
        _maybe_write_progress(
            mode="prototype_encoder",
            completed=completed,
            total=selection.query_count,
            started_at=started_at,
            progress_every=progress_every,
            progress_writer=progress_writer,
        )

    return HardSplitMetrics(
        retrieval_mode="prototype_encoder",
        split_field=selection.split_field,
        holdout_values=selection.holdout_values,
        train_count=selection.train_count,
        query_count=selection.query_count,
        evaluated=evaluated,
        top1_label_match_rate=_safe_rate(top1_matches, evaluated),
        topk_label_match_rate=_safe_rate(topk_matches, evaluated),
        encoder_version=PROTOTYPE_ENCODER_VERSION,
    )


def evaluate_learned_hard_split(
    index: CompactFeatureIndex,
    config: LearnedEncoderConfig,
    split_field: str = DEFAULT_SPLIT_FIELD,
    holdout_values: tuple[str, ...] = (),
    limit: int | None = None,
    top_k: int = 3,
    batch_size: int = 256,
    progress_every: int = 0,
    progress_writer: ProgressWriter | None = None,
) -> HardSplitMetrics:
    selection = select_hard_split(index, split_field, holdout_values, limit)
    if top_k <= 0 or selection.train_count == 0 or selection.query_count == 0:
        return _empty_metrics("learned_projection_encoder", selection, encoder_version=LEARNED_ENCODER_VERSION)

    state = fit_learned_encoder_state(index, selection.train_indices, config)
    candidate_embeddings = transform_learned_embeddings(index, state, selection.train_indices)
    candidate_norms = _row_norms(candidate_embeddings)
    started_at = time.perf_counter()
    evaluated = 0
    top1_matches = 0
    topk_matches = 0
    completed = 0
    for batch_indices in _batches(selection.query_indices, batch_size):
        query_embeddings = transform_learned_embeddings(index, state, batch_indices)
        score_matrix = _embedding_score_matrix(query_embeddings, candidate_embeddings, candidate_norms)
        top_positions = _top_positions_matrix(score_matrix, index.sample_tie_ranks[selection.train_indices], top_k)
        has_results, top1_mask, topk_mask = _label_match_masks(index, batch_indices, selection.train_indices, top_positions)
        evaluated += int(np.sum(has_results))
        top1_matches += int(np.sum(top1_mask))
        topk_matches += int(np.sum(topk_mask))
        completed += int(batch_indices.size)
        _maybe_write_progress(
            mode="learned_projection_encoder",
            completed=completed,
            total=selection.query_count,
            started_at=started_at,
            progress_every=progress_every,
            progress_writer=progress_writer,
        )

    return HardSplitMetrics(
        retrieval_mode="learned_projection_encoder",
        split_field=selection.split_field,
        holdout_values=selection.holdout_values,
        train_count=selection.train_count,
        query_count=selection.query_count,
        evaluated=evaluated,
        top1_label_match_rate=_safe_rate(top1_matches, evaluated),
        topk_label_match_rate=_safe_rate(topk_matches, evaluated),
        encoder_version=LEARNED_ENCODER_VERSION,
    )


def evaluate_hard_split_report(
    index: CompactFeatureIndex,
    split_field: str = DEFAULT_SPLIT_FIELD,
    min_query_count: int = 1,
    max_holdouts: int | None = None,
    limit_per_holdout: int | None = None,
    top_k: int = 3,
    use_query_label: bool = False,
    batch_size: int = 256,
    include_prototype: bool = False,
    prototype_config: PrototypeEncoderConfig | None = None,
    progress_every: int = 0,
    progress_writer: ProgressWriter | None = None,
) -> HardSplitReport:
    comparisons = []
    holdout_values = holdout_values_for_split(index, split_field, min_query_count, max_holdouts)
    for position, holdout_value in enumerate(holdout_values, start=1):
        _write_report_progress("hard_split_report_holdout", holdout_value, position, len(holdout_values), progress_writer)
        feature_metrics = evaluate_feature_hard_split(
            index=index,
            split_field=split_field,
            holdout_values=(holdout_value,),
            limit=limit_per_holdout,
            top_k=top_k,
            use_query_label=use_query_label,
            metadata_baseline=False,
            batch_size=batch_size,
            progress_every=progress_every,
            progress_writer=progress_writer,
        )
        metadata_metrics = evaluate_feature_hard_split(
            index=index,
            split_field=split_field,
            holdout_values=(holdout_value,),
            limit=limit_per_holdout,
            top_k=top_k,
            use_query_label=use_query_label,
            metadata_baseline=True,
            batch_size=batch_size,
            progress_every=progress_every,
            progress_writer=progress_writer,
        )
        prototype_metrics = None
        if include_prototype:
            prototype_metrics = evaluate_prototype_hard_split(
                index=index,
                config=prototype_config or PrototypeEncoderConfig(),
                split_field=split_field,
                holdout_values=(holdout_value,),
                limit=limit_per_holdout,
                top_k=top_k,
                batch_size=batch_size,
                progress_every=progress_every,
                progress_writer=progress_writer,
            )
        comparisons.append(
            HardSplitComparison(
                holdout_value=holdout_value,
                feature_retrieval=feature_metrics,
                metadata_baseline=metadata_metrics,
                prototype_encoder=prototype_metrics,
            )
        )
    return HardSplitReport(
        split_field=split_field,
        top_k=top_k,
        use_query_label=use_query_label,
        min_query_count=min_query_count,
        limit_per_holdout=limit_per_holdout,
        comparisons=comparisons,
    )


def evaluate_prototype_hard_split_report(
    index: CompactFeatureIndex,
    config: PrototypeEncoderConfig,
    split_field: str = DEFAULT_SPLIT_FIELD,
    min_query_count: int = 1,
    max_holdouts: int | None = None,
    limit_per_holdout: int | None = None,
    top_k: int = 3,
    batch_size: int = 256,
    progress_every: int = 0,
    progress_writer: ProgressWriter | None = None,
) -> HardSplitModeReport:
    metrics = []
    holdout_values = holdout_values_for_split(index, split_field, min_query_count, max_holdouts)
    for position, holdout_value in enumerate(holdout_values, start=1):
        _write_report_progress("prototype_hard_split_report_holdout", holdout_value, position, len(holdout_values), progress_writer)
        metrics.append(
            evaluate_prototype_hard_split(
                index=index,
                config=config,
                split_field=split_field,
                holdout_values=(holdout_value,),
                limit=limit_per_holdout,
                top_k=top_k,
                batch_size=batch_size,
                progress_every=progress_every,
                progress_writer=progress_writer,
            )
        )
    return HardSplitModeReport(
        retrieval_mode="prototype_encoder",
        split_field=split_field,
        top_k=top_k,
        min_query_count=min_query_count,
        limit_per_holdout=limit_per_holdout,
        metrics=metrics,
    )


def evaluate_learned_hard_split_report(
    index: CompactFeatureIndex,
    config: LearnedEncoderConfig,
    split_field: str = DEFAULT_SPLIT_FIELD,
    min_query_count: int = 1,
    max_holdouts: int | None = None,
    limit_per_holdout: int | None = None,
    top_k: int = 3,
    batch_size: int = 256,
    progress_every: int = 0,
    progress_writer: ProgressWriter | None = None,
) -> HardSplitModeReport:
    metrics = []
    holdout_values = holdout_values_for_split(index, split_field, min_query_count, max_holdouts)
    for position, holdout_value in enumerate(holdout_values, start=1):
        _write_report_progress("learned_hard_split_report_holdout", holdout_value, position, len(holdout_values), progress_writer)
        metrics.append(
            evaluate_learned_hard_split(
                index=index,
                config=config,
                split_field=split_field,
                holdout_values=(holdout_value,),
                limit=limit_per_holdout,
                top_k=top_k,
                batch_size=batch_size,
                progress_every=progress_every,
                progress_writer=progress_writer,
            )
        )
    return HardSplitModeReport(
        retrieval_mode="learned_projection_encoder",
        split_field=split_field,
        top_k=top_k,
        min_query_count=min_query_count,
        limit_per_holdout=limit_per_holdout,
        metrics=metrics,
    )


def select_hard_split(
    index: CompactFeatureIndex,
    split_field: str,
    holdout_values: tuple[str, ...] = (),
    limit: int | None = None,
) -> HardSplitSelection:
    case_groups = np.asarray([_group_value(index, case_index, split_field) for case_index in range(index.case_count)], dtype=object)
    resolved_holdout_values = _resolve_holdout_values(case_groups, holdout_values)
    holdout_mask = np.isin(case_groups, np.asarray(resolved_holdout_values, dtype=object))
    train_indices = np.flatnonzero(~holdout_mask).astype(np.int32)
    query_indices = np.flatnonzero(holdout_mask & index.label_available).astype(np.int32)
    if limit is not None:
        query_indices = query_indices[: max(0, limit)]
    return HardSplitSelection(
        split_field=split_field,
        holdout_values=resolved_holdout_values,
        train_indices=train_indices,
        query_indices=query_indices,
    )


def holdout_values_for_split(
    index: CompactFeatureIndex,
    split_field: str,
    min_query_count: int = 1,
    max_holdouts: int | None = None,
) -> list[str]:
    counts = Counter(
        _group_value(index, case_index, split_field)
        for case_index in range(index.case_count)
        if index.label_available[case_index]
    )
    minimum_count = max(1, min_query_count)
    values = [
        value
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_count and count < index.case_count
    ]
    if max_holdouts is None:
        return values
    return values[: max(0, max_holdouts)]


def hard_split_report_to_markdown(report: HardSplitReport) -> str:
    include_prototype = any(comparison.prototype_encoder is not None for comparison in report.comparisons)
    headers = [
        "Holdout",
        "Query",
        "Train",
    ]
    headers.extend(_rate_headers("Feature", report.top_k))
    headers.extend(_rate_headers("Metadata", report.top_k))
    if include_prototype:
        headers.extend(_rate_headers("Prototype", report.top_k))
    lines = [
        f"# Hard Split Report: {report.split_field}",
        "",
        f"- Top-k: `{report.top_k}`",
        f"- Query label usage: `{'visible' if report.use_query_label else 'hidden'}`",
        f"- Min query count: `{report.min_query_count}`",
        f"- Limit per holdout: `{report.limit_per_holdout if report.limit_per_holdout is not None else 'none'}`",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for comparison in report.comparisons:
        row = [
            comparison.holdout_value,
            str(comparison.feature_retrieval.query_count),
            str(comparison.feature_retrieval.train_count),
        ]
        row.extend(_rate_cells(comparison.feature_retrieval, report.top_k))
        row.extend(_rate_cells(comparison.metadata_baseline, report.top_k))
        if include_prototype:
            row.extend(_prototype_rate_cells(comparison, report.top_k))
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines) + "\n"


def prototype_hard_split_report_to_markdown(report: HardSplitModeReport) -> str:
    return _mode_report_to_markdown(report, title="Prototype Hard Split Report", metric_label="Prototype")


def learned_hard_split_report_to_markdown(report: HardSplitModeReport) -> str:
    return _mode_report_to_markdown(report, title="Learned Projection Hard Split Report", metric_label="Learned")


def _mode_report_to_markdown(report: HardSplitModeReport, title: str, metric_label: str) -> str:
    headers = [
        "Holdout",
        "Query",
        "Train",
    ]
    headers.extend(_rate_headers(metric_label, report.top_k))
    lines = [
        f"# {title}: {report.split_field}",
        "",
        f"- Retrieval mode: `{report.retrieval_mode}`",
        f"- Top-k: `{report.top_k}`",
        f"- Min query count: `{report.min_query_count}`",
        f"- Limit per holdout: `{report.limit_per_holdout if report.limit_per_holdout is not None else 'none'}`",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for metrics in report.metrics:
        row = [
            ", ".join(metrics.holdout_values),
            str(metrics.query_count),
            str(metrics.train_count),
        ]
        row.extend(_rate_cells(metrics, report.top_k))
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines) + "\n"


def default_progress_writer(payload: dict[str, int | float | str]) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def hard_split_failures_to_markdown(sample: HardSplitFailureSample) -> str:
    lines = [
        f"# Hard Split Failure Samples: {sample.split_field}",
        "",
        f"- Retrieval mode: `{sample.retrieval_mode}`",
        f"- Holdout values: `{', '.join(sample.holdout_values)}`",
        f"- Train/query: `{sample.train_count}` / `{sample.query_count}`",
        f"- Inspected queries: `{sample.inspected}`",
        f"- Top-k: `{sample.top_k}`",
        "",
    ]
    if not sample.failures:
        lines.append("No top-k label failures found.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "|Query sample|Query label|Top neighbors|",
            "|---|---|---|",
        ]
    )
    for failure in sample.failures:
        query_label = _label_text(failure.query_label_id, failure.query_label_name)
        neighbors = "<br>".join(
            f"{neighbor.rank}. {neighbor.sample_id} ({_label_text(neighbor.label_id, neighbor.label_name)}, {neighbor.score:.6f})"
            for neighbor in failure.neighbors
        )
        lines.append(f"|{failure.query_sample_id}|{query_label}|{neighbors}|")
    return "\n".join(lines) + "\n"


def _feature_score_matrix_for_batch(
    index: CompactFeatureIndex,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
    use_query_label: bool,
    metadata_baseline: bool,
) -> np.ndarray:
    metadata_scores = _metadata_score_matrix(index, query_indices, candidate_indices)
    label_scores = (
        _label_score_matrix(index, query_indices, candidate_indices)
        if use_query_label
        else _missing_matrix(query_indices.size, candidate_indices.size)
    )
    if metadata_baseline:
        return _weighted_score_matrix([(metadata_scores, METADATA_BASELINE_WEIGHT), (label_scores, LABEL_BASELINE_WEIGHT)])
    return _weighted_score_matrix(
        [
            (
                _cosine_score_matrix(
                    index.image_vectors,
                    index.image_norms,
                    index.image_available,
                    query_indices,
                    candidate_indices,
                ),
                IMAGE_WEIGHT,
            ),
            (
                _cosine_score_matrix(
                    index.timeseries_vectors,
                    index.timeseries_norms,
                    index.timeseries_available,
                    query_indices,
                    candidate_indices,
                ),
                TIMESERIES_WEIGHT,
            ),
            (metadata_scores, METADATA_WEIGHT),
            (label_scores, LABEL_WEIGHT),
        ]
    )


def _cosine_score_matrix(
    vectors: np.ndarray,
    norms: np.ndarray,
    available: np.ndarray,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    query_vectors = vectors[query_indices]
    candidate_vectors = vectors[candidate_indices]
    query_norms = norms[query_indices]
    candidate_norms = norms[candidate_indices]
    raw_scores = np.matmul(query_vectors, candidate_vectors.T).astype(np.float32, copy=False)
    denominators = query_norms[:, None] * candidate_norms[None, :]
    valid = (
        (query_norms[:, None] > 0.0)
        & (candidate_norms[None, :] > 0.0)
        & available[query_indices, None]
        & available[candidate_indices][None, :]
    )
    scores = np.divide(raw_scores, denominators, out=_missing_matrix(query_indices.size, candidate_indices.size), where=valid)
    return np.clip(scores, 0.0, 1.0, out=scores)


def _metadata_score_matrix(index: CompactFeatureIndex, query_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    scores = np.zeros((query_indices.size, candidate_indices.size), dtype=np.float32)
    total_weights = np.zeros((query_indices.size, candidate_indices.size), dtype=np.float32)
    for field, weight in METADATA_WEIGHTS.items():
        values = index.metadata_values[field]
        query_values = values[query_indices]
        candidate_values = values[candidate_indices]
        available = (query_values != "")[:, None] & (candidate_values != "")[None, :]
        if not np.any(available):
            continue
        total_weights[available] += weight
        scores[available & (query_values[:, None] == candidate_values[None, :])] += weight
    return np.divide(scores, total_weights, out=_missing_matrix(query_indices.size, candidate_indices.size), where=total_weights > 0.0)


def _label_score_matrix(index: CompactFeatureIndex, query_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    scores = _missing_matrix(query_indices.size, candidate_indices.size)
    candidate_available = index.label_available[candidate_indices]
    if not np.any(candidate_available):
        return scores
    scores[:, candidate_available] = 0.0
    matches = index.label_ids[query_indices, None] == index.label_ids[candidate_indices][None, :]
    scores[candidate_available[None, :] & matches] = 1.0
    return scores


def _embedding_score_matrix(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_norms: np.ndarray,
) -> np.ndarray:
    query_norms = _row_norms(query_embeddings)
    raw_scores = query_embeddings @ candidate_embeddings.T
    denominators = query_norms[:, None] * candidate_norms[None, :]
    valid = (query_norms[:, None] > 0.0) & (candidate_norms[None, :] > 0.0)
    scores = np.divide(raw_scores, denominators, out=np.full(raw_scores.shape, np.nan, dtype=np.float32), where=valid)
    return np.clip(scores, 0.0, 1.0, out=scores)


def _weighted_score_matrix(weighted_scores: list[tuple[np.ndarray, float]]) -> np.ndarray:
    total = np.zeros(weighted_scores[0][0].shape, dtype=np.float32)
    weights = np.zeros(weighted_scores[0][0].shape, dtype=np.float32)
    for scores, weight in weighted_scores:
        available = np.isfinite(scores)
        total[available] += scores[available] * weight
        weights[available] += weight
    return np.divide(total, weights, out=np.zeros_like(total), where=weights > 0.0)


def _top_positions_matrix(scores: np.ndarray, candidate_tie_ranks: np.ndarray, top_k: int) -> np.ndarray:
    rows = np.full((scores.shape[0], top_k), -1, dtype=np.int32)
    for row_index, row_scores in enumerate(scores):
        top_positions = _top_positions_for_row(row_scores, candidate_tie_ranks, top_k)
        rows[row_index, : top_positions.size] = top_positions
    return rows


def _top_positions_for_row(scores: np.ndarray, candidate_tie_ranks: np.ndarray, top_k: int) -> np.ndarray:
    valid_positions = np.flatnonzero(np.isfinite(scores))
    if valid_positions.size <= top_k:
        return _sort_ranked_positions(valid_positions, scores, candidate_tie_ranks)
    valid_scores = scores[valid_positions]
    threshold_position = valid_scores.size - top_k
    threshold = np.partition(valid_scores, threshold_position)[threshold_position]
    candidate_positions = valid_positions[valid_scores >= threshold]
    return _sort_ranked_positions(candidate_positions, scores, candidate_tie_ranks)[:top_k]


def _sort_ranked_positions(positions: np.ndarray, scores: np.ndarray, candidate_tie_ranks: np.ndarray) -> np.ndarray:
    sort_order = np.lexsort((candidate_tie_ranks[positions], scores[positions]))
    return positions[sort_order[::-1]]


def _label_match_masks(
    index: CompactFeatureIndex,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
    top_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    has_results = top_positions[:, 0] >= 0
    top_labels = _labels_for_top_positions(index, candidate_indices, top_positions)
    query_labels = index.label_ids[query_indices]
    top1_mask = has_results & (top_labels[:, 0] == query_labels)
    topk_mask = has_results & np.any(top_labels == query_labels[:, None], axis=1)
    return has_results, top1_mask, topk_mask


def _labels_for_top_positions(index: CompactFeatureIndex, candidate_indices: np.ndarray, top_positions: np.ndarray) -> np.ndarray:
    labels = np.full(top_positions.shape, -1, dtype=np.int32)
    valid = top_positions >= 0
    labels[valid] = index.label_ids[candidate_indices[top_positions[valid]]]
    return labels


def _resolve_holdout_values(case_groups: np.ndarray, holdout_values: tuple[str, ...]) -> tuple[str, ...]:
    if holdout_values:
        return tuple(str(value) for value in holdout_values)
    counts = Counter(str(value) for value in case_groups)
    if not counts:
        return ()
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count < case_groups.size:
            return (value,)
    return (next(iter(counts)),)


def _group_value(index: CompactFeatureIndex, case_index: int, field: str) -> str:
    case = index.cases[case_index]
    if field == "label_id":
        return str(case.label_id) if case.label_id is not None else "unknown"
    if field == "label_name":
        return case.label_name or "unknown"
    value = case.metadata.get(field, "")
    return value if value != "" else "unknown"


def _empty_metrics(
    retrieval_mode: str,
    selection: HardSplitSelection,
    encoder_version: str | None = None,
) -> HardSplitMetrics:
    return HardSplitMetrics(
        retrieval_mode=retrieval_mode,
        split_field=selection.split_field,
        holdout_values=selection.holdout_values,
        train_count=selection.train_count,
        query_count=selection.query_count,
        evaluated=0,
        top1_label_match_rate=0.0,
        topk_label_match_rate=0.0,
        encoder_version=encoder_version,
    )


def _empty_failure_sample(
    retrieval_mode: str,
    selection: HardSplitSelection,
    top_k: int,
) -> HardSplitFailureSample:
    return _failure_sample(retrieval_mode, selection, inspected=0, top_k=top_k, failures=[])


def _failure_sample(
    retrieval_mode: str,
    selection: HardSplitSelection,
    inspected: int,
    top_k: int,
    failures: list[HardSplitFailureCase],
) -> HardSplitFailureSample:
    return HardSplitFailureSample(
        retrieval_mode=retrieval_mode,
        split_field=selection.split_field,
        holdout_values=selection.holdout_values,
        train_count=selection.train_count,
        query_count=selection.query_count,
        inspected=inspected,
        top_k=top_k,
        failures=failures,
    )


def _failure_case(
    index: CompactFeatureIndex,
    query_index: int,
    candidate_indices: np.ndarray,
    top_positions: np.ndarray,
    scores: np.ndarray,
) -> HardSplitFailureCase:
    query = index.cases[query_index]
    return HardSplitFailureCase(
        query_sample_id=query.sample_id,
        query_label_id=query.label_id,
        query_label_name=query.label_name,
        query_image_path=query.image_path,
        query_timeseries_path=query.timeseries_path,
        query_metadata=query.metadata,
        neighbors=_neighbors(index, candidate_indices, top_positions, scores),
    )


def _neighbors(
    index: CompactFeatureIndex,
    candidate_indices: np.ndarray,
    top_positions: np.ndarray,
    scores: np.ndarray,
) -> list[HardSplitNeighbor]:
    neighbors = []
    for rank, position in enumerate(top_positions, start=1):
        if position < 0:
            continue
        candidate = index.cases[int(candidate_indices[position])]
        neighbors.append(
            HardSplitNeighbor(
                rank=rank,
                sample_id=candidate.sample_id,
                label_id=candidate.label_id,
                label_name=candidate.label_name,
                score=float(scores[position]),
                image_path=candidate.image_path,
                timeseries_path=candidate.timeseries_path,
                metadata=candidate.metadata,
            )
        )
    return neighbors


def _batches(indices: np.ndarray, batch_size: int):
    normalized_batch_size = max(1, batch_size)
    for start in range(0, indices.size, normalized_batch_size):
        yield indices[start : start + normalized_batch_size]


def _row_norms(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.linalg.norm(matrix, axis=1).astype(np.float32)


def _missing_matrix(row_count: int, column_count: int) -> np.ndarray:
    return np.full((row_count, column_count), np.nan, dtype=np.float32)


def _maybe_write_progress(
    mode: str,
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
            "event": "hard_split_progress",
            "mode": mode,
            "done": completed,
            "total": total,
            "queries_per_second": round(queries_per_second, 2),
            "eta_seconds": round(remaining_seconds, 1),
        }
    )


def _write_report_progress(
    event: str,
    holdout_value: str,
    position: int,
    total: int,
    progress_writer: ProgressWriter | None,
) -> None:
    if progress_writer is None:
        return
    progress_writer(
        {
            "event": event,
            "holdout_value": holdout_value,
            "done": position,
            "total": total,
        }
    )


def _rate_headers(name: str, top_k: int) -> list[str]:
    if top_k <= 1:
        return [f"{name} top-1"]
    return [f"{name} top-1", f"{name} top-{top_k}"]


def _rate_cells(metrics: HardSplitMetrics, top_k: int) -> list[str]:
    if top_k <= 1:
        return [_format_rate(metrics.top1_label_match_rate)]
    return [
        _format_rate(metrics.top1_label_match_rate),
        _format_rate(metrics.topk_label_match_rate),
    ]


def _prototype_rate_cells(comparison: HardSplitComparison, top_k: int) -> list[str]:
    if comparison.prototype_encoder is None:
        return [""] if top_k <= 1 else ["", ""]
    return _rate_cells(comparison.prototype_encoder, top_k)


def _format_rate(value: float) -> str:
    return f"{value:.6f}"


def _label_text(label_id: int | None, label_name: str) -> str:
    if label_id is None:
        return label_name or "unknown"
    if label_name:
        return f"{label_name} ({label_id})"
    return str(label_id)


def _safe_rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 6)
