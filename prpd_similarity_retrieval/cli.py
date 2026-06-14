from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from prpd_similarity_retrieval.batch_evaluation import BREAKDOWN_FIELDS, default_progress_writer, evaluate_compact_index
from prpd_similarity_retrieval.compact_index import (
    CompactFeatureIndex,
    is_compact_index_path,
    load_compact_feature_index,
    save_compact_feature_index,
)
from prpd_similarity_retrieval.features import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST_PATH,
    extract_case_features,
    load_manifest_cases,
)
from prpd_similarity_retrieval.feature_cache import append_feature_cache, load_feature_cache_map
from prpd_similarity_retrieval.hard_split_evaluation import (
    DEFAULT_SPLIT_FIELD,
    SPLIT_FIELDS,
    default_progress_writer as default_hard_split_progress_writer,
    evaluate_hard_split_report,
    evaluate_feature_hard_split,
    evaluate_prototype_hard_split,
    evaluate_prototype_hard_split_report,
    hard_split_failures_to_markdown,
    hard_split_report_to_markdown,
    prototype_hard_split_report_to_markdown,
    sample_feature_hard_split_failures,
)
from prpd_similarity_retrieval.human_review import (
    evaluate_human_reviews,
    human_review_metrics_to_markdown,
    load_human_review_records,
)
from prpd_similarity_retrieval.models import CaseFeatures, CaseRecord, SearchResult
from prpd_similarity_retrieval.prototype_encoder import (
    PrototypeEncoderConfig,
    build_prototype_embedding_index,
    evaluate_prototype_index,
    load_prototype_embedding_index,
    prototype_results_to_json,
    save_prototype_embedding_index,
)
from prpd_similarity_retrieval.retrieval import (
    build_feature_index,
    find_case,
    load_feature_index,
    results_to_json,
    save_feature_index,
    search_metadata_baseline,
    search_similar_cases,
)
from prpd_similarity_retrieval.review_artifact import hard_split_failure_review_to_html


DEFAULT_INDEX_PATH = Path("prpd_similarity_retrieval/case_feature_index.npz")
DEFAULT_PROTOTYPE_INDEX_PATH = Path("prpd_similarity_retrieval/case_embedding_index.prototype.npz")
FeatureIndex = CompactFeatureIndex | list[CaseFeatures]


@dataclass(frozen=True, slots=True)
class BuildResult:
    features: list[CaseFeatures]
    cache_hit_count: int
    extracted_count: int


def main() -> None:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query PRPD/time-series feature similarity index.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_index_parser = subparsers.add_parser("build-index", help="Build feature index from manifest.csv.")
    build_index_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    build_index_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    build_index_parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)
    build_index_parser.add_argument("--limit", type=int, default=None)
    build_index_parser.add_argument("--per-label-limit", type=int, default=None)
    build_index_parser.add_argument("--progress-every", type=int, default=1000)
    build_index_parser.add_argument("--workers", type=int, default=1)
    build_index_parser.add_argument("--cache", type=Path, default=None)
    build_index_parser.set_defaults(handler=handle_build_index)

    query_sample_parser = subparsers.add_parser("query-sample", help="Find similar cases for a sample already in the index.")
    query_sample_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    query_sample_parser.add_argument("--sample-id", required=True)
    query_sample_parser.add_argument("--top-k", type=int, default=5)
    query_sample_parser.set_defaults(handler=handle_query_sample)

    query_files_parser = subparsers.add_parser("query-files", help="Find similar cases for external PRPD/CSV files.")
    query_files_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    query_files_parser.add_argument("--sample-id", default="current_inspection")
    query_files_parser.add_argument("--image-path", type=Path, default=None)
    query_files_parser.add_argument("--timeseries-path", type=Path, default=None)
    query_files_parser.add_argument("--metadata-json", type=Path, default=None)
    query_files_parser.add_argument("--label-id", type=int, default=None)
    query_files_parser.add_argument("--label-name", default="")
    query_files_parser.add_argument("--top-k", type=int, default=5)
    query_files_parser.set_defaults(handler=handle_query_files)

    evaluate_parser = subparsers.add_parser("evaluate-index", help="Run leave-one-out label-match evaluation.")
    evaluate_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    evaluate_parser.add_argument("--limit", type=int, default=None)
    evaluate_parser.add_argument("--top-k", type=int, default=3)
    evaluate_parser.add_argument("--use-query-label", action="store_true")
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.add_argument("--progress-every", type=int, default=0)
    evaluate_parser.add_argument("--breakdown-field", action="append", choices=BREAKDOWN_FIELDS, default=[])
    evaluate_parser.set_defaults(handler=handle_evaluate_index)

    compare_parser = subparsers.add_parser("compare-baseline", help="Compare feature retrieval against metadata-only baseline.")
    compare_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    compare_parser.add_argument("--limit", type=int, default=None)
    compare_parser.add_argument("--top-k", type=int, default=3)
    compare_parser.add_argument("--use-query-label", action="store_true")
    compare_parser.add_argument("--batch-size", type=int, default=64)
    compare_parser.add_argument("--progress-every", type=int, default=0)
    compare_parser.add_argument("--breakdown-field", action="append", choices=BREAKDOWN_FIELDS, default=[])
    compare_parser.set_defaults(handler=handle_compare_baseline)

    hard_split_parser = subparsers.add_parser(
        "evaluate-hard-split",
        help="Evaluate retrieval with a group holdout query split and train-only candidates.",
    )
    hard_split_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    hard_split_parser.add_argument("--split-field", choices=SPLIT_FIELDS, default=DEFAULT_SPLIT_FIELD)
    hard_split_parser.add_argument("--holdout-value", action="append", default=[])
    hard_split_parser.add_argument("--limit", type=int, default=None)
    hard_split_parser.add_argument("--top-k", type=int, default=3)
    hard_split_parser.add_argument("--use-query-label", action="store_true")
    hard_split_parser.add_argument("--batch-size", type=int, default=256)
    hard_split_parser.add_argument("--progress-every", type=int, default=0)
    hard_split_parser.add_argument("--include-prototype", action="store_true")
    hard_split_parser.add_argument("--image-dim", type=int, default=128)
    hard_split_parser.add_argument("--timeseries-dim", type=int, default=64)
    hard_split_parser.add_argument("--centroid-weight", type=float, default=0.30)
    hard_split_parser.add_argument("--image-weight", type=float, default=0.55)
    hard_split_parser.add_argument("--timeseries-weight", type=float, default=0.45)
    hard_split_parser.add_argument("--random-seed", type=int, default=42)
    hard_split_parser.set_defaults(handler=handle_evaluate_hard_split)

    hard_split_report_parser = subparsers.add_parser(
        "evaluate-hard-split-report",
        help="Evaluate every holdout group for a split field and produce a report.",
    )
    hard_split_report_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    hard_split_report_parser.add_argument("--split-field", choices=SPLIT_FIELDS, default=DEFAULT_SPLIT_FIELD)
    hard_split_report_parser.add_argument("--min-query-count", type=int, default=1)
    hard_split_report_parser.add_argument("--max-holdouts", type=int, default=None)
    hard_split_report_parser.add_argument("--limit-per-holdout", type=int, default=None)
    hard_split_report_parser.add_argument("--top-k", type=int, default=3)
    hard_split_report_parser.add_argument("--use-query-label", action="store_true")
    hard_split_report_parser.add_argument("--batch-size", type=int, default=256)
    hard_split_report_parser.add_argument("--progress-every", type=int, default=0)
    hard_split_report_parser.add_argument("--include-prototype", action="store_true")
    hard_split_report_parser.add_argument("--image-dim", type=int, default=128)
    hard_split_report_parser.add_argument("--timeseries-dim", type=int, default=64)
    hard_split_report_parser.add_argument("--centroid-weight", type=float, default=0.30)
    hard_split_report_parser.add_argument("--image-weight", type=float, default=0.55)
    hard_split_report_parser.add_argument("--timeseries-weight", type=float, default=0.45)
    hard_split_report_parser.add_argument("--random-seed", type=int, default=42)
    hard_split_report_parser.add_argument("--format", choices=("json", "markdown"), default="json")
    hard_split_report_parser.add_argument("--output", type=Path, default=None)
    hard_split_report_parser.set_defaults(handler=handle_evaluate_hard_split_report)

    prototype_hard_split_report_parser = subparsers.add_parser(
        "evaluate-prototype-hard-split-report",
        help="Evaluate every holdout group with the prototype encoder only.",
    )
    prototype_hard_split_report_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    prototype_hard_split_report_parser.add_argument("--split-field", choices=SPLIT_FIELDS, default=DEFAULT_SPLIT_FIELD)
    prototype_hard_split_report_parser.add_argument("--min-query-count", type=int, default=1)
    prototype_hard_split_report_parser.add_argument("--max-holdouts", type=int, default=None)
    prototype_hard_split_report_parser.add_argument("--limit-per-holdout", type=int, default=None)
    prototype_hard_split_report_parser.add_argument("--top-k", type=int, default=3)
    prototype_hard_split_report_parser.add_argument("--batch-size", type=int, default=256)
    prototype_hard_split_report_parser.add_argument("--progress-every", type=int, default=0)
    prototype_hard_split_report_parser.add_argument("--image-dim", type=int, default=128)
    prototype_hard_split_report_parser.add_argument("--timeseries-dim", type=int, default=64)
    prototype_hard_split_report_parser.add_argument("--centroid-weight", type=float, default=0.30)
    prototype_hard_split_report_parser.add_argument("--image-weight", type=float, default=0.55)
    prototype_hard_split_report_parser.add_argument("--timeseries-weight", type=float, default=0.45)
    prototype_hard_split_report_parser.add_argument("--random-seed", type=int, default=42)
    prototype_hard_split_report_parser.add_argument("--format", choices=("json", "markdown"), default="json")
    prototype_hard_split_report_parser.add_argument("--output", type=Path, default=None)
    prototype_hard_split_report_parser.set_defaults(handler=handle_evaluate_prototype_hard_split_report)

    hard_split_failure_parser = subparsers.add_parser(
        "sample-hard-split-failures",
        help="Sample holdout queries whose top-k retrieved cases miss the query label.",
    )
    hard_split_failure_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    hard_split_failure_parser.add_argument("--split-field", choices=SPLIT_FIELDS, default=DEFAULT_SPLIT_FIELD)
    hard_split_failure_parser.add_argument("--holdout-value", action="append", default=[])
    hard_split_failure_parser.add_argument("--limit", type=int, default=None)
    hard_split_failure_parser.add_argument("--top-k", type=int, default=3)
    hard_split_failure_parser.add_argument("--max-failures", type=int, default=10)
    hard_split_failure_parser.add_argument("--use-query-label", action="store_true")
    hard_split_failure_parser.add_argument("--metadata-baseline", action="store_true")
    hard_split_failure_parser.add_argument("--batch-size", type=int, default=128)
    hard_split_failure_parser.add_argument("--format", choices=("json", "markdown", "html"), default="json")
    hard_split_failure_parser.add_argument("--output", type=Path, default=None)
    hard_split_failure_parser.set_defaults(handler=handle_sample_hard_split_failures)

    human_review_parser = subparsers.add_parser(
        "evaluate-human-reviews",
        help="Evaluate CSV/JSON exported from hard split HTML review pages.",
    )
    human_review_parser.add_argument("--input", type=Path, action="append", required=True)
    human_review_parser.add_argument("--top-k", type=int, default=3)
    human_review_parser.add_argument("--accepted-value", action="append", default=["similar"])
    human_review_parser.add_argument("--breakdown-field", action="append", default=[])
    human_review_parser.add_argument("--format", choices=("json", "markdown"), default="json")
    human_review_parser.add_argument("--output", type=Path, default=None)
    human_review_parser.set_defaults(handler=handle_evaluate_human_reviews)

    prototype_build_parser = subparsers.add_parser("build-prototype-index", help="Build prototype image/time-series embedding index from feature index.")
    prototype_build_parser.add_argument("--feature-index", type=Path, default=DEFAULT_INDEX_PATH)
    prototype_build_parser.add_argument("--output", type=Path, default=DEFAULT_PROTOTYPE_INDEX_PATH)
    prototype_build_parser.add_argument("--image-dim", type=int, default=128)
    prototype_build_parser.add_argument("--timeseries-dim", type=int, default=64)
    prototype_build_parser.add_argument("--centroid-weight", type=float, default=0.30)
    prototype_build_parser.add_argument("--image-weight", type=float, default=0.55)
    prototype_build_parser.add_argument("--timeseries-weight", type=float, default=0.45)
    prototype_build_parser.add_argument("--random-seed", type=int, default=42)
    prototype_build_parser.set_defaults(handler=handle_build_prototype_index)

    prototype_query_parser = subparsers.add_parser("query-prototype-sample", help="Find similar cases with prototype embedding index.")
    prototype_query_parser.add_argument("--index", type=Path, default=DEFAULT_PROTOTYPE_INDEX_PATH)
    prototype_query_parser.add_argument("--sample-id", required=True)
    prototype_query_parser.add_argument("--top-k", type=int, default=5)
    prototype_query_parser.set_defaults(handler=handle_query_prototype_sample)

    prototype_eval_parser = subparsers.add_parser("evaluate-prototype-index", help="Evaluate prototype embedding index by leave-one-out label match.")
    prototype_eval_parser.add_argument("--index", type=Path, default=DEFAULT_PROTOTYPE_INDEX_PATH)
    prototype_eval_parser.add_argument("--limit", type=int, default=None)
    prototype_eval_parser.add_argument("--top-k", type=int, default=3)
    prototype_eval_parser.add_argument("--batch-size", type=int, default=256)
    prototype_eval_parser.set_defaults(handler=handle_evaluate_prototype_index)
    return parser


def handle_build_index(args: argparse.Namespace) -> None:
    cases = load_manifest_cases(args.manifest, args.data_root, args.limit)
    cases = _select_per_label(cases, args.per_label_limit)
    build_result = _build_features(cases, args.progress_every, args.workers, args.cache)
    features = build_result.features
    _save_index(args.output, features)
    complete_image_count = sum(case.image_vector is not None for case in features)
    complete_timeseries_count = sum(case.timeseries_vector is not None for case in features)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "index_format": _index_format(args.output),
                "case_count": len(features),
                "image_feature_count": complete_image_count,
                "timeseries_feature_count": complete_timeseries_count,
                "cache": str(args.cache) if args.cache is not None else None,
                "cache_hit_count": build_result.cache_hit_count,
                "extracted_count": build_result.extracted_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _build_features(
    cases: list[CaseRecord],
    progress_every: int,
    workers: int,
    cache_path: Path | None,
) -> BuildResult:
    if cache_path is None:
        features = _build_feature_index(cases, progress_every, workers)
        return BuildResult(features=features, cache_hit_count=0, extracted_count=len(features))

    cached_features = load_feature_cache_map(cache_path)
    missing_cases = [case for case in cases if case.sample_id not in cached_features]
    _write_cache_status(cache_path, cache_hit_count=len(cases) - len(missing_cases), missing_count=len(missing_cases))
    extracted_features = _build_feature_index(missing_cases, progress_every, workers)
    append_feature_cache(cache_path, extracted_features)
    for feature in extracted_features:
        cached_features[feature.sample_id] = feature
    return BuildResult(
        features=[cached_features[case.sample_id] for case in cases],
        cache_hit_count=len(cases) - len(missing_cases),
        extracted_count=len(extracted_features),
    )


def _build_feature_index(cases: list[CaseRecord], progress_every: int, workers: int) -> list[CaseFeatures]:
    if progress_every <= 0 and workers <= 1:
        return build_feature_index(cases)
    if workers > 1:
        return _build_feature_index_parallel(cases, progress_every, workers)

    started_at = time.perf_counter()
    features: list[CaseFeatures] = []
    total = len(cases)
    for position, case in enumerate(cases, start=1):
        features.append(extract_case_features(case))
        if position % progress_every == 0 or position == total:
            _write_progress(position, total, started_at)
    return features


def _build_feature_index_parallel(
    cases: list[CaseRecord],
    progress_every: int,
    workers: int,
) -> list[CaseFeatures]:
    started_at = time.perf_counter()
    features: list[CaseFeatures] = []
    total = len(cases)
    chunk_size = max(1, min(128, total // max(workers * 8, 1)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for position, case_features in enumerate(executor.map(extract_case_features, cases, chunksize=chunk_size), start=1):
            features.append(case_features)
            if progress_every > 0 and (position % progress_every == 0 or position == total):
                _write_progress(position, total, started_at)
    return features


def _write_cache_status(cache_path: Path, cache_hit_count: int, missing_count: int) -> None:
    print(
        json.dumps(
            {
                "event": "cache_status",
                "cache": str(cache_path),
                "cache_hit_count": cache_hit_count,
                "missing_count": missing_count,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def _write_progress(done: int, total: int, started_at: float) -> None:
    elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
    cases_per_second = done / elapsed_seconds
    remaining_seconds = (total - done) / cases_per_second if cases_per_second > 0 else 0.0
    print(
        json.dumps(
            {
                "event": "build_progress",
                "done": done,
                "total": total,
                "cases_per_second": round(cases_per_second, 2),
                "eta_seconds": round(remaining_seconds, 1),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def handle_query_sample(args: argparse.Namespace) -> None:
    index = _load_index(args.index)
    query = _find_case(index, args.sample_id)
    results = _search_index(index, query, top_k=args.top_k)
    print(results_to_json(results))


def _select_per_label(cases: list[CaseRecord], per_label_limit: int | None) -> list[CaseRecord]:
    if per_label_limit is None:
        return cases
    selected: list[CaseRecord] = []
    counts: dict[int | None, int] = {}
    for case in cases:
        current_count = counts.get(case.label_id, 0)
        if current_count >= per_label_limit:
            continue
        selected.append(case)
        counts[case.label_id] = current_count + 1
    return selected


def handle_query_files(args: argparse.Namespace) -> None:
    index = _load_index(args.index)
    metadata = _load_metadata(args.metadata_json)
    query = extract_case_features(
        CaseRecord(
            sample_id=args.sample_id,
            label_id=args.label_id,
            label_name=args.label_name,
            image_path=args.image_path,
            timeseries_path=args.timeseries_path,
            metadata=metadata,
        )
    )
    results = _search_index(index, query, top_k=args.top_k, exclude_self=False)
    print(results_to_json(results))


def handle_evaluate_index(args: argparse.Namespace) -> None:
    index = _load_index(args.index)
    metrics = _evaluate_index(
        index,
        args.limit,
        args.top_k,
        args.use_query_label,
        metadata_baseline=False,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        breakdown_fields=tuple(args.breakdown_field),
    )
    payload = {
        "evaluated": metrics["evaluated"],
        "top_k": args.top_k,
        "use_query_label": args.use_query_label,
        "top1_label_match_rate": metrics["top1_label_match_rate"],
        "topk_label_match_rate": metrics["topk_label_match_rate"],
    }
    if "breakdowns" in metrics:
        payload["breakdowns"] = metrics["breakdowns"]
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_compare_baseline(args: argparse.Namespace) -> None:
    index = _load_index(args.index)
    feature_metrics = _evaluate_index(
        index,
        args.limit,
        args.top_k,
        args.use_query_label,
        metadata_baseline=False,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        breakdown_fields=tuple(args.breakdown_field),
    )
    metadata_metrics = _evaluate_index(
        index,
        args.limit,
        args.top_k,
        args.use_query_label,
        metadata_baseline=True,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        breakdown_fields=tuple(args.breakdown_field),
    )
    print(
        json.dumps(
            {
                "evaluated": feature_metrics["evaluated"],
                "top_k": args.top_k,
                "use_query_label": args.use_query_label,
                "feature_retrieval": feature_metrics,
                "metadata_baseline": metadata_metrics,
                "delta": {
                    "top1_label_match_rate": round(
                        feature_metrics["top1_label_match_rate"] - metadata_metrics["top1_label_match_rate"],
                        6,
                    ),
                    "topk_label_match_rate": round(
                        feature_metrics["topk_label_match_rate"] - metadata_metrics["topk_label_match_rate"],
                        6,
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_build_prototype_index(args: argparse.Namespace) -> None:
    feature_index = _load_compact_feature_index_for_prototype(args.feature_index)
    config = _prototype_config_from_args(args)
    prototype_index = build_prototype_embedding_index(feature_index, config)
    save_prototype_embedding_index(args.output, prototype_index)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "encoder_version": "prototype_centroid_encoder_v1",
                "case_count": prototype_index.case_count,
                "embedding_dim": int(prototype_index.embeddings.shape[1]),
                "image_dim": config.image_dim,
                "timeseries_dim": config.timeseries_dim,
                "centroid_weight": config.centroid_weight,
                "image_weight": config.image_weight,
                "timeseries_weight": config.timeseries_weight,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_evaluate_hard_split(args: argparse.Namespace) -> None:
    index = _load_compact_feature_index_for_hard_split(args.index)
    holdout_values = tuple(args.holdout_value)
    feature_metrics = evaluate_feature_hard_split(
        index=index,
        split_field=args.split_field,
        holdout_values=holdout_values,
        limit=args.limit,
        top_k=args.top_k,
        use_query_label=args.use_query_label,
        metadata_baseline=False,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        progress_writer=default_hard_split_progress_writer,
    )
    metadata_metrics = evaluate_feature_hard_split(
        index=index,
        split_field=args.split_field,
        holdout_values=holdout_values,
        limit=args.limit,
        top_k=args.top_k,
        use_query_label=args.use_query_label,
        metadata_baseline=True,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        progress_writer=default_hard_split_progress_writer,
    )
    payload = {
        "top_k": args.top_k,
        "use_query_label": args.use_query_label,
        "feature_retrieval": feature_metrics.to_dict(),
        "metadata_baseline": metadata_metrics.to_dict(),
        "delta": {
            "top1_label_match_rate": round(feature_metrics.top1_label_match_rate - metadata_metrics.top1_label_match_rate, 6),
            "topk_label_match_rate": round(feature_metrics.topk_label_match_rate - metadata_metrics.topk_label_match_rate, 6),
        },
    }
    if args.include_prototype:
        prototype_metrics = evaluate_prototype_hard_split(
            index=index,
            config=_prototype_config_from_args(args),
            split_field=args.split_field,
            holdout_values=holdout_values,
            limit=args.limit,
            top_k=args.top_k,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            progress_writer=default_hard_split_progress_writer,
        )
        payload["prototype_encoder"] = prototype_metrics.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def handle_evaluate_hard_split_report(args: argparse.Namespace) -> None:
    index = _load_compact_feature_index_for_hard_split(args.index)
    report = evaluate_hard_split_report(
        index=index,
        split_field=args.split_field,
        min_query_count=args.min_query_count,
        max_holdouts=args.max_holdouts,
        limit_per_holdout=args.limit_per_holdout,
        top_k=args.top_k,
        use_query_label=args.use_query_label,
        batch_size=args.batch_size,
        include_prototype=args.include_prototype,
        prototype_config=_prototype_config_from_args(args),
        progress_every=args.progress_every,
        progress_writer=default_hard_split_progress_writer,
    )
    content = _format_hard_split_report(report, args.format)
    if args.output is None:
        print(content)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "format": args.format,
                "split_field": report.split_field,
                "holdout_count": len(report.comparisons),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_evaluate_prototype_hard_split_report(args: argparse.Namespace) -> None:
    index = _load_compact_feature_index_for_hard_split(args.index)
    report = evaluate_prototype_hard_split_report(
        index=index,
        config=_prototype_config_from_args(args),
        split_field=args.split_field,
        min_query_count=args.min_query_count,
        max_holdouts=args.max_holdouts,
        limit_per_holdout=args.limit_per_holdout,
        top_k=args.top_k,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        progress_writer=default_hard_split_progress_writer,
    )
    content = _format_prototype_hard_split_report(report, args.format)
    if args.output is None:
        print(content)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "format": args.format,
                "split_field": report.split_field,
                "holdout_count": len(report.metrics),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_sample_hard_split_failures(args: argparse.Namespace) -> None:
    index = _load_compact_feature_index_for_hard_split(args.index)
    sample = sample_feature_hard_split_failures(
        index=index,
        split_field=args.split_field,
        holdout_values=tuple(args.holdout_value),
        limit=args.limit,
        top_k=args.top_k,
        max_failures=args.max_failures,
        use_query_label=args.use_query_label,
        metadata_baseline=args.metadata_baseline,
        batch_size=args.batch_size,
    )
    content = _format_hard_split_failure_sample(sample, args.format, args.output)
    if args.output is None:
        print(content)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "format": args.format,
                "split_field": sample.split_field,
                "holdout_values": list(sample.holdout_values),
                "failure_count": len(sample.failures),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_evaluate_human_reviews(args: argparse.Namespace) -> None:
    records = load_human_review_records(args.input)
    metrics = evaluate_human_reviews(
        records=records,
        top_k=args.top_k,
        accepted_values=tuple(args.accepted_value),
        breakdown_fields=tuple(args.breakdown_field),
    )
    content = _format_human_review_metrics(metrics, args.format)
    if args.output is None:
        print(content)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "format": args.format,
                "reviewed_rows": metrics.reviewed_rows,
                "reviewed_queries": metrics.reviewed_queries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_query_prototype_sample(args: argparse.Namespace) -> None:
    index = load_prototype_embedding_index(args.index)
    results = index.search_sample(args.sample_id, top_k=args.top_k)
    print(prototype_results_to_json(results))


def handle_evaluate_prototype_index(args: argparse.Namespace) -> None:
    index = load_prototype_embedding_index(args.index)
    metrics = evaluate_prototype_index(index, limit=args.limit, top_k=args.top_k, batch_size=args.batch_size)
    print(
        json.dumps(
            {
                "evaluated": metrics.evaluated,
                "top_k": args.top_k,
                "encoder_version": "prototype_centroid_encoder_v1",
                "top1_label_match_rate": metrics.top1_label_match_rate,
                "topk_label_match_rate": metrics.topk_label_match_rate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _save_index(path: Path, features: list[CaseFeatures]) -> None:
    if is_compact_index_path(path):
        save_compact_feature_index(path, features)
        return
    save_feature_index(path, features)


def _load_index(path: Path) -> FeatureIndex:
    if is_compact_index_path(path):
        return load_compact_feature_index(path)
    return load_feature_index(path)


def _load_compact_feature_index_for_prototype(path: Path) -> CompactFeatureIndex:
    index = _load_index(path)
    if not isinstance(index, CompactFeatureIndex):
        raise ValueError("build-prototype-index requires a compact .npz feature index")
    return index


def _load_compact_feature_index_for_hard_split(path: Path) -> CompactFeatureIndex:
    index = _load_index(path)
    if not isinstance(index, CompactFeatureIndex):
        raise ValueError("evaluate-hard-split requires a compact .npz feature index")
    return index


def _prototype_config_from_args(args: argparse.Namespace) -> PrototypeEncoderConfig:
    return PrototypeEncoderConfig(
        image_dim=args.image_dim,
        timeseries_dim=args.timeseries_dim,
        centroid_weight=args.centroid_weight,
        image_weight=args.image_weight,
        timeseries_weight=args.timeseries_weight,
        random_seed=args.random_seed,
    )


def _format_hard_split_report(report, output_format: str) -> str:
    if output_format == "markdown":
        return hard_split_report_to_markdown(report)
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)


def _format_prototype_hard_split_report(report, output_format: str) -> str:
    if output_format == "markdown":
        return prototype_hard_split_report_to_markdown(report)
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)


def _format_hard_split_failure_sample(sample, output_format: str, output_path: Path | None = None) -> str:
    if output_format == "html":
        return hard_split_failure_review_to_html(sample, output_path)
    if output_format == "markdown":
        return hard_split_failures_to_markdown(sample)
    return json.dumps(sample.to_dict(), ensure_ascii=False, indent=2)


def _format_human_review_metrics(metrics, output_format: str) -> str:
    if output_format == "markdown":
        return human_review_metrics_to_markdown(metrics)
    return json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2)


def _index_format(path: Path) -> str:
    return "compact_npz" if is_compact_index_path(path) else "json"


def _find_case(index: FeatureIndex, sample_id: str) -> CaseFeatures:
    if isinstance(index, CompactFeatureIndex):
        return index.find_case(sample_id)
    return find_case(index, sample_id)


def _search_index(
    index: FeatureIndex,
    query: CaseFeatures,
    top_k: int,
    exclude_self: bool = True,
    metadata_baseline: bool = False,
) -> list[SearchResult]:
    if top_k <= 0:
        return []
    if isinstance(index, CompactFeatureIndex):
        if metadata_baseline:
            return index.search_metadata_baseline(query, top_k=top_k, exclude_self=exclude_self)
        return index.search_similar_cases(query, top_k=top_k, exclude_self=exclude_self)
    if metadata_baseline:
        return search_metadata_baseline(query, index, top_k=top_k, exclude_self=exclude_self)
    return search_similar_cases(query, index, top_k=top_k, exclude_self=exclude_self)


def _evaluate_index(
    index: FeatureIndex,
    limit: int | None,
    top_k: int,
    use_query_label: bool,
    metadata_baseline: bool,
    batch_size: int = 64,
    progress_every: int = 0,
    breakdown_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    if isinstance(index, CompactFeatureIndex):
        return evaluate_compact_index(
            index=index,
            limit=limit,
            top_k=top_k,
            use_query_label=use_query_label,
            metadata_baseline=metadata_baseline,
            batch_size=batch_size,
            progress_every=progress_every,
            progress_writer=default_progress_writer,
            breakdown_fields=breakdown_fields,
        ).to_dict()

    top1_matches = 0
    topk_matches = 0
    evaluated = 0
    for query in _iter_query_cases(index, limit):
        if query.label_id is None:
            continue
        retrieval_query = query if use_query_label else replace(query, label_id=None, label_name="")
        results = _search_index(index, retrieval_query, top_k, metadata_baseline=metadata_baseline)
        if not results:
            continue
        evaluated += 1
        top1_matches += int(results[0].case.label_id == query.label_id)
        topk_matches += int(any(result.case.label_id == query.label_id for result in results))
    return {
        "evaluated": evaluated,
        "top1_label_match_rate": _safe_rate(top1_matches, evaluated),
        "topk_label_match_rate": _safe_rate(topk_matches, evaluated),
    }


def _iter_query_cases(index: FeatureIndex, limit: int | None):
    if isinstance(index, CompactFeatureIndex):
        count = index.case_count if limit is None else min(limit, index.case_count)
        for position in range(count):
            yield index.case_at(position)
        return

    query_cases = index if limit is None else index[:limit]
    yield from query_cases


def _load_metadata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metadata-json must contain an object")
    return {str(key): _stringify_metadata_value(value) for key, value in payload.items()}


def _stringify_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(_stringify_metadata_value(item) for item in value)
    return str(value)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _safe_rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 6)


if __name__ == "__main__":
    main()
