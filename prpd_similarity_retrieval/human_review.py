from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RELEVANCE_VALUES = ("similar", "uncertain", "not_similar")
DEFAULT_ACCEPTED_VALUES = ("similar",)
DEFAULT_PARTIAL_VALUES = ("similar", "uncertain")


@dataclass(frozen=True, slots=True)
class HumanReviewRecord:
    payload: dict[str, str]

    @property
    def query_sample_id(self) -> str:
        return self.payload.get("query_sample_id", "")

    @property
    def neighbor_rank(self) -> int:
        try:
            return int(self.payload.get("neighbor_rank", "0"))
        except ValueError:
            return 0

    @property
    def human_relevance(self) -> str:
        return self.payload.get("human_relevance", "").strip().lower()

    @property
    def is_reviewed(self) -> bool:
        return self.human_relevance in RELEVANCE_VALUES

    def value(self, field_name: str) -> str:
        value = self.payload.get(field_name, "")
        return value if value != "" else "unknown"


@dataclass(frozen=True, slots=True)
class HumanReviewMetrics:
    total_rows: int
    reviewed_rows: int
    unreviewed_rows: int
    invalid_rows: int
    total_queries: int
    reviewed_queries: int
    top_k: int
    accepted_values: tuple[str, ...]
    relevance_counts: dict[str, int]
    accepted_neighbor_rate: float
    uncertain_neighbor_rate: float
    not_similar_neighbor_rate: float
    human_relevance_at_k: float
    accepted_or_uncertain_at_k: float
    breakdowns: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_rows": self.total_rows,
            "reviewed_rows": self.reviewed_rows,
            "unreviewed_rows": self.unreviewed_rows,
            "invalid_rows": self.invalid_rows,
            "total_queries": self.total_queries,
            "reviewed_queries": self.reviewed_queries,
            "top_k": self.top_k,
            "accepted_values": list(self.accepted_values),
            "relevance_counts": self.relevance_counts,
            "accepted_neighbor_rate": self.accepted_neighbor_rate,
            "uncertain_neighbor_rate": self.uncertain_neighbor_rate,
            "not_similar_neighbor_rate": self.not_similar_neighbor_rate,
            "human_relevance_at_k": self.human_relevance_at_k,
            "accepted_or_uncertain_at_k": self.accepted_or_uncertain_at_k,
        }
        if self.breakdowns:
            payload["breakdowns"] = self.breakdowns
        return payload


def load_human_review_records(paths: list[Path]) -> list[HumanReviewRecord]:
    records: list[HumanReviewRecord] = []
    for path in paths:
        records.extend(_load_human_review_file(path))
    return records


def evaluate_human_reviews(
    records: list[HumanReviewRecord],
    top_k: int = 3,
    accepted_values: tuple[str, ...] = DEFAULT_ACCEPTED_VALUES,
    breakdown_fields: tuple[str, ...] = (),
) -> HumanReviewMetrics:
    normalized_accepted_values = tuple(value.strip().lower() for value in accepted_values if value.strip() != "")
    base_metrics = _metrics_for_records(records, top_k, normalized_accepted_values)
    if not breakdown_fields:
        return base_metrics
    return HumanReviewMetrics(
        total_rows=base_metrics.total_rows,
        reviewed_rows=base_metrics.reviewed_rows,
        unreviewed_rows=base_metrics.unreviewed_rows,
        invalid_rows=base_metrics.invalid_rows,
        total_queries=base_metrics.total_queries,
        reviewed_queries=base_metrics.reviewed_queries,
        top_k=base_metrics.top_k,
        accepted_values=base_metrics.accepted_values,
        relevance_counts=base_metrics.relevance_counts,
        accepted_neighbor_rate=base_metrics.accepted_neighbor_rate,
        uncertain_neighbor_rate=base_metrics.uncertain_neighbor_rate,
        not_similar_neighbor_rate=base_metrics.not_similar_neighbor_rate,
        human_relevance_at_k=base_metrics.human_relevance_at_k,
        accepted_or_uncertain_at_k=base_metrics.accepted_or_uncertain_at_k,
        breakdowns=_breakdowns(records, top_k, normalized_accepted_values, breakdown_fields),
    )


def human_review_metrics_to_markdown(metrics: HumanReviewMetrics) -> str:
    lines = [
        "# Human Review Metrics",
        "",
        f"- Reviewed rows: `{metrics.reviewed_rows}` / `{metrics.total_rows}`",
        f"- Reviewed queries: `{metrics.reviewed_queries}` / `{metrics.total_queries}`",
        f"- Top-k: `{metrics.top_k}`",
        f"- Accepted values: `{', '.join(metrics.accepted_values)}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accepted neighbor rate | {metrics.accepted_neighbor_rate:.6f} |",
        f"| Human relevance@{metrics.top_k} | {metrics.human_relevance_at_k:.6f} |",
        f"| Accepted or uncertain@{metrics.top_k} | {metrics.accepted_or_uncertain_at_k:.6f} |",
        f"| Uncertain neighbor rate | {metrics.uncertain_neighbor_rate:.6f} |",
        f"| Not similar neighbor rate | {metrics.not_similar_neighbor_rate:.6f} |",
    ]
    for field_name, rows in metrics.breakdowns.items():
        lines.extend(["", f"## Breakdown: {field_name}", "", "| Value | Reviewed rows | Accepted rate | Human relevance@k |", "|---|---:|---:|---:|"])
        for row in rows:
            lines.append(
                "|"
                + "|".join(
                    [
                        str(row["value"]),
                        str(row["reviewed_rows"]),
                        f"{row['accepted_neighbor_rate']:.6f}",
                        f"{row['human_relevance_at_k']:.6f}",
                    ]
                )
                + "|"
            )
    return "\n".join(lines) + "\n"


def _load_human_review_file(path: Path) -> list[HumanReviewRecord]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".csv":
        return _load_csv_records(path)
    raise ValueError(f"Unsupported human review file: {path}")


def _load_json_records(path: Path) -> list[HumanReviewRecord]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("reviews", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Human review JSON must contain a list or {'reviews': [...]}")
    return [HumanReviewRecord(_normalize_row(row)) for row in rows if isinstance(row, dict)]


def _load_csv_records(path: Path) -> list[HumanReviewRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [HumanReviewRecord(_normalize_row(row)) for row in csv.DictReader(handle)]


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(key): _stringify(value) for key, value in row.items()}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _metrics_for_records(
    records: list[HumanReviewRecord],
    top_k: int,
    accepted_values: tuple[str, ...],
) -> HumanReviewMetrics:
    reviewed_records = [record for record in records if record.is_reviewed]
    invalid_rows = sum(1 for record in records if record.human_relevance != "" and not record.is_reviewed)
    relevance_counts = {value: sum(1 for record in reviewed_records if record.human_relevance == value) for value in RELEVANCE_VALUES}
    reviewed_query_ids = {record.query_sample_id for record in reviewed_records if record.query_sample_id != ""}
    all_query_ids = {record.query_sample_id for record in records if record.query_sample_id != ""}
    accepted_set = set(accepted_values)
    partial_set = set(DEFAULT_PARTIAL_VALUES)
    return HumanReviewMetrics(
        total_rows=len(records),
        reviewed_rows=len(reviewed_records),
        unreviewed_rows=sum(1 for record in records if record.human_relevance == ""),
        invalid_rows=invalid_rows,
        total_queries=len(all_query_ids),
        reviewed_queries=len(reviewed_query_ids),
        top_k=top_k,
        accepted_values=accepted_values,
        relevance_counts=relevance_counts,
        accepted_neighbor_rate=_reviewed_rate(reviewed_records, accepted_set),
        uncertain_neighbor_rate=_reviewed_rate(reviewed_records, {"uncertain"}),
        not_similar_neighbor_rate=_reviewed_rate(reviewed_records, {"not_similar"}),
        human_relevance_at_k=_query_rate(reviewed_records, top_k, accepted_set),
        accepted_or_uncertain_at_k=_query_rate(reviewed_records, top_k, partial_set),
    )


def _breakdowns(
    records: list[HumanReviewRecord],
    top_k: int,
    accepted_values: tuple[str, ...],
    fields: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    return {
        field_name: _breakdown_rows(records, field_name, top_k, accepted_values)
        for field_name in fields
    }


def _breakdown_rows(
    records: list[HumanReviewRecord],
    field_name: str,
    top_k: int,
    accepted_values: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[HumanReviewRecord]] = {}
    for record in records:
        grouped.setdefault(record.value(field_name), []).append(record)
    rows = []
    for value, group_records in grouped.items():
        metrics = _metrics_for_records(group_records, top_k, accepted_values)
        rows.append(
            {
                "value": value,
                "total_rows": metrics.total_rows,
                "reviewed_rows": metrics.reviewed_rows,
                "reviewed_queries": metrics.reviewed_queries,
                "accepted_neighbor_rate": metrics.accepted_neighbor_rate,
                "human_relevance_at_k": metrics.human_relevance_at_k,
                "accepted_or_uncertain_at_k": metrics.accepted_or_uncertain_at_k,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["reviewed_rows"]), str(row["value"])))


def _reviewed_rate(records: list[HumanReviewRecord], accepted_values: set[str]) -> float:
    if not records:
        return 0.0
    return _rate(sum(1 for record in records if record.human_relevance in accepted_values), len(records))


def _query_rate(records: list[HumanReviewRecord], top_k: int, accepted_values: set[str]) -> float:
    grouped: dict[str, list[HumanReviewRecord]] = {}
    for record in records:
        if record.query_sample_id == "":
            continue
        grouped.setdefault(record.query_sample_id, []).append(record)
    if not grouped:
        return 0.0
    accepted_queries = 0
    for query_records in grouped.values():
        if any(record.neighbor_rank <= top_k and record.human_relevance in accepted_values for record in query_records):
            accepted_queries += 1
    return _rate(accepted_queries, len(grouped))


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 6)
