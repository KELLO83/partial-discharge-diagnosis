from __future__ import annotations

import json
from pathlib import Path

from prpd_similarity_retrieval.human_review import (
    evaluate_human_reviews,
    human_review_metrics_to_markdown,
    load_human_review_records,
)


def test_human_review_metrics_count_rows_queries_and_rates(tmp_path: Path) -> None:
    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(
        "\n".join(
            [
                "query_sample_id,neighbor_rank,human_relevance,query_equipment_name",
                "q1,1,not_similar,CNCV-W",
                "q1,2,similar,CNCV-W",
                "q2,1,uncertain,단상 유입변압기",
                "q3,1,bad_value,CNCV-W",
                "q4,1,,CNCV-W",
            ]
        ),
        encoding="utf-8",
    )

    metrics = evaluate_human_reviews(
        load_human_review_records([csv_path]),
        top_k=2,
        breakdown_fields=("query_equipment_name",),
    )

    assert metrics.total_rows == 5
    assert metrics.reviewed_rows == 3
    assert metrics.unreviewed_rows == 1
    assert metrics.invalid_rows == 1
    assert metrics.total_queries == 4
    assert metrics.reviewed_queries == 2
    assert metrics.accepted_neighbor_rate == 0.333333
    assert metrics.human_relevance_at_k == 0.5
    assert metrics.accepted_or_uncertain_at_k == 1.0
    assert metrics.breakdowns["query_equipment_name"][0]["value"] == "CNCV-W"


def test_human_review_metrics_load_json_reviews_object(tmp_path: Path) -> None:
    json_path = tmp_path / "reviews.json"
    json_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {"query_sample_id": "q1", "neighbor_rank": "1", "human_relevance": "similar"},
                    {"query_sample_id": "q2", "neighbor_rank": "1", "human_relevance": "not_similar"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics = evaluate_human_reviews(load_human_review_records([json_path]), top_k=1)

    assert metrics.reviewed_rows == 2
    assert metrics.human_relevance_at_k == 0.5


def test_human_review_metrics_markdown_contains_breakdown(tmp_path: Path) -> None:
    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(
        "\n".join(
            [
                "query_sample_id,neighbor_rank,human_relevance,query_equipment_name",
                "q1,1,similar,CNCV-W",
            ]
        ),
        encoding="utf-8",
    )
    metrics = evaluate_human_reviews(load_human_review_records([csv_path]), breakdown_fields=("query_equipment_name",))

    markdown = human_review_metrics_to_markdown(metrics)

    assert "Human Review Metrics" in markdown
    assert "Breakdown: query_equipment_name" in markdown
    assert "CNCV-W" in markdown
