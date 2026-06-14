from __future__ import annotations

from pathlib import Path

from prpd_similarity_retrieval.hard_split_evaluation import HardSplitFailureCase, HardSplitFailureSample, HardSplitNeighbor
from prpd_similarity_retrieval.review_artifact import hard_split_failure_review_to_html


def test_hard_split_failure_review_renders_waveform_and_cases(tmp_path: Path) -> None:
    query_csv = tmp_path / "query.csv"
    neighbor_csv = tmp_path / "neighbor.csv"
    query_csv.write_text("0\n1\n0\n-1\n", encoding="utf-8")
    neighbor_csv.write_text("0\n0.5\n0\n-0.5\n", encoding="utf-8")
    sample = HardSplitFailureSample(
        retrieval_mode="feature_retrieval",
        split_field="equipment_name",
        holdout_values=("CNCV-W",),
        train_count=10,
        query_count=2,
        inspected=1,
        top_k=1,
        failures=[
            HardSplitFailureCase(
                query_sample_id="query-sample",
                query_label_id=1,
                query_label_name="noise",
                query_image_path=str(tmp_path / "missing-query.png"),
                query_timeseries_path=str(query_csv),
                query_metadata={"equipment_name": "CNCV-W", "sensor_type": "HFCT"},
                neighbors=[
                    HardSplitNeighbor(
                        rank=1,
                        sample_id="neighbor-sample",
                        label_id=0,
                        label_name="normal",
                        score=0.91,
                        image_path=str(tmp_path / "missing-neighbor.png"),
                        timeseries_path=str(neighbor_csv),
                        metadata={"equipment_name": "ACSR-OC", "sensor_type": "HFCT"},
                    )
                ],
            )
        ],
    )

    html = hard_split_failure_review_to_html(sample, tmp_path / "review.html")

    assert "Hard Split Failure Review" in html
    assert "query-sample" in html
    assert "neighbor-sample" in html
    assert "0.910000" in html
    assert "<polyline" in html
    assert "No image" in html
    assert "Human relevance" in html
    assert 'value="similar"' in html
    assert 'value="not_similar"' in html
    assert "downloadReviews('csv')" in html
    assert 'data-query-sample-id="query-sample"' in html
    assert 'data-neighbor-sample-id="neighbor-sample"' in html
    assert 'data-query-equipment-name="CNCV-W"' in html
    assert 'data-neighbor-equipment-name="ACSR-OC"' in html
    assert '"query_equipment_name"' in html
