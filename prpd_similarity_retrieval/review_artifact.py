from __future__ import annotations

import os
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np

from prpd_similarity_retrieval.hard_split_evaluation import HardSplitFailureCase, HardSplitFailureSample, HardSplitNeighbor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAVEFORM_WIDTH = 320
WAVEFORM_HEIGHT = 96
WAVEFORM_POINTS = 220


def hard_split_failure_review_to_html(
    sample: HardSplitFailureSample,
    output_path: Path | None = None,
) -> str:
    output_dir = output_path.parent.resolve() if output_path is not None else Path.cwd()
    title = f"Hard Split Review - {', '.join(sample.holdout_values)}"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            _style_block(),
            _script_block(),
            "</head>",
            "<body>",
            _header_html(sample),
            "<main>",
            "".join(_failure_html(failure, output_dir, failure_index) for failure_index, failure in enumerate(sample.failures)),
            _empty_state_html(sample),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _header_html(sample: HardSplitFailureSample) -> str:
    return f"""
<header class="page-head">
  <div>
    <p class="kicker">PRPD Similarity Retrieval</p>
    <h1>Hard Split Failure Review</h1>
  </div>
  <dl class="run-meta">
    <div><dt>Mode</dt><dd>{escape(sample.retrieval_mode)}</dd></div>
    <div><dt>Split</dt><dd>{escape(sample.split_field)}</dd></div>
    <div><dt>Holdout</dt><dd>{escape(", ".join(sample.holdout_values))}</dd></div>
    <div><dt>Train / Query</dt><dd>{sample.train_count:,} / {sample.query_count:,}</dd></div>
    <div><dt>Inspected</dt><dd>{sample.inspected:,}</dd></div>
    <div><dt>Top-k</dt><dd>{sample.top_k}</dd></div>
  </dl>
  <div class="review-actions" aria-label="review export actions">
    <button type="button" onclick="downloadReviews('csv')">CSV</button>
    <button type="button" onclick="downloadReviews('json')">JSON</button>
    <output id="review-progress">0 reviewed</output>
  </div>
</header>
"""


def _failure_html(failure: HardSplitFailureCase, output_dir: Path, failure_index: int) -> str:
    neighbors = "".join(
        _neighbor_panel_html(failure, neighbor, output_dir, failure_index, neighbor_index)
        for neighbor_index, neighbor in enumerate(failure.neighbors)
    )
    return f"""
<section class="failure">
  <div class="failure-head">
    <div>
      <p class="kicker">Query</p>
      <h2>{escape(failure.query_sample_id)}</h2>
    </div>
    <div class="label-chip danger">{escape(_label_text(failure.query_label_id, failure.query_label_name))}</div>
  </div>
  <div class="comparison-grid">
    {_query_panel_html(failure, output_dir)}
    <div class="neighbor-stack">
      {neighbors}
    </div>
  </div>
</section>
"""


def _query_panel_html(failure: HardSplitFailureCase, output_dir: Path) -> str:
    return _case_panel_html(
        panel_class="query-panel",
        title="Holdout Query",
        sample_id=failure.query_sample_id,
        label_id=failure.query_label_id,
        label_name=failure.query_label_name,
        score=None,
        image_path=failure.query_image_path,
        timeseries_path=failure.query_timeseries_path,
        metadata=failure.query_metadata,
        output_dir=output_dir,
    )


def _neighbor_panel_html(
    failure: HardSplitFailureCase,
    neighbor: HardSplitNeighbor,
    output_dir: Path,
    failure_index: int,
    neighbor_index: int,
) -> str:
    return _case_panel_html(
        panel_class="neighbor-panel",
        title=f"Rank {neighbor.rank}",
        sample_id=neighbor.sample_id,
        label_id=neighbor.label_id,
        label_name=neighbor.label_name,
        score=neighbor.score,
        image_path=neighbor.image_path,
        timeseries_path=neighbor.timeseries_path,
        metadata=neighbor.metadata,
        output_dir=output_dir,
        data_attributes=_review_data_attributes(failure, neighbor),
        review_controls=_review_controls(failure_index, neighbor_index),
    )


def _case_panel_html(
    panel_class: str,
    title: str,
    sample_id: str,
    label_id: int | None,
    label_name: str,
    score: float | None,
    image_path: str | None,
    timeseries_path: str | None,
    metadata: dict[str, str],
    output_dir: Path,
    data_attributes: str = "",
    review_controls: str = "",
) -> str:
    score_html = "" if score is None else f'<span class="score">{score:.6f}</span>'
    return f"""
<article class="case-panel {panel_class}"{data_attributes}>
  <div class="case-topline">
    <span>{escape(title)}</span>
    {score_html}
  </div>
  <h3>{escape(sample_id)}</h3>
  <div class="label-chip">{escape(_label_text(label_id, label_name))}</div>
  <div class="media-grid">
    <figure>
      {_image_html(image_path, output_dir)}
      <figcaption>PRPD</figcaption>
    </figure>
    <figure>
      {_waveform_svg(timeseries_path)}
      <figcaption>Time-series</figcaption>
    </figure>
  </div>
  {_metadata_html(metadata)}
  {review_controls}
</article>
"""


def _image_html(raw_path: str | None, output_dir: Path) -> str:
    path = _resolve_path(raw_path)
    if path is None or not path.exists():
        return '<div class="missing-media">No image</div>'
    return f'<img src="{_asset_src(path, output_dir)}" alt="">'


def _waveform_svg(raw_path: str | None) -> str:
    values = _read_numeric_values(raw_path)
    if values.size == 0:
        return '<div class="missing-media">No signal</div>'
    y_values = _normalized_waveform(values)
    x_step = WAVEFORM_WIDTH / max(y_values.size - 1, 1)
    points = " ".join(f"{index * x_step:.2f},{value:.2f}" for index, value in enumerate(y_values))
    return f"""
<svg class="waveform" viewBox="0 0 {WAVEFORM_WIDTH} {WAVEFORM_HEIGHT}" role="img" aria-label="time-series waveform">
  <line x1="0" y1="{WAVEFORM_HEIGHT / 2:.1f}" x2="{WAVEFORM_WIDTH}" y2="{WAVEFORM_HEIGHT / 2:.1f}" />
  <polyline points="{points}" />
</svg>
"""


def _metadata_html(metadata: dict[str, str]) -> str:
    fields = ("equipment_name", "sensor_type", "insulator_type", "clearance_distance", "equipment_rated_voltage")
    chips = [
        f"<span>{escape(field)}: {escape(value)}</span>"
        for field in fields
        if (value := metadata.get(field, "")) != ""
    ]
    return f'<div class="metadata-row">{"".join(chips)}</div>' if chips else ""


def _review_data_attributes(failure: HardSplitFailureCase, neighbor: HardSplitNeighbor) -> str:
    attributes = {
        "query-sample-id": failure.query_sample_id,
        "query-label-id": "" if failure.query_label_id is None else str(failure.query_label_id),
        "query-label-name": failure.query_label_name,
        "neighbor-rank": str(neighbor.rank),
        "neighbor-sample-id": neighbor.sample_id,
        "neighbor-label-id": "" if neighbor.label_id is None else str(neighbor.label_id),
        "neighbor-label-name": neighbor.label_name,
        "query-equipment-name": failure.query_metadata.get("equipment_name", ""),
        "query-sensor-type": failure.query_metadata.get("sensor_type", ""),
        "neighbor-equipment-name": neighbor.metadata.get("equipment_name", ""),
        "neighbor-sensor-type": neighbor.metadata.get("sensor_type", ""),
        "similarity-score": f"{neighbor.score:.6f}",
    }
    return " " + " ".join(f'data-{key}="{escape(value, quote=True)}"' for key, value in attributes.items())


def _review_controls(failure_index: int, neighbor_index: int) -> str:
    control_name = f"review-{failure_index}-{neighbor_index}"
    return f"""
<div class="review-box">
  <div class="review-heading">Human relevance</div>
  <div class="review-options" role="radiogroup" aria-label="Human relevance">
    <label><input type="radio" name="{control_name}" value="similar"> 유사</label>
    <label><input type="radio" name="{control_name}" value="uncertain"> 애매</label>
    <label><input type="radio" name="{control_name}" value="not_similar"> 비유사</label>
  </div>
  <textarea data-review-note rows="2" placeholder="판정 메모"></textarea>
</div>
"""


def _empty_state_html(sample: HardSplitFailureSample) -> str:
    if sample.failures:
        return ""
    return f"""
<section class="empty-state">
  <h2>No failures found</h2>
  <p>{escape(", ".join(sample.holdout_values))} produced no top-k label misses in the inspected range.</p>
</section>
"""


def _resolve_path(raw_path: str | None) -> Path | None:
    if raw_path is None or raw_path == "":
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return path


def _asset_src(path: Path, output_dir: Path) -> str:
    relative_path = os.path.relpath(path.resolve(), output_dir)
    return quote(Path(relative_path).as_posix(), safe="/:.%#?&=-_")


def _read_numeric_values(raw_path: str | None) -> np.ndarray:
    path = _resolve_path(raw_path)
    if path is None or not path.exists():
        return np.asarray([], dtype=np.float32)
    try:
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except (OSError, ValueError):
        try:
            values = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        except (OSError, ValueError):
            return np.asarray([], dtype=np.float32)
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_values = flattened[np.isfinite(flattened)]
    if finite_values.size <= WAVEFORM_POINTS:
        return finite_values
    sample_positions = np.linspace(0, finite_values.size - 1, WAVEFORM_POINTS).astype(np.int32)
    return finite_values[sample_positions]


def _normalized_waveform(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.mean(values))
    max_abs = float(np.max(np.abs(centered)))
    if max_abs <= 0.0:
        return np.full(values.size, WAVEFORM_HEIGHT / 2, dtype=np.float32)
    normalized = centered / max_abs
    return ((0.5 - (normalized * 0.42)) * WAVEFORM_HEIGHT).astype(np.float32)


def _label_text(label_id: int | None, label_name: str) -> str:
    if label_id is None:
        return label_name or "unknown"
    if label_name:
        return f"{label_name} ({label_id})"
    return str(label_id)


def _style_block() -> str:
    return """
<style>
:root {
  --paper: #f2f5f4;
  --panel: #ffffff;
  --ink: #17201f;
  --muted: #65716f;
  --line: #cfd8d5;
  --strong: #0f766e;
  --warn: #b42318;
  --amber: #a15c07;
  --wave: #1d4ed8;
  --action: #1f4d5a;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Aptos", "Segoe UI", sans-serif;
  letter-spacing: 0;
}

.page-head {
  display: grid;
  grid-template-columns: minmax(260px, 1.05fr) minmax(320px, .95fr) minmax(220px, .45fr);
  gap: 28px;
  padding: 32px clamp(18px, 4vw, 56px);
  border-bottom: 1px solid var(--line);
  background: #10211f;
  color: #f7fbfa;
}

.kicker {
  margin: 0 0 8px;
  color: #83d5ca;
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

h1, h2, h3 { margin: 0; letter-spacing: 0; }
h1 { font-size: clamp(30px, 5vw, 58px); line-height: 1; }
h2 { font-size: 20px; line-height: 1.2; overflow-wrap: anywhere; }
h3 { font-size: 14px; line-height: 1.25; overflow-wrap: anywhere; }

.run-meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
}

.run-meta div {
  padding: 10px 12px;
  border: 1px solid rgba(255,255,255,.22);
  background: rgba(255,255,255,.06);
}

dt { color: #addbd5; font-size: 11px; text-transform: uppercase; }
dd { margin: 2px 0 0; font-weight: 750; overflow-wrap: anywhere; }

.review-actions {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-content: start;
  gap: 10px;
}

.review-actions button {
  min-height: 38px;
  border: 1px solid rgba(255,255,255,.26);
  background: #f7fbfa;
  color: #10211f;
  font-weight: 850;
  cursor: pointer;
}

.review-actions button:hover { background: #c7f1ea; }

#review-progress {
  grid-column: 1 / -1;
  color: #addbd5;
  font-size: 12px;
  font-weight: 800;
}

main {
  display: grid;
  gap: 22px;
  padding: 24px clamp(14px, 3vw, 44px) 48px;
}

.failure {
  border: 1px solid var(--line);
  border-left: 5px solid var(--warn);
  background: var(--panel);
}

.failure-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 20px;
  border-bottom: 1px solid var(--line);
}

.comparison-grid {
  display: grid;
  grid-template-columns: minmax(280px, .8fr) minmax(420px, 1.4fr);
  gap: 1px;
  background: var(--line);
}

.neighbor-stack {
  display: grid;
  gap: 1px;
}

.case-panel {
  min-width: 0;
  padding: 16px;
  background: var(--panel);
}

.query-panel { background: #fbfcfc; }

.case-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 9px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

.score { color: var(--amber); font-variant-numeric: tabular-nums; }

.label-chip {
  display: inline-block;
  margin: 10px 0 14px;
  color: var(--strong);
  font-size: 13px;
  font-weight: 800;
}

.label-chip.danger { color: var(--warn); margin: 0; white-space: nowrap; }

.media-grid {
  display: grid;
  grid-template-columns: minmax(160px, 1fr) minmax(220px, 1.2fr);
  gap: 12px;
  align-items: stretch;
}

figure { margin: 0; }

figcaption {
  margin-top: 6px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}

img, .missing-media, .waveform {
  width: 100%;
  aspect-ratio: 4 / 3;
  border: 1px solid var(--line);
  background: #eef3f1;
}

img { object-fit: contain; display: block; }

.missing-media {
  display: grid;
  place-items: center;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.waveform { height: auto; }
.waveform line { stroke: #c8d1ce; stroke-width: 1; }
.waveform polyline { fill: none; stroke: var(--wave); stroke-width: 2.2; }

.metadata-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin-top: 14px;
  color: var(--muted);
  font-size: 12px;
}

.metadata-row span { overflow-wrap: anywhere; }

.review-box {
  margin-top: 14px;
  padding: 12px;
  border: 1px solid var(--line);
  background: #f7faf9;
}

.review-heading {
  margin-bottom: 8px;
  color: var(--action);
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
}

.review-options {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
}

.review-options label {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 34px;
  padding: 6px 8px;
  border: 1px solid var(--line);
  background: #ffffff;
  color: var(--ink);
  font-size: 12px;
  font-weight: 800;
  cursor: pointer;
}

.review-options input { margin: 0 6px 0 0; }

.review-box textarea {
  width: 100%;
  min-height: 54px;
  margin-top: 8px;
  resize: vertical;
  border: 1px solid var(--line);
  padding: 8px;
  color: var(--ink);
  font: inherit;
  font-size: 12px;
}

.empty-state {
  padding: 32px;
  border: 1px solid var(--line);
  background: var(--panel);
}

@media (max-width: 980px) {
  .page-head, .comparison-grid, .media-grid, .review-options { grid-template-columns: 1fr; }
  .failure-head { align-items: flex-start; flex-direction: column; }
}
</style>
"""


def _script_block() -> str:
    return """
<script>
const reviewColumns = [
  "query_sample_id",
  "query_label_id",
  "query_label_name",
  "neighbor_rank",
  "neighbor_sample_id",
  "neighbor_label_id",
  "neighbor_label_name",
  "query_equipment_name",
  "query_sensor_type",
  "neighbor_equipment_name",
  "neighbor_sensor_type",
  "similarity_score",
  "human_relevance",
  "review_note"
];

function collectReviews() {
  return Array.from(document.querySelectorAll(".neighbor-panel")).map((panel) => {
    const checked = panel.querySelector("input[type='radio']:checked");
    const note = panel.querySelector("[data-review-note]");
    return {
      query_sample_id: panel.dataset.querySampleId || "",
      query_label_id: panel.dataset.queryLabelId || "",
      query_label_name: panel.dataset.queryLabelName || "",
      neighbor_rank: panel.dataset.neighborRank || "",
      neighbor_sample_id: panel.dataset.neighborSampleId || "",
      neighbor_label_id: panel.dataset.neighborLabelId || "",
      neighbor_label_name: panel.dataset.neighborLabelName || "",
      query_equipment_name: panel.dataset.queryEquipmentName || "",
      query_sensor_type: panel.dataset.querySensorType || "",
      neighbor_equipment_name: panel.dataset.neighborEquipmentName || "",
      neighbor_sensor_type: panel.dataset.neighborSensorType || "",
      similarity_score: panel.dataset.similarityScore || "",
      human_relevance: checked ? checked.value : "",
      review_note: note ? note.value.trim() : ""
    };
  });
}

function csvCell(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

function reviewsAsCsv(rows) {
  return [
    reviewColumns.map(csvCell).join(","),
    ...rows.map((row) => reviewColumns.map((column) => csvCell(row[column] || "")).join(","))
  ].join("\\n");
}

function downloadText(filename, text, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

function downloadReviews(format) {
  const rows = collectReviews();
  if (format === "json") {
    downloadText("hard_split_human_reviews.json", JSON.stringify(rows, null, 2), "application/json;charset=utf-8");
    return;
  }
  downloadText("hard_split_human_reviews.csv", reviewsAsCsv(rows), "text/csv;charset=utf-8");
}

function updateReviewProgress() {
  const rows = collectReviews();
  const reviewed = rows.filter((row) => row.human_relevance !== "").length;
  const output = document.getElementById("review-progress");
  if (output) {
    output.value = `${reviewed} / ${rows.length} reviewed`;
    output.textContent = output.value;
  }
}

document.addEventListener("change", updateReviewProgress);
document.addEventListener("input", updateReviewProgress);
window.addEventListener("DOMContentLoaded", updateReviewProgress);
</script>
"""
