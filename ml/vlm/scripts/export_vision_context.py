from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from ml.vlm.scripts.build_instruction_dataset import resolve_manifest_file_path
from service.backend.app.application.contracts import VisionToolInput
from service.backend.app.models.checkpoint_adapters import CheckpointVisionInferenceAdapter
from service.backend.app.models.model_artifacts import ModelArtifactRegistry


FIELDNAMES: tuple[str, ...] = (
    "sample_id",
    "vision_model_name",
    "vision_pred_label_id",
    "vision_confidence",
    "vision_prob_0",
    "vision_prob_1",
    "vision_prob_2",
    "vision_prob_3",
    "vision_prob_4",
)


def export_vision_context(
    manifest_path: Path,
    output_path: Path,
    sample_size: int | None,
    model_artifact_root: Path,
) -> int:
    rows = _read_manifest(manifest_path)
    selected_rows = _balanced_sample(rows, sample_size)
    adapter = _vision_adapter(model_artifact_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(_context_row(manifest_path, row, adapter))
    return len(selected_rows)


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _balanced_sample(rows: list[dict[str, str]], sample_size: int | None) -> list[dict[str, str]]:
    if sample_size is None or sample_size >= len(rows):
        return rows
    buckets: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(int(row.get("label_id", "0")), []).append(row)
    selected: list[dict[str, str]] = []
    cursor = 0
    label_ids = sorted(buckets)
    while len(selected) < sample_size:
        added = False
        for label_id in label_ids:
            bucket = buckets[label_id]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                added = True
                if len(selected) >= sample_size:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _vision_adapter(model_artifact_root: Path) -> CheckpointVisionInferenceAdapter:
    record = ModelArtifactRegistry(model_artifact_root).get("vision")
    if not record.ready:
        raise RuntimeError(f"vision model artifact is not ready: {record.error}")
    return CheckpointVisionInferenceAdapter(record)


def _context_row(
    manifest_path: Path,
    row: dict[str, str],
    adapter: CheckpointVisionInferenceAdapter,
) -> dict[str, object]:
    image_path = resolve_manifest_file_path(manifest_path, row["image_path"])
    result = adapter.run(VisionToolInput(image_path=image_path, image_sha256="context-export"))
    probabilities = result.probabilities or {}
    output: dict[str, object] = {
        "sample_id": row["sample_id"],
        "vision_model_name": result.model_name,
        "vision_pred_label_id": result.label_id,
        "vision_confidence": result.confidence,
    }
    output.update({f"vision_prob_{label_id}": probabilities.get(str(label_id), "") for label_id in range(5)})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model-artifact-root", type=Path, default=Path("artifacts/models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = export_vision_context(
        manifest_path=args.manifest,
        output_path=args.output,
        sample_size=args.sample_size,
        model_artifact_root=args.model_artifact_root,
    )
    print(json.dumps({"rows_written": rows_written, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
