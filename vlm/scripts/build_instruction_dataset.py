from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from vlm.src.prompts import build_prompt_text, build_target_json
from vlm.src.schema import DatasetBuildSummary, ManifestVlmRow, TimeSeriesContext


def build_instruction_dataset(
    manifest_path: Path,
    output_path: Path,
    sample_size: int | None,
    ts_context_path: Path | None,
) -> DatasetBuildSummary:
    contexts = _load_contexts(ts_context_path)
    rows = _load_manifest_rows(manifest_path)
    selected_rows = rows[:sample_size] if sample_size is not None else rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected_rows:
            image_path = Path(row.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"VLM image not found for sample {row.sample_id}: {image_path}")
            context = contexts.get(row.sample_id, TimeSeriesContext.unavailable(row.sample_id))
            prompt_text = build_prompt_text(row, context)
            target_json = build_target_json(row)
            record = {
                "sample_id": row.sample_id,
                "split": row.split,
                "label_id": row.label_id,
                "images": [row.image_path],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": row.image_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                    {"role": "assistant", "content": target_json},
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
    return DatasetBuildSummary(rows_written=rows_written, output_path=output_path)


def _load_manifest_rows(manifest_path: Path) -> list[ManifestVlmRow]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [ManifestVlmRow.from_mapping(row) for row in reader]


def _load_contexts(ts_context_path: Path | None) -> dict[str, TimeSeriesContext]:
    if ts_context_path is None:
        return {}
    with ts_context_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        contexts = [TimeSeriesContext.from_mapping(row) for row in reader]
    return {context.sample_id: context for context in contexts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--ts-context", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_instruction_dataset(
        manifest_path=args.manifest,
        output_path=args.output,
        sample_size=args.sample_size,
        ts_context_path=args.ts_context,
    )
    print(json.dumps({"rows_written": summary.rows_written, "output_path": str(summary.output_path)}))


if __name__ == "__main__":
    main()
