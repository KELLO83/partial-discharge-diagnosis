"""CSV experiment logger."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_experiment_result(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }
    if not output_path.exists():
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(serializable.keys()))
            writer.writeheader()
            writer.writerow(serializable)
        return

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = list(reader.fieldnames or [])
        old_rows = []
        for old_row in reader:
            old_row.pop(None, None)
            old_rows.append(old_row)

    fieldnames = old_fieldnames[:]
    for key in serializable:
        if key not in fieldnames:
            fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(serializable)
