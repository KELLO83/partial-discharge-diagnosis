from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Final


EXPECTED_ROWS: Final = 20
EXPECTED_COLS: Final = 7680


@dataclass(frozen=True, slots=True)
class CsvShape:
    rows: int
    cols: int
    valid: bool
    message: str


def inspect_csv_shape(content: bytes) -> CsvShape:
    text = content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows = 0
    cols: int | None = None
    for row in reader:
        if not row:
            continue
        rows += 1
        width = len(row)
        if cols is None:
            cols = width
        if width != cols:
            return CsvShape(rows=rows, cols=width, valid=False, message="csv rows must have a consistent width")
    actual_cols = cols if cols is not None else 0
    valid = rows == EXPECTED_ROWS and actual_cols == EXPECTED_COLS
    message = "ok" if valid else f"timeseries_csv must have shape ({EXPECTED_ROWS}, {EXPECTED_COLS})."
    return CsvShape(rows=rows, cols=actual_cols, valid=valid, message=message)


def is_png_upload(filename: str | None, content_type: str | None) -> bool:
    suffix_ok = filename is not None and filename.lower().endswith(".png")
    mime_ok = content_type == "image/png"
    return suffix_ok and mime_ok
