from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from math import sqrt
from typing import Final


EXPECTED_ROWS: Final = 20
EXPECTED_COLS: Final = 7680
ANOMALY_SIGMA_MULTIPLIER: Final = 3.0
MAX_ANOMALY_REGIONS: Final = 8


@dataclass(frozen=True, slots=True)
class CsvShape:
    rows: int
    cols: int
    valid: bool
    message: str


@dataclass(frozen=True, slots=True)
class SignalSummary:
    frame_count: int
    channel_count: int
    sample_count: int
    mean: float
    rms: float
    peak_abs: float
    p99_abs: float
    anomaly_threshold: float
    anomaly_count: int
    anomaly_rate: float
    anomaly_regions: tuple[dict[str, int | float], ...]

    def to_trace_payload(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "channel_count": self.channel_count,
            "sample_count": self.sample_count,
            "mean": round(self.mean, 6),
            "rms": round(self.rms, 6),
            "peak_abs": round(self.peak_abs, 6),
            "p99_abs": round(self.p99_abs, 6),
            "anomaly_threshold": round(self.anomaly_threshold, 6),
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": round(self.anomaly_rate, 6),
            "anomaly_regions": list(self.anomaly_regions),
        }


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


def summarize_csv_signal(content: bytes) -> SignalSummary:
    rows = _numeric_rows(content)
    values = [value for row in rows for value in row]
    if not values:
        return SignalSummary(
            frame_count=0,
            channel_count=0,
            sample_count=0,
            mean=0.0,
            rms=0.0,
            peak_abs=0.0,
            p99_abs=0.0,
            anomaly_threshold=0.0,
            anomaly_count=0,
            anomaly_rate=0.0,
            anomaly_regions=(),
        )

    sample_count = len(values)
    mean = sum(values) / sample_count
    rms = sqrt(sum(value * value for value in values) / sample_count)
    abs_values = sorted(abs(value) for value in values)
    peak_abs = abs_values[-1]
    p99_abs = _percentile(abs_values, 0.99)
    std = sqrt(sum((value - mean) ** 2 for value in values) / sample_count)
    anomaly_threshold = max(p99_abs, abs(mean) + (ANOMALY_SIGMA_MULTIPLIER * std))
    anomaly_regions = _anomaly_regions(rows, anomaly_threshold)
    anomaly_count = sum(1 for value in values if abs(value) >= anomaly_threshold) if anomaly_threshold > 0 else 0
    return SignalSummary(
        frame_count=len(rows),
        channel_count=max((len(row) for row in rows), default=0),
        sample_count=sample_count,
        mean=mean,
        rms=rms,
        peak_abs=peak_abs,
        p99_abs=p99_abs,
        anomaly_threshold=anomaly_threshold,
        anomaly_count=int(anomaly_count),
        anomaly_rate=anomaly_count / sample_count,
        anomaly_regions=tuple(anomaly_regions),
    )


def is_png_upload(filename: str | None, content_type: str | None) -> bool:
    suffix_ok = filename is not None and filename.lower().endswith(".png")
    mime_ok = content_type == "image/png"
    return suffix_ok and mime_ok


def _numeric_rows(content: bytes) -> list[list[float]]:
    text = content.decode("utf-8-sig")
    rows: list[list[float]] = []
    for row in csv.reader(io.StringIO(text)):
        values = [_optional_float(cell) for cell in row]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            rows.append(numeric_values)
    return rows


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * quantile)))
    return sorted_values[index]


def _anomaly_regions(rows: list[list[float]], threshold: float) -> list[dict[str, int | float]]:
    if threshold <= 0:
        return []
    regions: list[dict[str, int | float]] = []
    for frame_index, row in enumerate(rows):
        indices = [index for index, value in enumerate(row) if abs(value) >= threshold]
        if indices:
            regions.append(
                {
                    "frame": frame_index + 1,
                    "start_index": indices[0],
                    "end_index": indices[-1],
                    "count": len(indices),
                    "peak_abs": round(max(abs(row[index]) for index in indices), 6),
                }
            )
        if len(regions) >= MAX_ANOMALY_REGIONS:
            break
    return regions
