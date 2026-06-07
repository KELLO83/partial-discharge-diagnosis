from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from service.backend.app.schemas import MetadataInput, TimeSeriesResult, VlmResult


@dataclass(frozen=True, slots=True)
class TimeSeriesToolInput:
    csv_path: Path
    csv_sha256: str


@dataclass(frozen=True, slots=True)
class VlmToolInput:
    image_path: Path
    image_sha256: str
    safe_metadata: MetadataInput
    timeseries_result: TimeSeriesResult | None


class TimeSeriesInferenceAdapter:
    def run(self, tool_input: TimeSeriesToolInput) -> TimeSeriesResult:
        raise NotImplementedError


class VlmInferenceAdapter:
    def run(self, tool_input: VlmToolInput) -> VlmResult:
        raise NotImplementedError
