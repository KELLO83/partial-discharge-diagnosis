from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from service.backend.app.schemas import MetadataInput, RagResult, SimilarCaseResult, TimeSeriesResult, VisionResult, VlmResult


@dataclass(frozen=True, slots=True)
class TimeSeriesToolInput:
    csv_path: Path
    csv_sha256: str


@dataclass(frozen=True, slots=True)
class VisionToolInput:
    image_path: Path
    image_sha256: str


@dataclass(frozen=True, slots=True)
class SimilarCaseToolInput:
    safe_metadata: MetadataInput | None
    route: str
    timeseries_result: TimeSeriesResult | None
    vision_result: VisionResult | None
    image_path: Path | None = None
    timeseries_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RagToolInput:
    safe_metadata: MetadataInput | None
    route: str
    timeseries_result: TimeSeriesResult | None
    vision_result: VisionResult | None
    similar_case_result: SimilarCaseResult | None


@dataclass(frozen=True, slots=True)
class VlmToolInput:
    image_path: Path
    image_sha256: str
    safe_metadata: MetadataInput
    timeseries_result: TimeSeriesResult | None
    vision_result: VisionResult | None
    rag_result: RagResult | None


class TimeSeriesInferenceAdapter:
    def run(self, tool_input: TimeSeriesToolInput) -> TimeSeriesResult:
        raise NotImplementedError


class VisionInferenceAdapter:
    def run(self, tool_input: VisionToolInput) -> VisionResult:
        raise NotImplementedError


class RagRetrievalAdapter:
    def run(self, tool_input: RagToolInput) -> RagResult:
        raise NotImplementedError


class SimilarCaseRetrievalAdapter:
    def run(self, tool_input: SimilarCaseToolInput) -> SimilarCaseResult:
        raise NotImplementedError


class VlmInferenceAdapter:
    def run(self, tool_input: VlmToolInput) -> VlmResult:
        raise NotImplementedError
