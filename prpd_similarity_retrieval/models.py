from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CaseRecord:
    sample_id: str
    label_id: int | None
    label_name: str
    image_path: Path | None
    timeseries_path: Path | None
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class CaseFeatures:
    sample_id: str
    label_id: int | None
    label_name: str
    image_path: str | None
    timeseries_path: str | None
    metadata: dict[str, str]
    image_vector: list[float] | None
    timeseries_vector: list[float] | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseFeatures:
        return cls(
            sample_id=str(payload["sample_id"]),
            label_id=_optional_int(payload.get("label_id")),
            label_name=str(payload.get("label_name", "")),
            image_path=_optional_str(payload.get("image_path")),
            timeseries_path=_optional_str(payload.get("timeseries_path")),
            metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
            image_vector=_optional_float_list(payload.get("image_vector")),
            timeseries_vector=_optional_float_list(payload.get("timeseries_vector")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "image_path": self.image_path,
            "timeseries_path": self.timeseries_path,
            "metadata": self.metadata,
            "image_vector": self.image_vector,
            "timeseries_vector": self.timeseries_vector,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    case: CaseFeatures
    score: float
    image_score: float | None
    timeseries_score: float | None
    metadata_score: float | None
    label_score: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.case.sample_id,
            "label_id": self.case.label_id,
            "label_name": self.case.label_name,
            "similarity": round(self.score, 6),
            "component_scores": {
                "prpd_image": _rounded_or_none(self.image_score),
                "timeseries": _rounded_or_none(self.timeseries_score),
                "metadata": _rounded_or_none(self.metadata_score),
                "label": _rounded_or_none(self.label_score),
            },
            "reason": self.reason,
            "image_path": self.case.image_path,
            "timeseries_path": self.case.timeseries_path,
            "metadata": self.case.metadata,
        }


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    return [float(item) for item in value]


def _rounded_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 6)

