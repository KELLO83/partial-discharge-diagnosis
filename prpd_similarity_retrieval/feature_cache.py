from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prpd_similarity_retrieval import FEATURE_SCHEMA_VERSION
from prpd_similarity_retrieval.models import CaseFeatures


CACHE_RECORD_SCHEMA_VERSION = f"{FEATURE_SCHEMA_VERSION}_cache_v1"


def append_feature_cache(path: Path, features: list[CaseFeatures]) -> None:
    if not features:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for feature in features:
            handle.write(json.dumps(_cache_record(feature), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def load_feature_cache(path: Path) -> list[CaseFeatures]:
    if not path.exists():
        return []

    features: list[CaseFeatures] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if stripped_line == "":
                continue
            features.append(_parse_cache_record(stripped_line, line_number))
    return features


def load_feature_cache_map(path: Path) -> dict[str, CaseFeatures]:
    cached_features: dict[str, CaseFeatures] = {}
    for feature in load_feature_cache(path):
        cached_features[feature.sample_id] = feature
    return cached_features


def _cache_record(feature: CaseFeatures) -> dict[str, Any]:
    return {
        "schema_version": CACHE_RECORD_SCHEMA_VERSION,
        "case": feature.to_dict(),
    }


def _parse_cache_record(line: str, line_number: int) -> CaseFeatures:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid feature cache JSON at line {line_number}") from exc

    if payload.get("schema_version") != CACHE_RECORD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature cache schema at line {line_number}: {payload.get('schema_version')}")
    case_payload = payload.get("case")
    if not isinstance(case_payload, dict):
        raise ValueError(f"Invalid feature cache case payload at line {line_number}")
    return CaseFeatures.from_dict(case_payload)
