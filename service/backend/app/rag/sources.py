from __future__ import annotations

import csv
from pathlib import Path

from service.backend.app.rag.documents import RagSourceDocument
from service.backend.app.domain.similar_cases import DEFAULT_DATA_ROOT, DEFAULT_MANIFEST_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_KNOWLEDGE_ROOT = PROJECT_ROOT / "service" / "backend" / "knowledge_sources"


def load_markdown_documents(root: Path = DEFAULT_KNOWLEDGE_ROOT) -> list[RagSourceDocument]:
    documents: list[RagSourceDocument] = []
    for source_type in ("rulebook", "sop"):
        source_root = root / source_type
        if not source_root.exists():
            continue
        for path in sorted(source_root.glob("*.md")):
            documents.append(_markdown_document(path, source_type))
    return documents


def load_dataset_case_documents(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    limit: int | None = None,
) -> list[RagSourceDocument]:
    if not manifest_path.exists():
        return []
    documents: list[RagSourceDocument] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            documents.append(_dataset_case_document(row, data_root))
            if limit is not None and len(documents) >= limit:
                break
    return documents


def _markdown_document(path: Path, source_type: str) -> RagSourceDocument:
    text = path.read_text(encoding="utf-8").strip()
    metadata = _frontmatter(text)
    body = _strip_frontmatter(text)
    label_id = _optional_int(metadata.get("label_id"))
    return RagSourceDocument(
        document_key=f"{source_type}:{path.stem}",
        source_type=source_type,
        title=metadata.get("title") or path.stem.replace("_", " "),
        text=body,
        source_path=str(path),
        label_id=label_id,
        sensor_type=metadata.get("sensor_type"),
        equipment_type=metadata.get("equipment_type"),
        insulator_type=metadata.get("insulator_type"),
        source_ref=metadata.get("source_ref") or f"{source_type}:{path.name}",
        metadata=metadata,
    )


def _dataset_case_document(row: dict[str, str], data_root: Path) -> RagSourceDocument:
    sample_id = row["sample_id"]
    metadata = {
        "sample_id": sample_id,
        "label_id": row.get("label_id", ""),
        "label_name": row.get("label_name", ""),
        "equipment_name": row.get("equipment_name", ""),
        "equipment_manufacturer": row.get("equipment_manufacturer", ""),
        "equipment_type": row.get("equipment_type", ""),
        "equipment_id": row.get("equipment_id", ""),
        "sensor_type": row.get("sensor_type", ""),
        "insulator_type": row.get("insulator_type", ""),
        "insulator_name": row.get("insulator_name", ""),
        "clearance_distance": row.get("clearance_distance", ""),
        "equipment_rated_voltage": row.get("equipment_rated_voltage", ""),
        "equipment_rated_current": row.get("equipment_rated_current", ""),
        "recording_time": row.get("recording_time", ""),
        "recording_time_length": row.get("recording_time_length", ""),
        "power_supply_voltage_type": row.get("power_supply_voltage_type", ""),
        "power_supply_frequency": row.get("power_supply_frequency", ""),
        "temperature": row.get("temperature", ""),
        "humidity": row.get("humidity", ""),
        "iec_standard": row.get("IEC_standard", ""),
        "defect_nums": row.get("defect_nums", ""),
        "defect_details": row.get("defect_details", ""),
        "max_discharge_value": row.get("max_discharge_value", ""),
        "image_path": _relative_path(row.get("image_path", ""), data_root),
        "timeseries_path": _relative_path(row.get("timeseries_path", ""), data_root),
        "json_path": _relative_path(row.get("json_path", ""), data_root),
    }
    text = "\n".join(
        [
            f"sample_id={sample_id}",
            f"label_id={metadata['label_id']}",
            f"label={metadata['label_name']}",
            f"equipment={metadata['equipment_name']}",
            f"manufacturer={metadata['equipment_manufacturer']}",
            f"equipment_type={metadata['equipment_type']}",
            f"equipment_id={metadata['equipment_id']}",
            f"sensor={metadata['sensor_type']}",
            f"insulator={metadata['insulator_type']}",
            f"insulator_name={metadata['insulator_name']}",
            f"voltage={metadata['equipment_rated_voltage']}",
            f"current={metadata['equipment_rated_current']}",
            f"clearance={metadata['clearance_distance']}",
            f"recording_time={metadata['recording_time']}",
            f"duration={metadata['recording_time_length']}",
            f"power_supply={metadata['power_supply_voltage_type']}",
            f"power_frequency={metadata['power_supply_frequency']}",
            f"temperature={metadata['temperature']}",
            f"humidity={metadata['humidity']}",
            f"iec_standard={metadata['iec_standard']}",
            f"defect_nums={metadata['defect_nums']}",
            f"defect_details={metadata['defect_details']}",
            f"max_discharge={metadata['max_discharge_value']}",
            "pattern_summary=데이터셋 과거 사례 요약. 현재 진단의 설비, 센서, 절연, 라벨 후보와 비교하기 위한 근거입니다.",
        ]
    )
    return RagSourceDocument(
        document_key=f"dataset_case:{sample_id}",
        source_type="dataset_case",
        title=f"데이터셋 사례 {sample_id}",
        text=text,
        source_path=metadata["json_path"] or None,
        label_id=_optional_int(row.get("label_id")),
        sensor_type=row.get("sensor_type") or None,
        equipment_type=row.get("equipment_type") or None,
        insulator_type=row.get("insulator_type") or None,
        source_ref=sample_id,
        metadata=metadata,
    )


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    return parts[2].strip() if len(parts) == 3 else text


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _relative_path(value: str, data_root: Path) -> str:
    if value.strip() == "":
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(data_root / path)
