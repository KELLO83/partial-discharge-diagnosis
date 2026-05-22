import csv
import json
from pathlib import Path


LABEL_NAMES = {
    0: "정상",
    1: "노이즈",
    2: "표면 방전",
    3: "코로나 방전",
    4: "보이드 방전",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_file_index(root: Path, suffix: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob(f"*{suffix}"):
        sample_id = path.stem
        if sample_id in index:
            raise ValueError(f"Duplicate sample id for {suffix}: {sample_id}")
        index[sample_id] = path
    return index


def value(data: dict, *keys: str, default: str = ""):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    train_root = Path("Train")
    source_root = train_root / "01.원천데이터"
    label_root = train_root / "02.라벨링데이터"
    output_path = train_root / "manifest.csv"

    csv_index = build_file_index(source_root, ".csv")
    png_index = build_file_index(source_root, ".png")
    json_files = sorted(label_root.rglob("*.json"))

    fieldnames = [
        "sample_id",
        "split",
        "json_path",
        "image_path",
        "timeseries_path",
        "label_id",
        "label_name",
        "insulator_type",
        "insulator_name",
        "equipment_name",
        "equipment_manufacturer",
        "equipment_id",
        "equipment_rated_voltage",
        "equipment_rated_current",
        "recording_time",
        "recording_time_length",
        "data_collector",
        "power_supply_id",
        "power_supply_voltage_type",
        "power_supply_frequency",
        "power_supply_ramping_up_time",
        "power_supply_cutoff_current",
        "sensor_type",
        "temperature",
        "humidity",
        "clearance_distance",
        "IEC_standard",
        "engage_start_time",
        "defect_nums",
        "defect_details",
        "max_discharge_value",
        "json_image_path",
        "json_timeseries_path",
    ]

    rows = []
    missing = []

    for json_path in json_files:
        sample_id = json_path.stem
        image_path = png_index.get(sample_id)
        timeseries_path = csv_index.get(sample_id)
        if image_path is None or timeseries_path is None:
            missing.append(
                {
                    "sample_id": sample_id,
                    "json_path": str(json_path),
                    "has_image": image_path is not None,
                    "has_timeseries": timeseries_path is not None,
                }
            )
            continue

        data = read_json(json_path)
        label_id = int(value(data, "label", "PD_type", default=-1))
        equipment = value(data, "metadata", "equipment_information", default={})
        environment = value(data, "metadata", "environment", default={})
        discharge = value(data, "metadata", "discharge_information", default={})
        factors = value(data, "metadata", "discharge_evaluation_factors", default={})

        rows.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "json_path": json_path.as_posix(),
                "image_path": image_path.as_posix(),
                "timeseries_path": timeseries_path.as_posix(),
                "label_id": label_id,
                "label_name": LABEL_NAMES.get(label_id, ""),
                "insulator_type": equipment.get("insulator_type", ""),
                "insulator_name": equipment.get("insulator_name", ""),
                "equipment_name": equipment.get("equipment_name", ""),
                "equipment_manufacturer": equipment.get("equipment_manufacturer", ""),
                "equipment_id": equipment.get("equipment_id", ""),
                "equipment_rated_voltage": equipment.get("equipment_rated_voltage", ""),
                "equipment_rated_current": equipment.get("equipment_rated_current", ""),
                "recording_time": environment.get("recording_time", ""),
                "recording_time_length": environment.get("recording_time_length", ""),
                "data_collector": environment.get("data_collector", ""),
                "power_supply_id": environment.get("power_supply_id", ""),
                "power_supply_voltage_type": environment.get("power_supply_voltage type", ""),
                "power_supply_frequency": environment.get("power_supply_frequency", ""),
                "power_supply_ramping_up_time": environment.get(
                    "power_supply_ramping_up_time", ""
                ),
                "power_supply_cutoff_current": environment.get(
                    "power_supply_cutoff_current", ""
                ),
                "sensor_type": environment.get("sensor_type", ""),
                "temperature": environment.get("temperature", ""),
                "humidity": environment.get("humidity", ""),
                "clearance_distance": environment.get("clearance_distance", ""),
                "IEC_standard": environment.get("IEC_standard", ""),
                "engage_start_time": environment.get("engage_start_time", ""),
                "defect_nums": discharge.get("defect_nums", ""),
                "defect_details": json.dumps(
                    discharge.get("defect_details", ""), ensure_ascii=False
                ),
                "max_discharge_value": factors.get("max_discharge_value", ""),
                "json_image_path": value(data, "label", "image_path", default=""),
                "json_timeseries_path": value(
                    data, "label", "timeseries_path", default=""
                ),
            }
        )

    if missing:
        raise RuntimeError(f"Missing matched files: {missing[:5]}")

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {output_path}")
    print(f"rows: {len(rows)}")
    print(f"csv files indexed: {len(csv_index)}")
    print(f"png files indexed: {len(png_index)}")
    print(f"json files read: {len(json_files)}")


if __name__ == "__main__":
    main()
