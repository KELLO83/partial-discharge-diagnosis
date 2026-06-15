from __future__ import annotations

import re
from dataclasses import dataclass

from service.backend.app.rag.documents import RagSearchHit


PEAK_QUERY_TERMS = (
    "피크",
    "peak",
    "maxdischarge",
    "maxdischargevalue",
    "최대방전",
    "최대방전량",
)
VOLTAGE_QUERY_TERMS = (
    "전압",
    "voltage",
    "ratedvoltage",
    "equipmentratedvoltage",
)
TEMPERATURE_QUERY_TERMS = ("온도", "temperature")
HUMIDITY_QUERY_TERMS = ("습도", "humidity")
SENSOR_TYPES = ("HFCT", "UHF", "TEV")
POWER_FREQUENCY_QUERY_TERMS = (
    "전원주파수",
    "전원주파수hz",
    "powerfrequency",
    "powersupplyfrequency",
)
INSULATOR_TYPES = ("고체", "액체", "기체")
LABEL_NAME_ALIASES = {
    "정상": "정상",
    "노이즈": "노이즈",
    "표면": "표면 방전",
    "표면방전": "표면 방전",
    "코로나": "코로나 방전",
    "코로나방전": "코로나 방전",
    "보이드": "보이드 방전",
    "보이드방전": "보이드 방전",
}
LABEL_QUERY_TERMS = ("라벨", "label", "방전유형", "방전타입", "분류")
EQUIPMENT_NAME_ALIASES = {
    "단상유입변압기": "단상 유입변압기",
    "단상유압변압기": "단상 유입변압기",
    "전력용유입변압기": "전력용 유입변압기",
    "전력용유압변압기": "전력용 유입변압기",
    "계기용변압기": "계기용 변압기",
    "72kv배전반": "7.2kV 배전반",
    "258kvgis": "25.8kV GIS",
    "229kvgis": "22.9kV GIS",
    "acsroc": "ACSR-OC",
    "tfrcv": "TFR-CV",
}
METADATA_STOP_WORDS = {
    "검색",
    "결과",
    "데이터",
    "문서",
    "내용",
    "사례",
    "상태",
    "보여줘",
    "알려줘",
    "찾아줘",
    "설비",
    "절연",
    "라벨",
    "라벨은",
    "라벨는",
    "라벨이",
    "라벨가",
    "정격",
}
QUERY_VALUE_SUFFIXES = (
    "보여줘",
    "알려줘",
    "찾아줘",
    "검색",
    "결과",
    "데이터",
    "사례",
    "조건",
    "인것",
    "인거",
    "이면서",
    "면서",
)
QUERY_VALUE_PREFIXES = ("은", "는", "이", "가", "의", "값", "명")
STRUCTURED_TOKEN_MARKERS = ("_", "-", ".")
MIN_LONG_METADATA_TERM_LENGTH = 4
SAMPLE_ID_PATTERN = re.compile(
    r"[A-Za-z가-힣]+_[A-Za-z가-힣]+_[A-Za-z0-9가-힣.-]+_\d{6}_\d{6}_[A-Za-z0-9-]+_\d+(?:\.\d+)?"
)
RECORDING_TIME_PATTERN = re.compile(r"(?<!\d)(\d{6}[_-]?\d{6})(?!\d)")
OR_GROUP_PATTERN = re.compile(r"\s*(?:또는|혹은|아니면|\|)\s*|\s+\bor\b\s+", re.IGNORECASE)
NUMBER_PATTERN = r"(\d+(?:\.\d+)?)"
UNIT_PATTERN = r"(kv|v)?"
MAX_TERM_DISTANCE = 8
NUMERIC_TOLERANCE = 0.0001


@dataclass(frozen=True, slots=True)
class QueryMeasurement:
    value: float
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class AppliedQueryFilter:
    key: str
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class RagQueryConstraints:
    sample_id: str | None = None
    label_name: str | None = None
    peak_value: float | None = None
    voltage_value: float | None = None
    temperature_value: float | None = None
    humidity_value: float | None = None
    power_frequency_value: float | None = None
    equipment_name: str | None = None
    sensor_type: str | None = None
    insulator_type: str | None = None
    recording_time: str | None = None
    metadata_terms: tuple[str, ...] = ()
    or_groups: tuple[RagQueryConstraints, ...] = ()

    @property
    def has_sample_id(self) -> bool:
        return self.sample_id is not None

    @property
    def has_label_name(self) -> bool:
        return self.label_name is not None

    @property
    def has_peak(self) -> bool:
        return self.peak_value is not None

    @property
    def has_voltage(self) -> bool:
        return self.voltage_value is not None

    @property
    def has_temperature(self) -> bool:
        return self.temperature_value is not None

    @property
    def has_humidity(self) -> bool:
        return self.humidity_value is not None

    @property
    def has_power_frequency(self) -> bool:
        return self.power_frequency_value is not None

    @property
    def has_equipment_name(self) -> bool:
        return self.equipment_name is not None

    @property
    def has_sensor_type(self) -> bool:
        return self.sensor_type is not None

    @property
    def has_insulator_type(self) -> bool:
        return self.insulator_type is not None

    @property
    def has_recording_time(self) -> bool:
        return self.recording_time is not None

    @property
    def has_metadata_terms(self) -> bool:
        return len(self.metadata_terms) > 0

    @property
    def has_or_groups(self) -> bool:
        return any(group.has_constraints for group in self.or_groups)

    @property
    def has_constraints(self) -> bool:
        return (
            self.has_sample_id
            or self.has_label_name
            or self.has_peak
            or self.has_voltage
            or self.has_temperature
            or self.has_humidity
            or self.has_power_frequency
            or self.has_equipment_name
            or self.has_sensor_type
            or self.has_insulator_type
            or self.has_recording_time
            or self.has_metadata_terms
            or self.has_or_groups
        )


def extract_query_constraints(query: str) -> RagQueryConstraints:
    or_constraints = extract_or_query_constraints(query)
    if or_constraints is not None:
        return or_constraints
    return extract_single_query_constraints(query)


def extract_or_query_constraints(query: str) -> RagQueryConstraints | None:
    query_parts = [part.strip() for part in OR_GROUP_PATTERN.split(query) if part.strip()]
    if len(query_parts) < 2:
        return None
    groups = tuple(
        group
        for group in (extract_single_query_constraints(part) for part in query_parts)
        if group.has_constraints
    )
    if len(groups) < 2:
        return None
    return RagQueryConstraints(or_groups=groups)


def extract_single_query_constraints(query: str) -> RagQueryConstraints:
    recording_time = extract_recording_time(query)
    return RagQueryConstraints(
        sample_id=extract_sample_id(query),
        label_name=extract_label_name(query),
        peak_value=extract_peak_value(query),
        voltage_value=extract_voltage_value(query),
        temperature_value=extract_temperature_value(query),
        humidity_value=extract_humidity_value(query),
        power_frequency_value=extract_power_frequency_value(query),
        equipment_name=extract_equipment_name(query),
        sensor_type=extract_sensor_type(query),
        insulator_type=extract_insulator_type(query),
        recording_time=recording_time,
        metadata_terms=remove_structured_metadata_terms(
            extract_metadata_terms(query),
            (recording_time,),
        ),
    )


def extract_sample_id(query: str) -> str | None:
    match = SAMPLE_ID_PATTERN.search(query)
    return match.group(0) if match else None


def extract_label_name(query: str) -> str | None:
    compact_query = compact_text(query)
    for term in LABEL_QUERY_TERMS:
        label_value = _value_after_query_term(compact_query, compact_text(term))
        if label_value is not None:
            return LABEL_NAME_ALIASES.get(label_value, label_value)
    return None


def extract_peak_value(query: str) -> float | None:
    compact_query = compact_text(query)
    if compact_query == "":
        return None

    for term in PEAK_QUERY_TERMS:
        measurement = _measurement_after_term(compact_query, term) or _measurement_before_term(compact_query, term)
        if measurement is not None:
            return measurement.value
    return None


def extract_voltage_value(query: str) -> float | None:
    compact_query = compact_text(query)
    if compact_query == "":
        return None

    for term in VOLTAGE_QUERY_TERMS:
        measurement = _measurement_after_term(compact_query, term) or _measurement_before_term(compact_query, term)
        if measurement is not None:
            return normalize_voltage_value(measurement)
    return None


def extract_temperature_value(query: str) -> float | None:
    return _extract_plain_measurement_value(query, TEMPERATURE_QUERY_TERMS) or _measurement_value_from_pattern(
        query,
        rf"(?<!\d){NUMBER_PATTERN}\s*도(?![A-Za-z가-힣])",
    )


def extract_humidity_value(query: str) -> float | None:
    return _extract_plain_measurement_value(query, HUMIDITY_QUERY_TERMS) or _measurement_value_from_pattern(
        query,
        rf"(?<!\d){NUMBER_PATTERN}\s*(?:%|퍼센트|프로)",
    )


def extract_power_frequency_value(query: str) -> float | None:
    return _extract_plain_measurement_value(query, POWER_FREQUENCY_QUERY_TERMS)


def extract_equipment_name(query: str) -> str | None:
    compact_query = compact_text(query)
    for alias, equipment_name in EQUIPMENT_NAME_ALIASES.items():
        if alias in compact_query:
            return equipment_name
    return None


def extract_sensor_type(query: str) -> str | None:
    compact_query = compact_text(query)
    for sensor_type in SENSOR_TYPES:
        if compact_text(sensor_type) in compact_query:
            return sensor_type
    return None


def extract_insulator_type(query: str) -> str | None:
    compact_query = compact_text(query)
    for insulator_type in INSULATOR_TYPES:
        if compact_text(insulator_type) in compact_query:
            return insulator_type
    return None


def extract_recording_time(query: str) -> str | None:
    match = RECORDING_TIME_PATTERN.search(query)
    if match is None:
        return None
    return normalize_recording_time(match.group(1))


def normalize_recording_time(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) != 12:
        return value
    return f"{digits[:6]}_{digits[6:]}"


def applied_query_filters(constraints: RagQueryConstraints) -> list[AppliedQueryFilter]:
    if constraints.has_or_groups:
        return [
            AppliedQueryFilter(
                key="or_groups",
                label="OR 조건",
                value=" 또는 ".join(applied_filter_text(group) for group in constraints.or_groups),
            )
        ]
    filters: list[AppliedQueryFilter] = []
    _append_filter(filters, "sample_id", "샘플 ID", constraints.sample_id)
    _append_filter(filters, "label_name", "방전유형", constraints.label_name)
    _append_filter(filters, "peak_value", "피크", _number_text(constraints.peak_value))
    _append_filter(filters, "voltage_value", "전압", _number_text(constraints.voltage_value, "V"))
    _append_filter(filters, "temperature_value", "온도", _number_text(constraints.temperature_value, "도"))
    _append_filter(filters, "humidity_value", "습도", _number_text(constraints.humidity_value, "%"))
    _append_filter(filters, "power_frequency_value", "전원주파수", _number_text(constraints.power_frequency_value, "Hz"))
    _append_filter(filters, "equipment_name", "설비", constraints.equipment_name)
    _append_filter(filters, "sensor_type", "센서", constraints.sensor_type)
    _append_filter(filters, "insulator_type", "절연", constraints.insulator_type)
    _append_filter(filters, "recording_time", "기록시각", constraints.recording_time)
    if constraints.metadata_terms:
        _append_filter(filters, "metadata_terms", "문서 키워드", ", ".join(constraints.metadata_terms))
    return filters


def applied_filter_text(constraints: RagQueryConstraints) -> str:
    filters = applied_query_filters(constraints)
    return ", ".join(f"{item.label}={item.value}" for item in filters) or "조건 없음"


def extract_metadata_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    for raw_token in re.findall(r"[A-Za-z0-9_.-]+|[가-힣]+", query):
        term = compact_text(raw_token)
        if not is_metadata_term(term, raw_token):
            continue
        terms.append(term)
    return tuple(dict.fromkeys(terms))


def remove_structured_metadata_terms(
    metadata_terms: tuple[str, ...],
    structured_values: tuple[str | None, ...],
) -> tuple[str, ...]:
    structured_terms = {
        compact_text(value)
        for value in structured_values
        if value is not None
    }
    return tuple(term for term in metadata_terms if term not in structured_terms)


def is_metadata_term(term: str, raw_token: str = "") -> bool:
    if term in METADATA_STOP_WORDS:
        return False
    if not any(character.isdigit() for character in term):
        return False
    if len(term) >= MIN_LONG_METADATA_TERM_LENGTH:
        return True
    return any(marker in raw_token for marker in STRUCTURED_TOKEN_MARKERS)


def filter_hits_by_constraints(
    hits: list[RagSearchHit],
    constraints: RagQueryConstraints,
) -> list[RagSearchHit]:
    if not constraints.has_constraints:
        return hits
    return [hit for hit in hits if hit_matches_constraints(hit, constraints)]


def hit_matches_constraints(hit: RagSearchHit, constraints: RagQueryConstraints) -> bool:
    if constraints.has_or_groups:
        return any(hit_matches_constraints(hit, group) for group in constraints.or_groups)
    if constraints.sample_id is not None and not metadata_text_matches(hit, "sample_id", constraints.sample_id):
        return False
    if constraints.label_name is not None and not metadata_text_matches(hit, "label_name", constraints.label_name):
        return False
    if constraints.peak_value is not None:
        hit_peak_value = hit_peak(hit)
        if hit_peak_value is None or not same_number(hit_peak_value, constraints.peak_value):
            return False
    if constraints.voltage_value is not None:
        hit_voltage_value = hit_voltage(hit)
        if hit_voltage_value is None or not same_number(hit_voltage_value, constraints.voltage_value):
            return False
    if constraints.temperature_value is not None:
        hit_temperature_value = hit_temperature(hit)
        if hit_temperature_value is None or not same_number(hit_temperature_value, constraints.temperature_value):
            return False
    if constraints.humidity_value is not None:
        hit_humidity_value = hit_humidity(hit)
        if hit_humidity_value is None or not same_number(hit_humidity_value, constraints.humidity_value):
            return False
    if constraints.power_frequency_value is not None:
        hit_power_frequency_value = hit_power_frequency(hit)
        if hit_power_frequency_value is None or not same_number(
            hit_power_frequency_value,
            constraints.power_frequency_value,
        ):
            return False
    if constraints.equipment_name is not None and not metadata_text_contains(
        hit,
        "equipment_name",
        constraints.equipment_name,
    ):
        return False
    if constraints.sensor_type is not None and not metadata_text_matches(
        hit,
        "sensor_type",
        constraints.sensor_type,
    ):
        return False
    if constraints.insulator_type is not None and not metadata_text_matches(
        hit,
        "insulator_type",
        constraints.insulator_type,
    ):
        return False
    if constraints.recording_time is not None and not metadata_text_matches(
        hit,
        "recording_time",
        constraints.recording_time,
    ):
        return False
    if constraints.metadata_terms and not hit_contains_metadata_term(hit, constraints.metadata_terms):
        return False
    return True


def hit_peak(hit: RagSearchHit) -> float | None:
    metadata_value = hit.metadata.get("max_discharge_value")
    if metadata_value is not None:
        return optional_float(metadata_value)
    match = re.search(r"max_discharge\s*=\s*(\d+(?:\.\d+)?)", hit.text)
    return optional_float(match.group(1)) if match else None


def hit_voltage(hit: RagSearchHit) -> float | None:
    metadata_value = hit.metadata.get("equipment_rated_voltage")
    if metadata_value is not None:
        return optional_float(metadata_value)
    match = re.search(r"voltage\s*=\s*(\d+(?:\.\d+)?)\s*(kv|v)?", hit.text, re.IGNORECASE)
    if not match:
        return None
    return normalize_voltage_value(QueryMeasurement(optional_float(match.group(1)) or 0.0, match.group(2)))


def hit_temperature(hit: RagSearchHit) -> float | None:
    return metadata_number(hit, "temperature", r"temperature\s*=\s*(\d+(?:\.\d+)?)")


def hit_humidity(hit: RagSearchHit) -> float | None:
    return metadata_number(hit, "humidity", r"humidity\s*=\s*(\d+(?:\.\d+)?)")


def hit_power_frequency(hit: RagSearchHit) -> float | None:
    return metadata_number(hit, "power_supply_frequency", r"power_frequency\s*=\s*(\d+(?:\.\d+)?)")


def metadata_number(hit: RagSearchHit, key: str, text_pattern: str) -> float | None:
    metadata_value = hit.metadata.get(key)
    if metadata_value is not None:
        return optional_float(metadata_value)
    match = re.search(text_pattern, hit.text, re.IGNORECASE)
    return optional_float(match.group(1)) if match else None


def metadata_text_contains(hit: RagSearchHit, key: str, expected: str) -> bool:
    return compact_text(expected) in compact_text(str(metadata_text_value(hit, key)))


def metadata_text_matches(hit: RagSearchHit, key: str, expected: str) -> bool:
    return compact_text(expected) == compact_text(str(metadata_text_value(hit, key)))


def metadata_text_value(hit: RagSearchHit, key: str) -> object:
    if key in hit.metadata:
        return hit.metadata[key]
    return getattr(hit, key, "")


def hit_contains_metadata_term(hit: RagSearchHit, metadata_terms: tuple[str, ...]) -> bool:
    haystack = compact_text(
        " ".join(
            [
                hit.title,
                hit.text,
                hit.source,
                " ".join(str(value) for value in hit.metadata.values()),
            ]
        )
    )
    return all(term in haystack for term in metadata_terms)


def compact_text(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum() or character == ".")


def optional_float(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        match = re.search(NUMBER_PATTERN, text)
        return float(match.group(1)) if match else None


def normalize_voltage_value(measurement: QueryMeasurement) -> float:
    if measurement.unit == "kv":
        return measurement.value * 1000
    return measurement.value


def same_number(left: float, right: float) -> bool:
    return abs(left - right) <= NUMERIC_TOLERANCE


def _append_filter(
    filters: list[AppliedQueryFilter],
    key: str,
    label: str,
    value: str | None,
) -> None:
    if value is not None and value != "":
        filters.append(AppliedQueryFilter(key=key, label=label, value=value))


def _number_text(value: float | None, suffix: str = "") -> str | None:
    if value is None:
        return None
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:g}{suffix}"


def _extract_plain_measurement_value(query: str, query_terms: tuple[str, ...]) -> float | None:
    compact_query = compact_text(query)
    if compact_query == "":
        return None
    for term in query_terms:
        measurement = _measurement_after_term(compact_query, term) or _measurement_before_term(compact_query, term)
        if measurement is not None:
            return measurement.value
    return None


def _measurement_value_from_pattern(query: str, pattern: str) -> float | None:
    match = re.search(pattern, query, re.IGNORECASE)
    if match is None:
        return None
    return optional_float(match.group(1))


def _value_after_query_term(compact_query: str, term: str) -> str | None:
    index = compact_query.find(term)
    if index < 0:
        return None
    value = compact_query[index + len(term):]
    value = _strip_query_value_prefix(value)
    if value == "":
        return None
    for alias in sorted(LABEL_NAME_ALIASES, key=len, reverse=True):
        if value.startswith(alias):
            return alias
    return _trim_query_value_suffix(value)


def _strip_query_value_prefix(value: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in QUERY_VALUE_PREFIXES:
            if value.startswith(prefix):
                value = value[len(prefix):]
                changed = True
    return value


def _trim_query_value_suffix(value: str) -> str:
    suffix_positions = [index for suffix in QUERY_VALUE_SUFFIXES if (index := value.find(suffix)) > 0]
    if suffix_positions:
        return value[:min(suffix_positions)]
    return value


def _measurement_after_term(compact_query: str, term: str) -> QueryMeasurement | None:
    pattern = rf"{re.escape(term)}\D{{0,{MAX_TERM_DISTANCE}}}{NUMBER_PATTERN}{UNIT_PATTERN}"
    match = re.search(pattern, compact_query)
    return _measurement_from_match(match)


def _measurement_before_term(compact_query: str, term: str) -> QueryMeasurement | None:
    pattern = rf"{NUMBER_PATTERN}{UNIT_PATTERN}\D{{0,{MAX_TERM_DISTANCE}}}{re.escape(term)}"
    match = re.search(pattern, compact_query)
    return _measurement_from_match(match)


def _measurement_from_match(match: re.Match[str] | None) -> QueryMeasurement | None:
    if match is None:
        return None
    value = optional_float(match.group(1))
    if value is None:
        return None
    unit = match.group(2) or None
    return QueryMeasurement(value=value, unit=unit)
