from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    label_id: int
    label_name: str
    risk_level: str
    recommended_action: str


LABEL_POLICIES: dict[int, LabelPolicy] = {
    0: LabelPolicy(0, "정상", "낮음", "정상 상태로 판단되며 정기 모니터링을 유지하세요."),
    1: LabelPolicy(1, "노이즈", "낮음", "센서 접촉 상태와 주변 전자기 간섭 가능성을 점검하세요."),
    2: LabelPolicy(2, "표면 방전", "주의", "절연체 표면 오염과 트래킹 흔적을 점검하세요."),
    3: LabelPolicy(3, "코로나 방전", "주의", "고전압 접속부와 전계 집중 부위를 점검하세요."),
    4: LabelPolicy(4, "보이드 방전", "위험", "절연체 내부 결함 가능성을 고려해 정밀 진단을 진행하세요."),
}

MIN_CONFIDENCE = 0.60
PROBABILITY_SUM_TOLERANCE = 0.05


def label_name(label_id: int) -> str:
    return _policy(label_id).label_name


def risk_level(label_id: int) -> str:
    return _policy(label_id).risk_level


def recommended_action(label_id: int) -> str:
    return _policy(label_id).recommended_action


def valid_label_id(label_id: int) -> bool:
    return label_id in LABEL_POLICIES


def _policy(label_id: int) -> LabelPolicy:
    try:
        return LABEL_POLICIES[label_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported label_id: {label_id}") from exc
