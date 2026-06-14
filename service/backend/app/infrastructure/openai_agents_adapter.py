from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentsSdkAvailability:
    installed: bool
    reason: str


def check_agents_sdk() -> AgentsSdkAvailability:
    try:
        import agents  # noqa: F401
    except ImportError:
        return AgentsSdkAvailability(False, "openai-agents package is not installed")
    return AgentsSdkAvailability(True, "openai-agents package is installed")
