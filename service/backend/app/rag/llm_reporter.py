from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from service.backend.app.config.env import load_project_env
from service.backend.app.application.contracts import VlmInferenceAdapter, VlmToolInput
from service.backend.app.domain.fusion import vlm_evidence
from service.backend.app.domain.policy import label_name, recommended_action, risk_level
from service.backend.app.rag.llm_prompt_builder import build_llm_rag_messages
from service.backend.app.rag.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterClient,
    OpenRouterConfigError,
    OpenRouterRequestError,
    OpenRouterSettings,
)
from service.backend.app.schemas import VlmResult


DEFAULT_LLM_RAG_PROVIDER = "auto"
SUPPORTED_LLM_RAG_PROVIDERS = {"auto", "mock", "openrouter"}


class LlmRagReportError(RuntimeError):
    pass


class ChatCompleter(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True, slots=True)
class LlmRagReporterStatus:
    provider: str
    active_adapter: str
    model: str | None
    ready: bool
    error: str | None = None


class LlmRagReportPayload(BaseModel):
    label_id: int = Field(ge=0, le=4)
    diagnosis: str = Field(min_length=1)
    risk_level: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)


class OpenRouterLlmRagInferenceAdapter(VlmInferenceAdapter):
    model_name = "openrouter_llm_rag"

    def __init__(self, client: ChatCompleter, model_version: str) -> None:
        self._client = client
        self.model_version = model_version

    def run(self, tool_input: VlmToolInput) -> VlmResult:
        messages = build_llm_rag_messages(tool_input)
        try:
            content = self._client.complete(messages)
            payload = parse_llm_rag_payload(content)
        except (OpenRouterRequestError, ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise LlmRagReportError(str(exc)) from exc
        return _result_from_payload(self.model_name, self.model_version, payload)


class FallbackVlmInferenceAdapter(VlmInferenceAdapter):
    def __init__(
        self,
        primary: VlmInferenceAdapter,
        fallback: VlmInferenceAdapter,
        status: LlmRagReporterStatus,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._status = status

    @property
    def model_name(self) -> str:
        if self._status.ready:
            return str(self._primary.model_name)
        return str(self._fallback.model_name)

    @property
    def model_version(self) -> str:
        if self._status.ready:
            return str(self._primary.model_version)
        return str(self._fallback.model_version)

    def run(self, tool_input: VlmToolInput) -> VlmResult:
        if not self._status.ready:
            return self._fallback.run(tool_input)
        try:
            return self._primary.run(tool_input)
        except LlmRagReportError:
            return self._fallback.run(tool_input)


def build_llm_rag_reporter(fallback: VlmInferenceAdapter) -> tuple[VlmInferenceAdapter, LlmRagReporterStatus]:
    provider = _provider_from_env()
    if provider == "mock":
        status = _inactive_status(provider, fallback, None)
        return fallback, status
    try:
        settings = OpenRouterSettings.from_env()
    except OpenRouterConfigError as exc:
        status = _inactive_status(provider, fallback, str(exc))
        return fallback, status
    primary = OpenRouterLlmRagInferenceAdapter(OpenRouterClient(settings), settings.model)
    status = LlmRagReporterStatus(
        provider="openrouter" if provider == "auto" else provider,
        active_adapter=primary.model_name,
        model=settings.model,
        ready=True,
    )
    return FallbackVlmInferenceAdapter(primary, fallback, status), status


def parse_llm_rag_payload(content: str) -> LlmRagReportPayload:
    payload = json.loads(_extract_json_object(content))
    return LlmRagReportPayload.model_validate(payload)


def _provider_from_env() -> str:
    load_project_env()
    provider = os.getenv("LLM_RAG_PROVIDER", DEFAULT_LLM_RAG_PROVIDER).strip().lower()
    if provider in SUPPORTED_LLM_RAG_PROVIDERS:
        return provider
    return DEFAULT_LLM_RAG_PROVIDER


def _inactive_status(
    provider: str,
    fallback: VlmInferenceAdapter,
    error: str | None,
) -> LlmRagReporterStatus:
    return LlmRagReporterStatus(
        provider=provider,
        active_adapter=str(fallback.model_name),
        model=DEFAULT_OPENROUTER_MODEL if provider != "mock" else None,
        ready=False,
        error=error,
    )


def _result_from_payload(model_name: str, model_version: str, payload: LlmRagReportPayload) -> VlmResult:
    result = VlmResult(
        model_name=model_name,
        model_version=model_version,
        label_id=payload.label_id,
        diagnosis=payload.diagnosis.strip() or label_name(payload.label_id),
        risk_level=payload.risk_level.strip() or risk_level(payload.label_id),
        confidence=payload.confidence,
        reason=payload.reason.strip(),
        recommended_action=payload.recommended_action.strip() or recommended_action(payload.label_id),
    )
    return result.model_copy(update={"standard_evidence": vlm_evidence(result)})


def _extract_json_object(content: str) -> str:
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.strip("`").strip()
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return clean[start : end + 1]
