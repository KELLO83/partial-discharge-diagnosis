from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from service.backend.app.config.env import load_project_env


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_SITE_URL = "http://127.0.0.1:5173"
DEFAULT_APP_NAME = "Partial Discharge Diagnosis"
DEFAULT_TIMEOUT_SECONDS = 30.0


class OpenRouterConfigError(RuntimeError):
    pass


class OpenRouterRequestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenRouterSettings:
    api_key: str
    model: str = DEFAULT_OPENROUTER_MODEL
    site_url: str = DEFAULT_SITE_URL
    app_name: str = DEFAULT_APP_NAME
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> OpenRouterSettings:
        load_project_env()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise OpenRouterConfigError("OPENROUTER_API_KEY is not set")
        return cls(
            api_key=api_key,
            model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL,
            site_url=os.getenv("OPENROUTER_SITE_URL", DEFAULT_SITE_URL).strip() or DEFAULT_SITE_URL,
            app_name=os.getenv("OPENROUTER_APP_NAME", DEFAULT_APP_NAME).strip() or DEFAULT_APP_NAME,
            timeout_seconds=_env_float("OPENROUTER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        )


class OpenRouterClient:
    def __init__(self, settings: OpenRouterSettings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        response = self._post_json(payload)
        return _message_content(response)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.settings.site_url,
                "X-OpenRouter-Title": self.settings.app_name,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise OpenRouterRequestError(_http_error_message(exc)) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OpenRouterRequestError(str(exc)) from exc


def _message_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterRequestError("OpenRouter response did not include message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise OpenRouterRequestError("OpenRouter response content was empty")
    return content


def _http_error_message(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8")
    except Exception:
        body = ""
    if not body:
        return f"OpenRouter request failed with HTTP {exc.code}"
    return f"OpenRouter request failed with HTTP {exc.code}: {body[:500]}"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default
