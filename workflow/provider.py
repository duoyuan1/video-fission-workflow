"""OpenAI-compatible API client for structured workflow generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


class ProviderError(RuntimeError):
    """Raised when the provider request fails or returns invalid data."""


@dataclass(frozen=True)
class ProviderSettings:
    model: str
    api_base: str
    api_key: str
    temperature: float = 0.2
    timeout_seconds: int = 180
    use_json_mode: bool = True


def resolve_provider_settings(
    *,
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    timeout_seconds: int,
    use_json_mode: bool,
) -> ProviderSettings | None:
    resolved_model = model or os.getenv("OPENAI_MODEL")
    resolved_base = api_base or os.getenv("OPENAI_BASE_URL")
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_model:
        return None
    if not resolved_base:
        raise ProviderError("API base URL is required. Pass --api-base or set OPENAI_BASE_URL.")
    if not resolved_key:
        raise ProviderError("API key is required. Pass --api-key or set OPENAI_API_KEY.")
    return ProviderSettings(
        model=resolved_model,
        api_base=resolved_base,
        api_key=resolved_key,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        use_json_mode=use_json_mode,
    )


def call_openai_compatible_json(
    *,
    prompt: str,
    response_template: dict[str, Any],
    settings: ProviderSettings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion_url = normalized_completion_url(settings.api_base)
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate valid JSON only. Do not include markdown fences, commentary, or prose outside JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "## Response template\n"
                    "Return a JSON object matching this shape exactly. Fill every field with meaningful content.\n\n"
                    f"{json.dumps(response_template, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "temperature": settings.temperature,
    }
    if settings.use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    req = request.Request(
        completion_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=settings.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"Provider returned HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise ProviderError(f"Provider request failed: {exc.reason}") from exc

    try:
        provider_response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("Provider response was not valid JSON.") from exc

    content = extract_message_content(provider_response)
    parsed = parse_json_content(content)
    return provider_response, parsed


def normalized_completion_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def extract_message_content(provider_response: dict[str, Any]) -> str:
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("Provider response did not contain choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProviderError("Provider response did not contain a message object.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    raise ProviderError("Provider response did not contain text content.")


def parse_json_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ProviderError("Could not extract a JSON object from provider content.")
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError("Extracted provider content was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("Provider content must decode to a JSON object.")
    return parsed


def write_provider_response(path: Path, provider_response: dict[str, Any]) -> None:
    path.write_text(json.dumps(provider_response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
