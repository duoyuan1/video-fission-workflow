"""OpenAI-compatible API client for structured workflow generation."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse


class ProviderError(RuntimeError):
    """Raised when the provider request fails or returns invalid data."""


@dataclass(frozen=True)
class ProviderAttachment:
    kind: str
    source: str
    name: str
    mime_type: str


@dataclass(frozen=True)
class ProviderSettings:
    model: str
    api_base: str
    api_key: str
    auth_scheme: str = "bearer"
    temperature: float = 0.2
    timeout_seconds: int = 180
    use_json_mode: bool = True


def resolve_provider_settings(
    *,
    command: str,
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    timeout_seconds: int,
    use_json_mode: bool,
) -> ProviderSettings | None:
    stage_settings = resolve_stage_env_settings(command)
    resolved_model = model or stage_settings["model"] or os.getenv("OPENAI_MODEL")
    if not resolved_model:
        return None
    resolved_base = api_base or stage_settings["api_base"] or os.getenv("OPENAI_BASE_URL")
    resolved_key = api_key or stage_settings["api_key"] or os.getenv("OPENAI_API_KEY")
    auth_scheme = stage_settings["auth_scheme"] or os.getenv("OPENAI_AUTH_SCHEME", "")
    if is_dashscope_model(resolved_model):
        resolved_base = resolved_base or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        resolved_key = resolved_key or os.getenv("DASHSCOPE_API_KEY")
        auth_scheme = auth_scheme or os.getenv("DASHSCOPE_AUTH_SCHEME", "bearer")
    if is_yunwu_base(resolved_base):
        resolved_key = resolved_key or os.getenv("YUNWU_API_KEY")
        auth_scheme = auth_scheme or os.getenv("YUNWU_AUTH_SCHEME", "raw")
    if not resolved_base:
        raise ProviderError("API base URL is required. Pass --api-base or set OPENAI_BASE_URL.")
    if not resolved_key:
        if is_dashscope_model(resolved_model):
            raise ProviderError("API key is required. Pass --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY.")
        if is_yunwu_base(resolved_base):
            raise ProviderError("API key is required. Pass --api-key or set YUNWU_API_KEY / OPENAI_API_KEY.")
        raise ProviderError("API key is required. Pass --api-key or set OPENAI_API_KEY.")
    return ProviderSettings(
        model=resolved_model,
        api_base=resolved_base,
        api_key=resolved_key,
        auth_scheme=normalize_auth_scheme(auth_scheme or "bearer"),
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        use_json_mode=use_json_mode,
    )


def call_openai_compatible_json(
    *,
    prompt: str,
    response_template: dict[str, Any],
    settings: ProviderSettings,
    attachments: list[ProviderAttachment] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    text_block = (
        f"{prompt}\n\n"
        "## Response template\n"
        "Return a JSON object matching this shape exactly. Fill every field with meaningful content.\n\n"
        f"{json.dumps(response_template, ensure_ascii=False, indent=2)}"
    )
    completion_url = normalized_completion_url(settings.api_base)
    user_content: str | list[dict[str, Any]]
    if attachments:
        content_items: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
        for attachment in attachments:
            content_item = attachment_to_content_item(attachment)
            if content_item is not None:
                content_items.append(content_item)
        user_content = content_items
    else:
        user_content = text_block

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
                "content": user_content,
            },
        ],
        "temperature": settings.temperature,
    }
    if settings.use_json_mode and not requires_streaming_response(settings):
        payload["response_format"] = {"type": "json_object"}
    if requires_streaming_response(settings):
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        payload["modalities"] = ["text"]

    req = request.Request(
        completion_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": build_authorization_header_value(settings),
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=settings.timeout_seconds) as resp:
            if requires_streaming_response(settings):
                provider_response = read_streaming_response(resp)
            else:
                raw = resp.read().decode("utf-8")
                provider_response = json.loads(raw)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"Provider returned HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise ProviderError(f"Provider request failed: {exc.reason}") from exc

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


def resolve_stage_env_settings(command: str) -> dict[str, str]:
    settings = {"model": "", "api_base": "", "api_key": "", "auth_scheme": ""}
    for prefix in command_env_prefixes(command):
        settings["model"] = settings["model"] or os.getenv(f"{prefix}_MODEL", "")
        settings["api_base"] = settings["api_base"] or os.getenv(f"{prefix}_API_BASE", "")
        settings["api_key"] = settings["api_key"] or os.getenv(f"{prefix}_API_KEY", "")
        settings["auth_scheme"] = settings["auth_scheme"] or os.getenv(f"{prefix}_AUTH_SCHEME", "")
    return settings


def command_env_prefixes(command: str) -> list[str]:
    mapping = {
        "analyze": ["ANALYZE"],
        "plan-directions": ["PLAN", "PLANNING"],
        "gen-script": ["SCRIPT", "PLANNING"],
        "gen-assets": ["ASSET", "PLANNING"],
        "gen-storyboards": ["STORYBOARD", "PLANNING"],
        "qa": ["QA", "PLANNING"],
        "render-videos": ["RENDER"],
    }
    return mapping.get(command, [])


def normalize_auth_scheme(value: str) -> str:
    normalized = (value or "bearer").strip().lower()
    if normalized in {"bearer", "raw"}:
        return normalized
    raise ProviderError("AUTH_SCHEME must be `bearer` or `raw`.")


def build_authorization_header_value(settings: ProviderSettings) -> str:
    if settings.auth_scheme == "raw":
        return settings.api_key
    return f"Bearer {settings.api_key}"


def is_dashscope_model(model: str) -> bool:
    return model.startswith("qwen")


def is_yunwu_base(api_base: str | None) -> bool:
    return bool(api_base) and "yunwu.ai" in api_base


def is_qwen_omni_model(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith("qwen") and "omni" in normalized


def requires_streaming_response(settings: ProviderSettings) -> bool:
    return is_qwen_omni_model(settings.model)


def read_streaming_response(response: Any) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ProviderError("Streaming provider response contained invalid JSON.") from exc
        chunks.append(chunk)

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage = chunk_usage

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        delta_text = extract_delta_text(delta)
        if delta_text:
            text_parts.append(delta_text)

    joined = "".join(text_parts).strip()
    if not joined:
        raise ProviderError("Streaming provider response did not contain text content.")

    return {
        "object": "chat.completion.stream.aggregate",
        "choices": [{"message": {"role": "assistant", "content": joined}}],
        "usage": usage or {},
        "chunks": chunks,
    }


def extract_delta_text(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def attachment_to_content_item(attachment: ProviderAttachment) -> dict[str, Any] | None:
    if attachment.kind == "image":
        return {
            "type": "image_url",
            "image_url": {
                "url": attachment_to_url(attachment),
            },
        }
    if attachment.kind == "video":
        return {
            "type": "video_url",
            "video_url": {
                "url": attachment_to_url(attachment),
            },
        }
    return None


def attachment_to_url(attachment: ProviderAttachment) -> str:
    parsed = urlparse(attachment.source)
    if parsed.scheme in {"http", "https"}:
        return attachment.source

    path = Path(attachment.source)
    if not path.exists():
        raise ProviderError(f"Attachment file was not found: {path}")
    if not path.is_file():
        raise ProviderError(f"Attachment path must be a file: {path}")

    mime_type = attachment.mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
