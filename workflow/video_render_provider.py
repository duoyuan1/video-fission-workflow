"""Async video-render provider support for execution-layer workflow stages."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from workflow.provider import (
    ProviderAttachment,
    ProviderError,
    attachment_to_url,
    normalize_auth_scheme,
    resolve_stage_env_settings,
)


@dataclass(frozen=True)
class RenderProviderSettings:
    provider_label: str
    model: str
    api_base: str
    api_key: str
    auth_scheme: str
    create_path: str
    status_path: str
    request_format: str
    duration: str
    resolution: str
    service_tier: str
    generate_audio: bool
    auto_poll: bool
    poll_seconds: int
    max_polls: int
    max_images: int
    timeout_seconds: int


def resolve_render_provider_settings(
    *,
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    timeout_seconds: int,
    config: dict[str, Any],
) -> RenderProviderSettings | None:
    stage_settings = resolve_stage_env_settings("render-videos")
    resolved_model = model or stage_settings["model"] or os.getenv("RENDER_MODEL", "")
    if not resolved_model:
        return None

    resolved_base = (
        api_base
        or stage_settings["api_base"]
        or os.getenv("RENDER_API_BASE")
        or os.getenv("YUNWU_BASE_URL")
        or ""
    )
    if not resolved_base:
        raise ProviderError("Render API base URL is required. Set RENDER_API_BASE or pass --api-base.")

    resolved_key = (
        api_key
        or stage_settings["api_key"]
        or os.getenv("RENDER_API_KEY")
        or (os.getenv("YUNWU_API_KEY") if "yunwu.ai" in resolved_base else "")
        or os.getenv("OPENAI_API_KEY")
    )
    if not resolved_key:
        raise ProviderError("Render API key is required. Set RENDER_API_KEY or pass --api-key.")

    auth_scheme = stage_settings["auth_scheme"] or os.getenv("RENDER_AUTH_SCHEME", "")
    if "yunwu.ai" in resolved_base:
        auth_scheme = auth_scheme or os.getenv("YUNWU_AUTH_SCHEME", "raw")
    auth_scheme = normalize_auth_scheme(auth_scheme or "bearer")

    create_path = os.getenv("RENDER_CREATE_PATH", "/v1/videos")
    status_path = os.getenv("RENDER_STATUS_PATH", "/v1/videos/{task_id}")
    request_format = (os.getenv("RENDER_REQUEST_FORMAT", "content") or "content").strip().lower()
    if request_format not in {"content", "simple"}:
        raise ProviderError("RENDER_REQUEST_FORMAT must be `content` or `simple`.")

    duration = str(os.getenv("RENDER_DURATION", config["episode_duration_seconds"]))
    resolution = os.getenv("RENDER_RESOLUTION", "").strip()
    service_tier = os.getenv("RENDER_SERVICE_TIER", "").strip()
    generate_audio = env_bool("RENDER_GENERATE_AUDIO", default=True)
    auto_poll = env_bool("RENDER_AUTO_POLL", default=True)
    poll_seconds = env_int("RENDER_POLL_SECONDS", default=20)
    max_polls = env_int("RENDER_MAX_POLLS", default=90)
    max_images = env_int("RENDER_MAX_IMAGES", default=4)
    provider_label = os.getenv("RENDER_PROVIDER", "yunwu-seedance-1.5").strip() or "yunwu-seedance-1.5"

    return RenderProviderSettings(
        provider_label=provider_label,
        model=resolved_model,
        api_base=resolved_base.rstrip("/"),
        api_key=resolved_key,
        auth_scheme=auth_scheme,
        create_path=create_path,
        status_path=status_path,
        request_format=request_format,
        duration=duration,
        resolution=resolution,
        service_tier=service_tier,
        generate_audio=generate_audio,
        auto_poll=auto_poll,
        poll_seconds=poll_seconds,
        max_polls=max_polls,
        max_images=max_images,
        timeout_seconds=timeout_seconds,
    )


def render_videos_via_provider(
    *,
    project_root: Path,
    config: dict[str, Any],
    execution_plan: dict[str, Any],
    settings: RenderProviderSettings,
    existing_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resumable = bool(existing_payload) and payload_has_segment_results(existing_payload) and has_pending_tasks(existing_payload)
    if resumable:
        episodes = [dict(item) for item in existing_payload.get("episodes", [])]
        provider_response = {"provider": settings.provider_label, "resumed": True, "polls": []}
    else:
        provider_response = {"provider": settings.provider_label, "resumed": False, "submissions": [], "polls": []}
        episodes = []
        for episode in execution_plan["episodes"]:
            if not episode["ready_to_render"]:
                episodes.append(build_skipped_episode_result(config, episode))
                continue

            segment_results: list[dict[str, Any]] = []
            for segment in episode.get("generation_segments", []):
                submission_payload = build_render_submission_payload(project_root, config, episode, segment, settings)
                submission_response = request_json(
                    method="POST",
                    url=join_url(settings.api_base, settings.create_path),
                    api_key=settings.api_key,
                    auth_scheme=settings.auth_scheme,
                    payload=submission_payload,
                    timeout_seconds=settings.timeout_seconds,
                )
                provider_response["submissions"].append(
                    {
                        "episode_number": episode["episode_number"],
                        "segment_id": segment["segment_id"],
                        "request": submission_payload,
                        "response": submission_response,
                    }
                )
                task_id = extract_task_id(submission_response)
                status = map_render_status(extract_status(submission_response) or "submitted")
                output_video = extract_output_video(submission_response)
                segment_results.append(
                    build_segment_result(
                        segment=segment,
                        status="completed" if output_video and status == "completed" else status,
                        output_video=output_video,
                        error=extract_error_message(submission_response),
                        notes="Submitted to async video provider with segment prompt and reference images.",
                        task_id=task_id,
                    )
                )

            episodes.append(build_episode_result(config=config, episode=episode, segments=segment_results))

    if settings.auto_poll:
        pending_task_ids = build_pending_segment_ids(episodes)
        for poll_index in range(settings.max_polls):
            if not pending_task_ids:
                break
            poll_batch: list[dict[str, Any]] = []
            for item in episodes:
                for segment in item.get("segments", []):
                    if segment["status"] not in NON_TERMINAL_STATUSES or not segment["task_id"]:
                        continue
                    status_response = request_json(
                        method="GET",
                        url=join_url(settings.api_base, settings.status_path.format(task_id=segment["task_id"])),
                        api_key=settings.api_key,
                        auth_scheme=settings.auth_scheme,
                        payload=None,
                        timeout_seconds=settings.timeout_seconds,
                    )
                    poll_batch.append(
                        {
                            "episode_number": item["episode_number"],
                            "segment_id": segment["segment_id"],
                            "response": status_response,
                        }
                    )
                    segment["status"] = map_render_status(extract_status(status_response) or segment["status"])
                    segment["output_video"] = extract_output_video(status_response) or segment["output_video"]
                    segment["error"] = extract_error_message(status_response) or segment["error"]
                    segment["notes"] = update_notes(segment["notes"], status_response)
                update_episode_from_segments(item)
            provider_response["polls"].append({"poll_index": poll_index + 1, "segments": poll_batch})
            pending_task_ids = build_pending_segment_ids(episodes)
            if pending_task_ids and poll_index + 1 < settings.max_polls:
                time.sleep(settings.poll_seconds)

    if any(item["status"] in NON_TERMINAL_STATUSES for item in episodes):
        for item in episodes:
            if item["status"] in NON_TERMINAL_STATUSES:
                item["notes"] = append_note(
                    item["notes"],
                    "Still waiting on async segment generation. Re-run `render-videos` to continue polling.",
                )

    for item in episodes:
        if item["status"] == "completed" and not item["output_video"]:
            item["notes"] = append_note(
                item["notes"],
                "All segment renders completed. Stitch the segment clips downstream to create the final assembled video.",
            )

    response = {
        "provider": settings.provider_label,
        "run_label": f"{settings.model}@{int(time.time())}",
        "episodes": episodes,
    }
    return provider_response, response


NON_TERMINAL_STATUSES = {"submitted", "queued", "in_progress"}


def build_render_submission_payload(
    project_root: Path,
    config: dict[str, Any],
    episode: dict[str, Any],
    segment: dict[str, Any],
    settings: RenderProviderSettings,
) -> dict[str, Any]:
    prompt_lines = [
        f"Segment: {segment['segment_id']} ({segment['duration_seconds']}s)",
        f"Purpose: {segment['purpose']}",
        f"Coverage: {segment['coverage']}",
        "",
        segment["timeline_prompt"],
        "",
        f"Sound design: {segment['sound_design']}",
        f"End frame: {segment['end_frame']}",
    ]
    if segment.get("continuity_to_next"):
        prompt_lines.append(f"Continuity to next: {segment['continuity_to_next']}")
    prompt = "\n".join(line for line in prompt_lines if line).strip()
    image_urls = collect_render_image_urls(project_root, segment.get("resolved_inputs", []), max_images=settings.max_images)

    payload: dict[str, Any] = {
        "model": settings.model,
        "aspect_ratio": config["aspect_ratio"],
        "duration": str(segment["duration_seconds"]),
        "generate_audio": settings.generate_audio,
    }
    if settings.resolution:
        payload["resolution"] = settings.resolution
    if settings.service_tier:
        payload["service_tier"] = settings.service_tier

    if settings.request_format == "content":
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        payload["content"] = content
    else:
        payload["prompt"] = prompt
        if image_urls:
            payload["image_urls"] = image_urls
    return payload


def collect_render_image_urls(project_root: Path, resolved_inputs: list[dict[str, Any]], *, max_images: int) -> list[str]:
    urls: list[str] = []
    for input_item in resolved_inputs:
        for source_file in input_item["source_files"]:
            resolved_source = resolve_render_media_source(
                project_root,
                source_file.get("resolved_source") or source_file.get("source", ""),
            )

            mime_type = guess_media_type(resolved_source)
            if not mime_type.startswith("image/"):
                continue
            try:
                url = attachment_to_url(
                    ProviderAttachment(
                        kind="image",
                        source=resolved_source,
                        name=Path(source_file.get("source", resolved_source)).name,
                        mime_type=mime_type,
                    )
                )
            except ProviderError:
                continue
            urls.append(url)
            if len(urls) >= max_images:
                return urls
    return urls


def build_skipped_episode_result(config: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    skipped_segments = [
        build_segment_result(
            segment=segment,
            status="skipped",
            output_video="",
            error=" / ".join(episode["blockers"]),
            notes="Skipped because the execution plan is not ready for rendering.",
            task_id="",
        )
        for segment in episode.get("generation_segments", [])
    ]
    return build_episode_result(config=config, episode=episode, segments=skipped_segments)


def build_segment_result(
    *,
    segment: dict[str, Any],
    status: str,
    output_video: str,
    error: str,
    notes: str,
    task_id: str,
) -> dict[str, Any]:
    reference_ids = sorted(
        {
            reference_id
            for item in segment.get("resolved_inputs", [])
            for reference_id in item.get("reference_ids", [])
        }
    )
    material_ids = sorted(
        {
            material_id
            for item in segment.get("resolved_inputs", [])
            for material_id in item.get("material_ids", [])
        }
    )
    return {
        "segment_id": segment["segment_id"],
        "status": status,
        "task_id": task_id,
        "output_video": output_video,
        "duration_seconds": segment["duration_seconds"],
        "used_reference_ids": reference_ids,
        "used_material_ids": material_ids,
        "notes": notes,
        "error": error,
    }


def build_episode_result(
    *,
    config: dict[str, Any],
    episode: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_ids = sorted(
        {
            reference_id
            for item in segments
            for reference_id in item.get("used_reference_ids", [])
        }
    )
    material_ids = sorted(
        {
            material_id
            for item in segments
            for material_id in item.get("used_material_ids", [])
        }
    )
    status = summarize_episode_status(segments)
    output_video = summarize_episode_output_video(segments)
    return {
        "episode_number": episode["episode_number"],
        "status": status,
        "task_id": "",
        "output_video": output_video,
        "duration_seconds": sum(int(item.get("duration_seconds", 0)) for item in segments) or config["episode_duration_seconds"],
        "used_reference_ids": reference_ids,
        "used_material_ids": material_ids,
        "notes": summarize_episode_notes(segments),
        "error": summarize_episode_error(segments),
        "segments": segments,
    }


def summarize_episode_status(segments: list[dict[str, Any]]) -> str:
    statuses = [item.get("status", "") for item in segments]
    if not statuses:
        return "failed"
    if any(status in NON_TERMINAL_STATUSES for status in statuses):
        return "in_progress"
    if any(status == "failed" for status in statuses):
        return "failed"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    if all(status == "completed" for status in statuses):
        return "completed"
    if all(status in {"completed", "skipped"} for status in statuses):
        return "completed"
    return "submitted"


def summarize_episode_output_video(segments: list[dict[str, Any]]) -> str:
    completed_outputs = [item.get("output_video", "") for item in segments if item.get("status") == "completed" and item.get("output_video")]
    if len(completed_outputs) == 1:
        return completed_outputs[0]
    return ""


def summarize_episode_error(segments: list[dict[str, Any]]) -> str:
    errors = [item.get("error", "") for item in segments if item.get("error")]
    return " / ".join(dict.fromkeys(errors))


def summarize_episode_notes(segments: list[dict[str, Any]]) -> str:
    notes = [item.get("notes", "") for item in segments if item.get("notes")]
    return " ".join(dict.fromkeys(notes))


def update_episode_from_segments(episode: dict[str, Any]) -> None:
    segments = episode.get("segments", [])
    episode["status"] = summarize_episode_status(segments)
    episode["output_video"] = summarize_episode_output_video(segments)
    episode["error"] = summarize_episode_error(segments)
    episode["notes"] = summarize_episode_notes(segments)
    episode["used_reference_ids"] = sorted(
        {reference_id for item in segments for reference_id in item.get("used_reference_ids", [])}
    )
    episode["used_material_ids"] = sorted(
        {material_id for item in segments for material_id in item.get("used_material_ids", [])}
    )
    episode["duration_seconds"] = sum(int(item.get("duration_seconds", 0)) for item in segments) or episode.get("duration_seconds", 0)


def build_pending_segment_ids(episodes: list[dict[str, Any]]) -> list[str]:
    return [
        segment["task_id"]
        for item in episodes
        for segment in item.get("segments", [])
        if segment.get("status") in NON_TERMINAL_STATUSES and segment.get("task_id")
    ]


def payload_has_segment_results(payload: dict[str, Any]) -> bool:
    episodes = payload.get("episodes", [])
    return any(isinstance(item, dict) and isinstance(item.get("segments"), list) for item in episodes)


def has_pending_tasks(payload: dict[str, Any]) -> bool:
    episodes = payload.get("episodes", [])
    for item in episodes:
        if not isinstance(item, dict):
            continue
        if item.get("status") in NON_TERMINAL_STATUSES and item.get("task_id"):
            return True
        for segment in item.get("segments", []):
            if isinstance(segment, dict) and segment.get("status") in NON_TERMINAL_STATUSES and segment.get("task_id"):
                return True
    return False


def request_json(
    *,
    method: str,
    url: str,
    api_key: str,
    auth_scheme: str,
    payload: dict[str, Any] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": api_key if auth_scheme == "raw" else f"Bearer {api_key}",
        },
        method=method,
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"Render provider returned HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise ProviderError(f"Render provider request failed: {exc.reason}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("Render provider response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ProviderError("Render provider response must be a JSON object.")
    return data


def join_url(api_base: str, path: str) -> str:
    base = api_base.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def extract_task_id(data: dict[str, Any]) -> str:
    return str(first_candidate(data, ["task_id", "id"]) or first_candidate(data.get("data"), ["task_id", "id"]) or "")


def extract_status(data: dict[str, Any]) -> str:
    return str(
        first_candidate(data, ["status", "state"])
        or first_candidate(data.get("data"), ["status", "state"])
        or ""
    )


def extract_output_video(data: dict[str, Any]) -> str:
    candidates = [
        first_candidate(data, ["output_video", "video_url", "url"]),
        first_candidate(data.get("data"), ["output_video", "video_url", "url"]),
    ]
    output = next((item for item in candidates if isinstance(item, str) and item.strip()), "")
    if output:
        return output
    data_block = data.get("data")
    if isinstance(data_block, dict):
        video = data_block.get("video")
        if isinstance(video, dict):
            nested = first_candidate(video, ["url", "video_url", "output_video"])
            if isinstance(nested, str):
                return nested
    return ""


def extract_error_message(data: dict[str, Any]) -> str:
    error_block = data.get("error")
    if isinstance(error_block, dict):
        message = error_block.get("message")
        if isinstance(message, str):
            return message
    message = first_candidate(data, ["message", "error_message"])
    return message if isinstance(message, str) else ""


def first_candidate(data: Any, keys: list[str]) -> Any:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value not in {None, ""}:
            return value
    return None


def map_render_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"completed", "succeeded", "success", "done", "finished"}:
        return "completed"
    if normalized in {"queued", "pending", "submitted"}:
        return "queued"
    if normalized in {"processing", "running", "in_progress", "generating"}:
        return "in_progress"
    if normalized in {"failed", "error", "cancelled", "canceled"}:
        return "failed"
    return "submitted"


def update_notes(existing: str, status_response: dict[str, Any]) -> str:
    status = extract_status(status_response)
    if not status:
        return existing
    return append_note(existing, f"Provider status: {status}")


def append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing} {note}"

def resolve_render_media_source(project_root: Path, source: str) -> str:
    if source.startswith(("http://", "https://")):
        return source
    path = Path(source)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists() or not path.is_file():
        raise ProviderError(f"Render attachment file was not found: {path}")
    return str(path)


def guess_media_type(source: str) -> str:
    mime_type, _ = mimetypes.guess_type(source)
    return mime_type or "application/octet-stream"


def env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ProviderError(f"{name} must be an integer.") from exc
