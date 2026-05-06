"""Project source-input helpers for videos, references, and material libraries."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import urlparse
from typing import Any


SOURCE_VIDEO_FILE = "source_video.json"
MATERIAL_LIBRARY_FILE = "material_library.json"


class SourceInputError(RuntimeError):
    """Raised when source inputs are missing or malformed."""


def default_source_video_manifest(source_video: str) -> dict[str, Any]:
    return {
        "source": source_video,
        "title": "",
        "analysis_notes": "",
        "transcript_excerpt": "",
        "key_moments": [],
        "frame_reference_images": [],
    }


def default_material_library_manifest() -> dict[str, Any]:
    return {"materials": []}


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _require_dict(data: Any, path: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SourceInputError(f"{path} must be an object.")
    return data


def _require_list(data: Any, path: str) -> list[Any]:
    if not isinstance(data, list):
        raise SourceInputError(f"{path} must be a list.")
    return data


def _require_str(data: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(data, str):
        raise SourceInputError(f"{path} must be a string.")
    value = data.strip()
    if not allow_empty and not value:
        raise SourceInputError(f"{path} must be a non-empty string.")
    return value


def _require_str_list(data: Any, path: str) -> list[str]:
    items = _require_list(data, path)
    return [_require_str(item, f"{path}[{index}]") for index, item in enumerate(items)]


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_source(project_root: Path, source: str, path: str) -> dict[str, Any]:
    if is_url(source):
        mime_type, _ = mimetypes.guess_type(source)
        return {
            "source": source,
            "resolved_source": source,
            "source_kind": "remote_url",
            "file_name": Path(urlparse(source).path).name or source,
            "media_type": mime_type or "application/octet-stream",
            "size_bytes": None,
        }

    local_path = Path(source)
    if not local_path.is_absolute():
        local_path = (project_root / local_path).resolve()
    else:
        local_path = local_path.resolve()

    if not local_path.exists():
        raise SourceInputError(f"{path} points to a missing local file: {local_path}")
    if not local_path.is_file():
        raise SourceInputError(f"{path} must point to a file: {local_path}")

    mime_type, _ = mimetypes.guess_type(local_path.name)
    return {
        "source": source,
        "resolved_source": str(local_path),
        "source_kind": "local_file",
        "file_name": local_path.name,
        "media_type": mime_type or "application/octet-stream",
        "size_bytes": local_path.stat().st_size,
    }


def load_source_video_input(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = project_root / "source" / SOURCE_VIDEO_FILE
    if path.exists():
        raw = _require_dict(_read_json(path), "source_video")
    else:
        raw = default_source_video_manifest(config["source_video"])

    source = _require_str(raw.get("source") or config["source_video"], "source_video.source")
    resolved = _resolve_source(project_root, source, "source_video.source")
    frame_reference_images = _require_str_list(
        raw.get("frame_reference_images", []), "source_video.frame_reference_images"
    )

    normalized_frames: list[dict[str, Any]] = []
    for index, item in enumerate(frame_reference_images):
        resolved_frame = _resolve_source(project_root, item, f"source_video.frame_reference_images[{index}]")
        if not resolved_frame["media_type"].startswith("image/"):
            raise SourceInputError(
                f"source_video.frame_reference_images[{index}] must be an image file or URL, got {resolved_frame['media_type']}."
            )
        normalized_frames.append(resolved_frame)

    media_type = resolved["media_type"]
    if not media_type.startswith("video/") and resolved["source_kind"] == "local_file":
        raise SourceInputError(f"source_video.source must be a video file, got {media_type}.")

    return {
        **resolved,
        "title": _require_str(raw.get("title", ""), "source_video.title", allow_empty=True),
        "analysis_notes": _require_str(
            raw.get("analysis_notes", ""), "source_video.analysis_notes", allow_empty=True
        ),
        "transcript_excerpt": _require_str(
            raw.get("transcript_excerpt", ""), "source_video.transcript_excerpt", allow_empty=True
        ),
        "key_moments": _require_str_list(raw.get("key_moments", []), "source_video.key_moments"),
        "frame_reference_images": normalized_frames,
    }


def load_material_library(project_root: Path) -> dict[str, Any]:
    path = project_root / "source" / MATERIAL_LIBRARY_FILE
    if not path.exists():
        return default_material_library_manifest()

    raw = _require_dict(_read_json(path), "material_library")
    materials = _require_list(raw.get("materials", []), "material_library.materials")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(materials):
        material = _require_dict(item, f"material_library.materials[{index}]")
        resolved = _resolve_source(
            project_root,
            _require_str(material.get("source"), f"material_library.materials[{index}].source"),
            f"material_library.materials[{index}].source",
        )
        declared_media_type = _require_str(
            material.get("media_type"), f"material_library.materials[{index}].media_type"
        )
        if not resolved["media_type"].startswith(declared_media_type):
            if declared_media_type not in {"image", "video", "audio", "application"}:
                raise SourceInputError(
                    f"material_library.materials[{index}].media_type must start with image/video/audio/application."
                )
            if not resolved["media_type"].startswith(f"{declared_media_type}/"):
                raise SourceInputError(
                    f"material_library.materials[{index}] media_type mismatch: declared {declared_media_type}, actual {resolved['media_type']}."
                )
        normalized.append(
            {
                "material_id": _require_str(
                    material.get("material_id"), f"material_library.materials[{index}].material_id"
                ),
                "name": _require_str(material.get("name"), f"material_library.materials[{index}].name"),
                "category": _require_str(material.get("category"), f"material_library.materials[{index}].category"),
                "media_type": declared_media_type,
                "resolved_media_type": resolved["media_type"],
                "source": resolved["source"],
                "resolved_source": resolved["resolved_source"],
                "source_kind": resolved["source_kind"],
                "file_name": resolved["file_name"],
                "usage_notes": _require_str(
                    material.get("usage_notes"), f"material_library.materials[{index}].usage_notes"
                ),
                "tags": _require_str_list(material.get("tags", []), f"material_library.materials[{index}].tags"),
                "available_for": _require_str_list(
                    material.get("available_for", []), f"material_library.materials[{index}].available_for"
                ),
                "linked_reference_ids": _require_str_list(
                    material.get("linked_reference_ids", []),
                    f"material_library.materials[{index}].linked_reference_ids",
                ),
            }
        )
    return {"materials": normalized}


def render_source_video_summary(source_video: dict[str, Any]) -> str:
    lines = [
        f"- Source kind: {source_video['source_kind']}",
        f"- Source: {source_video['source']}",
        f"- File name: {source_video['file_name']}",
        f"- Media type: {source_video['media_type']}",
    ]
    if source_video["size_bytes"] is not None:
        lines.append(f"- Size bytes: {source_video['size_bytes']}")
    if source_video["title"]:
        lines.append(f"- Title hint: {source_video['title']}")
    if source_video["analysis_notes"]:
        lines.append(f"- Analysis notes: {source_video['analysis_notes']}")
    if source_video["transcript_excerpt"]:
        lines.append(f"- Transcript excerpt: {source_video['transcript_excerpt']}")
    if source_video["key_moments"]:
        lines.append(f"- Key moments: {' / '.join(source_video['key_moments'])}")
    if source_video["frame_reference_images"]:
        lines.append(
            "- Frame references: "
            + " / ".join(frame["file_name"] for frame in source_video["frame_reference_images"])
        )
    else:
        lines.append("- Frame references: none")
    return "\n".join(lines)


def render_material_library_summary(material_library: dict[str, Any]) -> str:
    materials = material_library["materials"]
    if not materials:
        return "- No reusable materials were registered in source/material_library.json."
    lines = []
    for item in materials:
        tags = " / ".join(item["tags"]) if item["tags"] else "none"
        available_for = " / ".join(item["available_for"]) if item["available_for"] else "all stages"
        lines.append(
            f"- {item['material_id']} | {item['category']} | {item['media_type']} | {item['name']} | "
            f"{item['source_kind']} -> {item['source']} | tags: {tags} | stages: {available_for} | notes: {item['usage_notes']}"
        )
    return "\n".join(lines)
