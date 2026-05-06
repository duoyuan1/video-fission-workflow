"""Stage-specific logic for the first three workflow nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow.constants import STAGE_BY_COMMAND, STAGE_BY_NAME
from workflow.provider import ProviderAttachment
from workflow.source_inputs import (
    SourceInputError,
    load_material_library,
    load_source_video_input,
    render_material_library_summary,
    render_source_video_summary,
)
from workflow.schemas import StageArtifact, utc_now


class NodeValidationError(RuntimeError):
    """Raised when imported model output does not match the expected shape."""


@dataclass(frozen=True)
class PreparedRequest:
    prompt: str
    response_template: dict[str, Any]
    summary: str


def _require_dict(data: Any, path: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise NodeValidationError(f"{path} must be an object.")
    return data


def _require_list(data: Any, path: str) -> list[Any]:
    if not isinstance(data, list):
        raise NodeValidationError(f"{path} must be a list.")
    return data


def _require_str(data: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(data, str):
        raise NodeValidationError(f"{path} must be a string.")
    value = data.strip()
    if not allow_empty and not value:
        raise NodeValidationError(f"{path} must be a non-empty string.")
    return value


def _require_int(data: Any, path: str) -> int:
    if not isinstance(data, int):
        raise NodeValidationError(f"{path} must be an integer.")
    return data


def _require_bool(data: Any, path: str) -> bool:
    if not isinstance(data, bool):
        raise NodeValidationError(f"{path} must be a boolean.")
    return data


def _require_str_list(data: Any, path: str) -> list[str]:
    items = _require_list(data, path)
    return [_require_str(item, f"{path}[{index}]") for index, item in enumerate(items)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _stage_output_root(project_root: Path, command: str) -> Path:
    stage = STAGE_BY_COMMAND[command]
    return project_root / "outputs" / stage.output_dir


def prepare_request(project_root: Path, command: str, config: dict[str, Any], direction_id: str | None = None) -> tuple[Path, Path]:
    if command == "analyze":
        source_video = load_source_video_input(project_root, config)
        prepared = prepare_video_analysis(config, source_video)
    elif command == "plan-directions":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        prepared = prepare_direction_planning(config, analysis, reference_library, material_library)
    elif command == "gen-script":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        directions = load_artifact_payload(project_root, "direction_planner")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        prepared = prepare_script_generation(
            config, analysis, directions, reference_library, material_library, direction_id=direction_id
        )
    elif command == "gen-assets":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        script = load_artifact_payload(project_root, "script_generator")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        prepared = prepare_asset_planning(config, analysis, script, reference_library, material_library)
    elif command == "gen-storyboards":
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        prepared = prepare_storyboard_generation(config, script, assets, reference_library, material_library)
    elif command == "qa":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        directions = load_artifact_payload(project_root, "direction_planner")
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        storyboards = load_artifact_payload(project_root, "storyboard_generator")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        prepared = prepare_qa_review(
            config, analysis, directions, script, assets, storyboards, reference_library, material_library
        )
    elif command == "render-videos":
        execution_plan = load_artifact_payload(project_root, "execution_planner")
        prepared = prepare_video_rendering(config, execution_plan)
    else:
        raise NodeValidationError(f"Stage `{command}` does not support request preparation.")

    output_root = _stage_output_root(project_root, command)
    prompt_path = output_root / "request.md"
    template_path = output_root / "response_template.json"
    prompt_path.write_text(prepared.prompt, encoding="utf-8")
    template_path.write_text(_render_json(prepared.response_template) + "\n", encoding="utf-8")
    return prompt_path, template_path


def finalize_response(
    project_root: Path,
    command: str,
    config: dict[str, Any],
    response: dict[str, Any],
    direction_id: str | None = None,
) -> tuple[StageArtifact, str]:
    if command == "analyze":
        return finalize_video_analysis(config, response)
    if command == "plan-directions":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        return finalize_direction_planning(config, analysis, response)
    if command == "gen-script":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        directions = load_artifact_payload(project_root, "direction_planner")
        return finalize_script_generation(config, analysis, directions, response, direction_id=direction_id)
    if command == "gen-assets":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        script = load_artifact_payload(project_root, "script_generator")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        return finalize_asset_planning(config, analysis, script, response, reference_library, material_library)
    if command == "gen-storyboards":
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        return finalize_storyboard_generation(config, script, assets, response, reference_library, material_library)
    if command == "qa":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        directions = load_artifact_payload(project_root, "direction_planner")
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        storyboards = load_artifact_payload(project_root, "storyboard_generator")
        reference_library = load_reference_library(project_root)
        material_library = load_material_library(project_root)
        return finalize_qa_review(
            config, analysis, directions, script, assets, storyboards, response, reference_library, material_library
        )
    if command == "render-videos":
        execution_plan = load_artifact_payload(project_root, "execution_planner")
        return finalize_video_rendering(config, execution_plan, response)
    raise NodeValidationError(f"Stage `{command}` does not support response finalization.")


def load_artifact_payload(project_root: Path, stage_name: str) -> dict[str, Any]:
    stage = STAGE_BY_NAME[stage_name]
    artifact_path = project_root / "outputs" / stage.output_dir / stage.json_filename
    if not artifact_path.exists():
        raise NodeValidationError(f"Required artifact not found: {artifact_path}")
    artifact = _read_json(artifact_path)
    return _require_dict(artifact.get("payload"), f"{stage_name}.payload")


def load_reference_library(project_root: Path) -> dict[str, Any]:
    path = project_root / "source" / "reference_assets.json"
    if not path.exists():
        return {"references": []}

    data = _require_dict(_read_json(path), "reference_assets")
    references = _require_list(data.get("references", []), "reference_assets.references")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(references):
        reference = _require_dict(item, f"reference_assets.references[{index}]")
        normalized.append(
            {
                "reference_id": _require_str(
                    reference.get("reference_id"), f"reference_assets.references[{index}].reference_id"
                ),
                "name": _require_str(reference.get("name"), f"reference_assets.references[{index}].name"),
                "category": _require_str(reference.get("category"), f"reference_assets.references[{index}].category"),
                "source_type": _require_str(
                    reference.get("source_type"), f"reference_assets.references[{index}].source_type"
                ),
                "source": _require_str(reference.get("source"), f"reference_assets.references[{index}].source"),
                "resolved_source": _resolve_local_media_source(
                    project_root,
                    _require_str(reference.get("source"), f"reference_assets.references[{index}].source"),
                ),
                "file_name": Path(
                    _require_str(reference.get("source"), f"reference_assets.references[{index}].source")
                ).name,
                "media_type": _guess_media_type(
                    _require_str(reference.get("source"), f"reference_assets.references[{index}].source")
                ),
                "usage_notes": _require_str(
                    reference.get("usage_notes"), f"reference_assets.references[{index}].usage_notes"
                ),
                "must_keep": _require_str_list(
                    reference.get("must_keep", []), f"reference_assets.references[{index}].must_keep"
                ),
            }
        )
    return {"references": normalized}


def render_reference_library_summary(reference_library: dict[str, Any]) -> str:
    references = reference_library["references"]
    if not references:
        return "- No external reference assets were registered for this project."
    lines = []
    for item in references:
        must_keep = " / ".join(item["must_keep"]) if item["must_keep"] else "None"
        lines.append(
            f"- {item['reference_id']} | {item['category']} | {item['name']} | "
            f"{item['source_type']} -> {item['source']} | keep: {must_keep} | notes: {item['usage_notes']}"
        )
    return "\n".join(lines)


def build_provider_attachments(project_root: Path, command: str, config: dict[str, Any]) -> list[ProviderAttachment]:
    attachments: list[ProviderAttachment] = []
    max_inline_video_bytes = 10 * 1024 * 1024

    def append_image(source: str, file_name: str, media_type: str) -> None:
        if len(attachments) >= 8:
            return
        if not media_type.startswith("image/"):
            return
        attachments.append(
            ProviderAttachment(
                kind="image",
                source=source,
                name=file_name,
                mime_type=media_type,
            )
        )

    def append_video(source: str, file_name: str, media_type: str, *, size_bytes: int | None) -> None:
        if len(attachments) >= 8:
            return
        if not media_type.startswith("video/"):
            return
        if source.startswith(("http://", "https://")):
            attachments.append(
                ProviderAttachment(
                    kind="video",
                    source=source,
                    name=file_name,
                    mime_type=media_type,
                )
            )
            return
        if size_bytes is None or size_bytes > max_inline_video_bytes:
            return
        attachments.append(
            ProviderAttachment(
                kind="video",
                source=source,
                name=file_name,
                mime_type=media_type,
            )
        )

    if command == "analyze":
        source_video = load_source_video_input(project_root, config)
        append_video(
            source_video["resolved_source"],
            source_video["file_name"],
            source_video["media_type"],
            size_bytes=source_video["size_bytes"],
        )
        for item in source_video["frame_reference_images"]:
            append_image(item["resolved_source"], item["file_name"], item["media_type"])
        return attachments

    if command in {"gen-assets", "gen-storyboards"}:
        # These stages now rely on the prompt's structured asset/material summaries only.
        # Keeping them text-only makes them compatible with providers such as DeepSeek
        # official chat/completions, which do not currently accept image_url content items
        # for this workflow's request format.
        return attachments
    return attachments


def _guess_media_type(source: str) -> str:
    suffix = Path(source).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def _resolve_local_media_source(project_root: Path, source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        return source
    path = Path(source)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise SourceInputError(f"Missing local media source: {path}")
    if not path.is_file():
        raise SourceInputError(f"Media source must be a file: {path}")
    return str(path)


def prepare_video_analysis(config: dict[str, Any], source_video: dict[str, Any]) -> PreparedRequest:
    source_video_summary = render_source_video_summary(source_video)
    prompt = f"""# Video Analysis Request

你是“视频内容理解与结构化拆解”专家。你的任务不是改写，不是创作新剧情，也不是提前做裂变策划，而是先忠实理解输入视频，为后续“方向规划 -> 裂变剧本 -> 素材库 -> 分镜”提供干净的事实层基础。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Target platform: `{config["platform"]}`
- Target aspect ratio: `{config["aspect_ratio"]}`
- Default episode duration: `{config["episode_duration_seconds"]}` seconds
- Target episode count: `{config["target_episode_count"]}`
- Locale: `{config["locale"]}`

## Source Video Intake

{source_video_summary}

## Your goals

1. 识别视频中的核心叙事：开场、冲突、转折、结局、主题。
2. 提取角色、场景、关键动作、情绪变化、视觉记忆点。
3. 提取对白、旁白、屏幕文字、关键视觉母题等可验证信息。
4. 给后续阶段保留真实边界：哪些事实必须保留，哪些点位存在歧义需要谨慎解释。

## Output requirements

- 严格输出 JSON。
- 保持忠于视频内容，不要提前编造完整新剧本。
- 不要输出传播建议、平台打法、改编方案、recommended use 这类规划内容。
- 角色、场景、镜头节拍尽量具体。
- `hook_score` 使用 1-5 分。
- `factual_constraints` 里只保留 `must_keep` 和 `uncertain_points`。

## JSON shape

请按照 `response_template.json` 的结构填写，并确保所有必填字段都有内容。
"""

    template = {
        "source_overview": {
            "working_title": "",
            "summary": "",
            "story_type": "",
            "estimated_duration_seconds": 0,
            "visual_style": "",
        },
        "core_narrative": {
            "setup": "",
            "conflict": "",
            "turning_point": "",
            "resolution": "",
            "theme": "",
        },
        "characters": [
            {
                "character_id": "CH01",
                "name": "",
                "role": "",
                "visual_traits": [""],
                "motivation": "",
                "key_actions": [""],
            }
        ],
        "scenes": [
            {
                "scene_id": "SC01",
                "name": "",
                "time_of_day": "",
                "location_type": "",
                "description": "",
                "visual_markers": [""],
            }
        ],
        "beats": [
            {
                "beat_id": "B01",
                "time_range": "",
                "summary": "",
                "emotion": "",
                "visual_focus": "",
                "hook_score": 3,
            }
        ],
        "dialogue_audio_cues": [
            {
                "cue_id": "A01",
                "time_range": "",
                "speaker": "",
                "content": "",
                "function": "",
            }
        ],
        "on_screen_text": [
            {
                "text_id": "T01",
                "time_range": "",
                "text": "",
                "placement": "",
                "importance": "",
            }
        ],
        "visual_motifs": [
            {
                "motif": "",
                "evidence": "",
                "importance": "",
            }
        ],
        "factual_constraints": {
            "must_keep": [""],
            "uncertain_points": [""],
        },
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a structured video analysis request.")


def finalize_video_analysis(config: dict[str, Any], response: dict[str, Any]) -> tuple[StageArtifact, str]:
    source_overview = _require_dict(response.get("source_overview"), "source_overview")
    core_narrative = _require_dict(response.get("core_narrative"), "core_narrative")
    characters = _require_list(response.get("characters"), "characters")
    scenes = _require_list(response.get("scenes"), "scenes")
    beats = _require_list(response.get("beats"), "beats")
    dialogue_audio_cues = _require_list(response.get("dialogue_audio_cues", []), "dialogue_audio_cues")
    on_screen_text = _require_list(response.get("on_screen_text", []), "on_screen_text")
    visual_motifs = _require_list(response.get("visual_motifs", []), "visual_motifs")
    factual_constraints = _require_dict(
        response.get("factual_constraints") or response.get("adaptation_facts"), "factual_constraints"
    )

    payload = {
        "source_overview": {
            "working_title": _require_str(source_overview.get("working_title"), "source_overview.working_title"),
            "summary": _require_str(source_overview.get("summary"), "source_overview.summary"),
            "story_type": _require_str(source_overview.get("story_type"), "source_overview.story_type"),
            "estimated_duration_seconds": _require_int(
                source_overview.get("estimated_duration_seconds"), "source_overview.estimated_duration_seconds"
            ),
            "visual_style": _require_str(source_overview.get("visual_style"), "source_overview.visual_style"),
        },
        "core_narrative": {
            "setup": _require_str(core_narrative.get("setup"), "core_narrative.setup"),
            "conflict": _require_str(core_narrative.get("conflict"), "core_narrative.conflict"),
            "turning_point": _require_str(core_narrative.get("turning_point"), "core_narrative.turning_point"),
            "resolution": _require_str(core_narrative.get("resolution"), "core_narrative.resolution"),
            "theme": _require_str(core_narrative.get("theme"), "core_narrative.theme"),
        },
        "characters": [
            {
                "character_id": _require_str(_require_dict(item, f"characters[{index}]").get("character_id"), f"characters[{index}].character_id"),
                "name": _require_str(_require_dict(item, f"characters[{index}]").get("name"), f"characters[{index}].name"),
                "role": _require_str(_require_dict(item, f"characters[{index}]").get("role"), f"characters[{index}].role"),
                "visual_traits": _require_str_list(_require_dict(item, f"characters[{index}]").get("visual_traits"), f"characters[{index}].visual_traits"),
                "motivation": _require_str(_require_dict(item, f"characters[{index}]").get("motivation"), f"characters[{index}].motivation"),
                "key_actions": _require_str_list(_require_dict(item, f"characters[{index}]").get("key_actions"), f"characters[{index}].key_actions"),
            }
            for index, item in enumerate(characters)
        ],
        "scenes": [
            {
                "scene_id": _require_str(_require_dict(item, f"scenes[{index}]").get("scene_id"), f"scenes[{index}].scene_id"),
                "name": _require_str(_require_dict(item, f"scenes[{index}]").get("name"), f"scenes[{index}].name"),
                "time_of_day": _require_str(_require_dict(item, f"scenes[{index}]").get("time_of_day"), f"scenes[{index}].time_of_day"),
                "location_type": _require_str(_require_dict(item, f"scenes[{index}]").get("location_type"), f"scenes[{index}].location_type"),
                "description": _require_str(_require_dict(item, f"scenes[{index}]").get("description"), f"scenes[{index}].description"),
                "visual_markers": _require_str_list(_require_dict(item, f"scenes[{index}]").get("visual_markers"), f"scenes[{index}].visual_markers"),
            }
            for index, item in enumerate(scenes)
        ],
        "beats": [
            {
                "beat_id": _require_str(_require_dict(item, f"beats[{index}]").get("beat_id"), f"beats[{index}].beat_id"),
                "time_range": _require_str(_require_dict(item, f"beats[{index}]").get("time_range"), f"beats[{index}].time_range"),
                "summary": _require_str(_require_dict(item, f"beats[{index}]").get("summary"), f"beats[{index}].summary"),
                "emotion": _require_str(_require_dict(item, f"beats[{index}]").get("emotion"), f"beats[{index}].emotion"),
                "visual_focus": _require_str(_require_dict(item, f"beats[{index}]").get("visual_focus"), f"beats[{index}].visual_focus"),
                "hook_score": _require_int(_require_dict(item, f"beats[{index}]").get("hook_score"), f"beats[{index}].hook_score"),
            }
            for index, item in enumerate(beats)
        ],
        "dialogue_audio_cues": [
            {
                "cue_id": _require_str(
                    _require_dict(item, f"dialogue_audio_cues[{index}]").get("cue_id"),
                    f"dialogue_audio_cues[{index}].cue_id",
                ),
                "time_range": _require_str(
                    _require_dict(item, f"dialogue_audio_cues[{index}]").get("time_range"),
                    f"dialogue_audio_cues[{index}].time_range",
                ),
                "speaker": _require_str(
                    _require_dict(item, f"dialogue_audio_cues[{index}]").get("speaker"),
                    f"dialogue_audio_cues[{index}].speaker",
                ),
                "content": _require_str(
                    _require_dict(item, f"dialogue_audio_cues[{index}]").get("content"),
                    f"dialogue_audio_cues[{index}].content",
                ),
                "function": _require_str(
                    _require_dict(item, f"dialogue_audio_cues[{index}]").get("function"),
                    f"dialogue_audio_cues[{index}].function",
                ),
            }
            for index, item in enumerate(dialogue_audio_cues)
        ],
        "on_screen_text": [
            {
                "text_id": _require_str(
                    _require_dict(item, f"on_screen_text[{index}]").get("text_id"),
                    f"on_screen_text[{index}].text_id",
                ),
                "time_range": _require_str(
                    _require_dict(item, f"on_screen_text[{index}]").get("time_range"),
                    f"on_screen_text[{index}].time_range",
                ),
                "text": _require_str(
                    _require_dict(item, f"on_screen_text[{index}]").get("text"),
                    f"on_screen_text[{index}].text",
                ),
                "placement": _require_str(
                    _require_dict(item, f"on_screen_text[{index}]").get("placement"),
                    f"on_screen_text[{index}].placement",
                ),
                "importance": _require_str(
                    _require_dict(item, f"on_screen_text[{index}]").get("importance"),
                    f"on_screen_text[{index}].importance",
                ),
            }
            for index, item in enumerate(on_screen_text)
        ],
        "visual_motifs": [
            {
                "motif": _require_str(
                    _require_dict(item, f"visual_motifs[{index}]").get("motif"),
                    f"visual_motifs[{index}].motif",
                ),
                "evidence": _require_str(
                    _require_dict(item, f"visual_motifs[{index}]").get("evidence"),
                    f"visual_motifs[{index}].evidence",
                ),
                "importance": _require_str(
                    _require_dict(item, f"visual_motifs[{index}]").get("importance"),
                    f"visual_motifs[{index}].importance",
                ),
            }
            for index, item in enumerate(visual_motifs)
        ],
        "factual_constraints": {
            "must_keep": _require_str_list(factual_constraints.get("must_keep"), "factual_constraints.must_keep"),
            "uncertain_points": _require_str_list(
                factual_constraints.get("uncertain_points", []), "factual_constraints.uncertain_points"
            ),
        },
    }

    artifact = StageArtifact(
        stage="video_analyzer",
        generated_at=utc_now(),
        status="completed",
        summary="Structured video analysis ready for direction planning.",
        required_inputs=["Source video and project config."],
        expected_outputs=["Normalized content summary for downstream planning."],
        next_stage="direction_planner",
        payload=payload,
    )

    markdown = render_video_analysis_markdown(config, payload)
    return artifact, markdown


def render_video_analysis_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['source_overview']['working_title']} - 视频解析",
        "",
        f"- Project: `{config['project_name']}`",
        f"- Source video: `{config['source_video']}`",
        f"- Story type: {payload['source_overview']['story_type']}",
        f"- Estimated duration: {payload['source_overview']['estimated_duration_seconds']}s",
        f"- Visual style: {payload['source_overview']['visual_style']}",
        "",
        "## 概览",
        "",
        payload["source_overview"]["summary"],
        "",
        "## 核心叙事",
        "",
        f"- 开场：{payload['core_narrative']['setup']}",
        f"- 冲突：{payload['core_narrative']['conflict']}",
        f"- 转折：{payload['core_narrative']['turning_point']}",
        f"- 结局：{payload['core_narrative']['resolution']}",
        f"- 主题：{payload['core_narrative']['theme']}",
        "",
        "## 角色",
        "",
    ]
    for item in payload["characters"]:
        lines.extend(
            [
                f"### {item['character_id']} {item['name']}",
                "",
                f"- 角色功能：{item['role']}",
                f"- 视觉特征：{' / '.join(item['visual_traits'])}",
                f"- 动机：{item['motivation']}",
                f"- 关键动作：{' / '.join(item['key_actions'])}",
                "",
            ]
        )
    lines.extend(["## 场景", ""])
    for item in payload["scenes"]:
        lines.extend(
            [
                f"### {item['scene_id']} {item['name']}",
                "",
                f"- 时段：{item['time_of_day']}",
                f"- 类型：{item['location_type']}",
                f"- 描述：{item['description']}",
                f"- 视觉标记：{' / '.join(item['visual_markers'])}",
                "",
            ]
        )
    lines.extend(["## 节拍", ""])
    for item in payload["beats"]:
        lines.append(
            f"- `{item['beat_id']}` {item['time_range']} | {item['emotion']} | hook={item['hook_score']} | {item['summary']}"
        )
    if payload["dialogue_audio_cues"]:
        lines.extend(["", "## 对白与音频线索", ""])
        for item in payload["dialogue_audio_cues"]:
            lines.append(
                f"- `{item['cue_id']}` {item['time_range']} | {item['speaker']} | {item['content']} | 作用：{item['function']}"
            )
    if payload["on_screen_text"]:
        lines.extend(["", "## 屏幕文字", ""])
        for item in payload["on_screen_text"]:
            lines.append(
                f"- `{item['text_id']}` {item['time_range']} | {item['text']} | 位置：{item['placement']} | 重要性：{item['importance']}"
            )
    if payload["visual_motifs"]:
        lines.extend(["", "## 视觉母题", ""])
        for item in payload["visual_motifs"]:
            lines.append(f"- {item['motif']}：{item['evidence']} | 重要性：{item['importance']}")
    lines.extend(["", "## 事实边界", ""])
    lines.extend(f"- 必须保留：{item}" for item in payload["factual_constraints"]["must_keep"])
    if payload["factual_constraints"]["uncertain_points"]:
        lines.extend(f"- 存在歧义：{item}" for item in payload["factual_constraints"]["uncertain_points"])
    lines.append("")
    return "\n".join(lines)


def analysis_must_keep(analysis: dict[str, Any]) -> list[str]:
    constraints = analysis.get("factual_constraints")
    if isinstance(constraints, dict) and isinstance(constraints.get("must_keep"), list):
        return [str(item).strip() for item in constraints["must_keep"] if str(item).strip()]
    legacy = analysis.get("adaptation_facts")
    if isinstance(legacy, dict) and isinstance(legacy.get("must_keep"), list):
        return [str(item).strip() for item in legacy["must_keep"] if str(item).strip()]
    return []


def analysis_uncertain_points(analysis: dict[str, Any]) -> list[str]:
    constraints = analysis.get("factual_constraints")
    if isinstance(constraints, dict) and isinstance(constraints.get("uncertain_points"), list):
        return [str(item).strip() for item in constraints["uncertain_points"] if str(item).strip()]
    legacy = analysis.get("adaptation_facts")
    if isinstance(legacy, dict):
        combined: list[str] = []
        for key in ("flexible", "avoid"):
            values = legacy.get(key)
            if isinstance(values, list):
                combined.extend(str(item).strip() for item in values if str(item).strip())
        return combined
    return []


def analysis_dialogue_audio_cues(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    cues = analysis.get("dialogue_audio_cues")
    return cues if isinstance(cues, list) else []


def analysis_visual_motifs(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    motifs = analysis.get("visual_motifs")
    return motifs if isinstance(motifs, list) else []


def prepare_direction_planning(
    config: dict[str, Any],
    analysis: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> PreparedRequest:
    must_keep = analysis_must_keep(analysis)
    uncertain_points = analysis_uncertain_points(analysis)
    dialogue_summary = " / ".join(item.get("content", "") for item in analysis_dialogue_audio_cues(analysis)[:4] if item.get("content")) or "none"
    motif_summary = " / ".join(item.get("motif", "") for item in analysis_visual_motifs(analysis)[:4] if item.get("motif")) or "none"
    reference_summary = render_reference_library_summary(reference_library)
    material_summary = render_material_library_summary(material_library)
    prompt = f"""# Direction Planning Request

你是“内容裂变方向规划”专家。现在你已经拿到了原视频的结构化解析，请不要重复做视频理解，而是基于这份解析，提出 3-5 个明确区分的改编方向。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Target platform: `{config["platform"]}`
- Target aspect ratio: `{config["aspect_ratio"]}`
- Recommended total runtime per finished video: 30-40 seconds

## Analysis Summary

- Working title: {analysis["source_overview"]["working_title"]}
- Story type: {analysis["source_overview"]["story_type"]}
- Theme: {analysis["core_narrative"]["theme"]}
- Must keep facts: {' / '.join(must_keep)}
- Ambiguous points: {' / '.join(uncertain_points) if uncertain_points else 'none'}
- Dialogue and audio cues: {dialogue_summary}
- Visual motifs: {motif_summary}

## Available asset context

### External Reference Library
{reference_summary}

### Reusable Material Library
{material_summary}

## Planning goals

1. 给出 3-5 个明显不同的裂变方向，不要只是换同义词。
2. 每个方向都必须是“独立成片的裂变短视频”概念，不是连续剧、不是分集剧情、不是上中下三段。
3. 每个方向都要说明目标受众、情绪曲线、内容钩子、适合的平台打法，并自行从节拍、对白、视觉母题里提炼传播点。
4. 每个方向都要给出一个可以单独成立的视频标题和一个单条视频开头示例，不要写“下一集”“后续”“系列延展”。
5. 默认把最终成片理解为 30-40 秒的独立视频。如果内容较多，可以假设后续执行时会拆成多个连续生成分镜段，但每段最好不超过 10 秒，总时长仍保持在 30-40 秒。
6. 保持与原视频解析一致，不要违背 must_keep 事实，不要为了凑内容新增原视频未出现的核心设定。
7. 如果参考素材里已经有合适的猫狗 meme 角色，可以选择性使用其中 0-2 个作为角色锚点来增强识别度或喜剧感；不需要为了“用素材”而强行同时塞满猫和狗。
8. 如果现有素材库体现出明显的《三角洲行动》战术氛围、工业据点、搜打撤或军事调度气质，应尽量维持这种世界观氛围；但除非剧情自然需要，不要机械塞入具体探员、枪械、配件、载具或地图名。
9. 如果决定使用现有 meme 角色，必须优先从已注册素材里选定具体原型，再在整条方向描述中保持命名稳定。例如一旦选定 `Cheems / Balltze` 或 `Smudge the Cat`，后续都应沿用该名字或明确角色名，不要中途改成别的狗/猫 meme。
10. 如果某个方向明显依赖现成素材，请在 `asset_usage_hint` 中简要说清它会如何选择性使用角色锚点或环境氛围。

## Output requirements

- 严格输出 JSON。
- `score` 使用 1-10 分。
- 必须只有一个 `recommended=true` 的方向。
- `selected_direction_id` 必须指向推荐方向。
"""
    template = {
        "planning_basis": {
            "target_platform": config["platform"],
            "primary_audience": "",
            "overall_strategy": "",
        },
        "directions": [
            {
                "direction_id": "D01",
                "name": "",
                "positioning": "",
                "hook": "",
                "tone": "",
                "audience": "",
                "emotion_curve": [""],
                "differentiators": [""],
                "risks": [""],
                "sample_title": "",
                "sample_opening": "",
                "asset_usage_hint": "",
                "score": 8,
                "recommended": True,
            }
        ],
        "selected_direction_id": "D01",
        "selection_reason": "",
        "global_rules": [""],
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a direction-planning request.")


def finalize_direction_planning(
    config: dict[str, Any], analysis: dict[str, Any], response: dict[str, Any]
) -> tuple[StageArtifact, str]:
    planning_basis = _require_dict(response.get("planning_basis"), "planning_basis")
    directions = _require_list(response.get("directions"), "directions")
    selected_direction_id = _require_str(response.get("selected_direction_id"), "selected_direction_id")
    selection_reason = _require_str(response.get("selection_reason"), "selection_reason")
    global_rules = _require_str_list(response.get("global_rules"), "global_rules")
    must_keep = analysis_must_keep(analysis)

    normalized_directions: list[dict[str, Any]] = []
    recommended_ids: list[str] = []
    for index, item in enumerate(directions):
        direction = _require_dict(item, f"directions[{index}]")
        normalized = {
            "direction_id": _require_str(direction.get("direction_id"), f"directions[{index}].direction_id"),
            "name": _require_str(direction.get("name"), f"directions[{index}].name"),
            "positioning": _require_str(direction.get("positioning"), f"directions[{index}].positioning"),
            "hook": _require_str(direction.get("hook"), f"directions[{index}].hook"),
            "tone": _require_str(direction.get("tone"), f"directions[{index}].tone"),
            "audience": _require_str(direction.get("audience"), f"directions[{index}].audience"),
            "emotion_curve": _require_str_list(direction.get("emotion_curve"), f"directions[{index}].emotion_curve"),
            "differentiators": _require_str_list(direction.get("differentiators"), f"directions[{index}].differentiators"),
            "risks": _require_str_list(direction.get("risks"), f"directions[{index}].risks"),
            "sample_title": _require_str(direction.get("sample_title"), f"directions[{index}].sample_title"),
            "sample_opening": _require_str(direction.get("sample_opening"), f"directions[{index}].sample_opening"),
            "asset_usage_hint": _require_str(direction.get("asset_usage_hint"), f"directions[{index}].asset_usage_hint"),
            "score": _require_int(direction.get("score"), f"directions[{index}].score"),
            "recommended": _require_bool(direction.get("recommended"), f"directions[{index}].recommended"),
        }
        if normalized["recommended"]:
            recommended_ids.append(normalized["direction_id"])
        normalized_directions.append(normalized)

    if len(recommended_ids) != 1:
        raise NodeValidationError("Exactly one direction must have recommended=true.")
    if recommended_ids[0] != selected_direction_id:
        raise NodeValidationError("selected_direction_id must match the recommended direction.")

    payload = {
        "planning_basis": {
            "target_platform": _require_str(planning_basis.get("target_platform"), "planning_basis.target_platform"),
            "primary_audience": _require_str(planning_basis.get("primary_audience"), "planning_basis.primary_audience"),
            "overall_strategy": _require_str(planning_basis.get("overall_strategy"), "planning_basis.overall_strategy"),
        },
        "analysis_anchor": {
            "working_title": analysis["source_overview"]["working_title"],
            "must_keep": must_keep,
        },
        "directions": normalized_directions,
        "selected_direction_id": selected_direction_id,
        "selection_reason": selection_reason,
        "global_rules": global_rules,
    }

    artifact = StageArtifact(
        stage="direction_planner",
        generated_at=utc_now(),
        status="completed",
        summary="Direction planning completed and one裂变方向 selected for script generation.",
        required_inputs=["Completed video analysis artifact."],
        expected_outputs=["Recommended content direction and guardrails."],
        next_stage="script_generator",
        payload=payload,
    )
    markdown = render_direction_planning_markdown(config, payload)
    return artifact, markdown


def render_direction_planning_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {config['project_name']} - 方向规划",
        "",
        f"- 平台：{payload['planning_basis']['target_platform']}",
        f"- 主要受众：{payload['planning_basis']['primary_audience']}",
        f"- 总体策略：{payload['planning_basis']['overall_strategy']}",
        f"- 推荐方向：`{payload['selected_direction_id']}`",
        "",
        "## 全局规则",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["global_rules"])
    lines.extend(["", "## 方向清单", ""])
    for item in payload["directions"]:
        marker = " [Recommended]" if item["direction_id"] == payload["selected_direction_id"] else ""
        lines.extend(
            [
                f"### {item['direction_id']} {item['name']}{marker}",
                "",
                f"- 定位：{item['positioning']}",
                f"- 钩子：{item['hook']}",
                f"- 调性：{item['tone']}",
                f"- 受众：{item['audience']}",
                f"- 情绪曲线：{' -> '.join(item['emotion_curve'])}",
                f"- 差异点：{' / '.join(item['differentiators'])}",
                f"- 风险：{' / '.join(item['risks'])}",
                f"- 示例标题：{item['sample_title']}",
                f"- 示例开头：{item['sample_opening']}",
                f"- 素材使用：{item['asset_usage_hint']}",
                f"- 评分：{item['score']}",
                "",
            ]
        )
    lines.extend(["## 选择理由", "", payload["selection_reason"], ""])
    return "\n".join(lines)


def prepare_script_generation(
    config: dict[str, Any],
    analysis: dict[str, Any],
    directions: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
    direction_id: str | None = None,
) -> PreparedRequest:
    selected = choose_direction(directions, direction_id)
    must_keep = analysis_must_keep(analysis)
    uncertain_points = analysis_uncertain_points(analysis)
    reference_summary = render_reference_library_summary(reference_library)
    material_summary = render_material_library_summary(material_library)
    prompt = f"""# Script Generation Request

你是“裂变短视频编剧”专家。请基于原视频解析和已选方向，只生成 1 条可以独立成片的裂变视频脚本，不要写成系列剧包，不要分集连载。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Aspect ratio: `{config["aspect_ratio"]}`
- Target finished-video runtime: usually 30-40 seconds total
- Preferred generation chunk size: each chunk should stay within 10 seconds

## Analysis Anchor

- Working title: {analysis["source_overview"]["working_title"]}
- Story theme: {analysis["core_narrative"]["theme"]}
- Must keep facts: {' / '.join(must_keep)}
- Ambiguous points: {' / '.join(uncertain_points) if uncertain_points else 'none'}

## Selected Direction

- Direction ID: {selected["direction_id"]}
- Name: {selected["name"]}
- Positioning: {selected["positioning"]}
- Hook: {selected["hook"]}
- Tone: {selected["tone"]}
- Emotion curve: {' -> '.join(selected["emotion_curve"])}
- Direction-level asset usage hint: {selected.get("asset_usage_hint", "none")}

## Available asset context

### External Reference Library
{reference_summary}

### Reusable Material Library
{material_summary}

## Writing requirements

1. 保留 must_keep 事实，不要脱离原视频根基。
2. 对存在歧义的细节可以做保守补全，但不要把歧义点写成长期世界观设定，不要新增原视频未出现的核心道具、机制或规则。
3. 这是“单条独立视频”，不是连续剧。不要出现“下一集”“下次”“下一波”“下一季”等连载化表达。
4. 抖音账号信息与引导页如果使用，只能放在最后收口，不能插入视频中段。
5. 先自行评估这条视频更适合做成多长，总时长通常应保持在 30-40 秒之间。
6. 如果一个完整视频无法在单次生成时稳定完成，可以规划为 2-4 个连续生成分镜段（generation segments）。每个分镜段最好 6-10 秒，且不得超过 10 秒；这些分镜段共享同一套人物、场景、道具与风格设定，合起来仍然是同一条完整视频。
7. 如果背景、场地、时间、叙事阶段或镜头视角发生明显变化，应优先拆成新的 generation segment，而不是强行塞进同一段。
8. 如果后续阶段会提供固定人物/场景/道具素材，请默认这些核心元素在所有 generation segments 中保持一致，不要随意改名、换造型或重设身份。
9. 如果原视频信息不足以支撑复杂扩写，优先围绕同一事件做不同切面强化，而不是凭空发明新剧情。
10. 输出必须同时包含结构化信息和该条视频可读的脚本正文。
11. 正文要写得足够具体，能够直接支撑后续分镜与视频生成。每个 `△` 至少尽量体现以下 3 类信息中的 2 类：画面/景别、角色动作/表情、台词/屏幕文字/音效。
12. 正文参考现有 skill 的格式，至少包含：
   - `第1条`
   - `道具：`
   - `出场人物：`
   - 至少 8 个以 `△ ` 开头的镜头描述
13. 情绪弧线和方向定位要一致，不要写成普通流水账。
14. 选择性使用现有角色/素材。如果素材库里已有合适的猫狗 meme 角色，可把其中 0-2 个作为角色锚点或表情变体；如果当前方向不适合，就不要硬塞。
15. 如果一开始选定了某个 meme 角色原型，必须在整条脚本里锁定这个选择，并明确说明角色是谁。例如可写成“主角使用 Cheems / Balltze meme dog 作为原型”“上司使用 Smudge the Cat meme 作为原型”；后文只允许继续用这个名字或稳定角色名，不要中途换成别的猫狗 meme。
16. 如果现有素材库体现出《三角洲行动》氛围，请优先保持战术办公室、工业据点、空投、搜打撤、军事调度这类整体气质，把场景写成《三角洲行动》式战术工业场景；但除非剧情自然需要，不要强行点名探员、枪械、配件、载具或地图名。
17. 必须输出 `asset_usage_strategy`，写清这条视频实际选用了哪些角色锚点、环境氛围、道具策略，以及哪些素材刻意没有用。

## Fictional reference example

下面是一个完全虚构的写法示例，只用于说明“详细程度”和“分镜拆分原则”，不要照抄其中的人物、设定或剧情：

- 总时长：34 秒
- generation segments：
  - G01 8s：办公室里，上司压低声音下达任务，主角愣住，镜头从中景推到近景。
  - G02 9s：主角冲向窗边，窗外突然出现异常物体，环境从安静转为混乱。
  - G03 9s：异常物体落地引发夸张反应，字幕和音效同时爆发，完成主要笑点。
  - G04 8s：角色收尾吐槽，镜头定格在引导页或最终表情，完成闭环。

这个虚构示例的重点是：
- 总时长保持在 30-40 秒
- 场景或叙事阶段变化时拆成新的 generation segment
- 正文镜头描述要具体到动作、表情、画面和声音，而不是只写一句概括

## Output requirements

- 严格输出 JSON。
- `episodes` 只允许 1 条，用来承载这次生成的单条视频脚本。
- 必须输出 `video_plan`，说明总时长判断以及生成分镜段规划。
- `video_plan.total_duration_seconds` 建议在 30-40 之间。
- `video_plan.generation_segments` 必须有 2-4 段，每段 `duration_seconds` 不得超过 10。
- `script_body` 必须是可直接写入 markdown 的多行文本。
"""

    template = {
        "direction_id": selected["direction_id"],
        "series_title": "",
        "core_hook": "",
        "one_line_sell": "",
        "audience_promise": "",
        "asset_usage_strategy": {
            "character_anchor": "",
            "supporting_references": [""],
            "environment_anchor": "",
            "prop_discipline": "",
            "restraint_notes": "",
        },
        "video_plan": {
            "total_duration_seconds": 32,
            "planning_reason": "",
            "generation_segments": [
                {
                    "segment_id": "G01",
                    "duration_seconds": 8,
                    "purpose": "",
                    "coverage": "",
                }
            ],
        },
        "characters": [
            {
                "name": "",
                "visual_profile": "",
                "identity": "",
                "traits": [""],
                "catchphrase": "",
            }
        ],
        "story_outline": {
            "setup": "",
            "development": "",
            "climax": "",
            "resolution": "",
        },
        "episodes": [
            {
                "episode_number": 1,
                "title": "",
                "purpose": "",
                "emotion": "",
                "summary": "",
                "cliffhanger": "",
                "script_body": "第1条\n1-1 日 内 虚构办公室\n道具：任务纸条、窗边警报灯、掉落物动画、结尾引导牌\n出场人物：上司、执行者\n\n△ 中景，上司坐在办公桌前，压低声音下达任务，气氛紧张。\n△ 近景，执行者先愣一下，再立刻点头，表情从茫然切成认真。\n△ 跟拍镜头，执行者快步冲向窗边，窗外突然闪过异常影子。\n△ 特写，警报灯亮起，环境音由安静切成急促提示音。\n△ 广角，异常物体从天而降，角色抬头后退半步。\n△ 冲击瞬间，字幕与音效一起爆发，形成主要笑点。\n△ 反应镜头，另一个角色吐槽，执行者狼狈收场。\n△ 定格在结尾引导牌或角色最终表情，完成收口。",
            }
        ],
        "production_notes": {
            "visual_emphasis": [""],
            "dialogue_style": "",
            "continuity_rules": [""],
        },
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a standalone fission-video script request.")


def choose_direction(directions: dict[str, Any], direction_id: str | None = None) -> dict[str, Any]:
    available = _require_list(directions.get("directions"), "directions")
    if direction_id is None:
        direction_id = _require_str(directions.get("selected_direction_id"), "selected_direction_id")
    for item in available:
        direction = _require_dict(item, "directions[]")
        if direction.get("direction_id") == direction_id:
            return direction
    raise NodeValidationError(f"Direction `{direction_id}` was not found in directions artifact.")


def finalize_script_generation(
    config: dict[str, Any],
    analysis: dict[str, Any],
    directions: dict[str, Any],
    response: dict[str, Any],
    direction_id: str | None = None,
) -> tuple[StageArtifact, str]:
    selected = choose_direction(directions, direction_id)
    response_direction_id = _require_str(response.get("direction_id"), "direction_id")
    if response_direction_id != selected["direction_id"]:
        raise NodeValidationError("direction_id in response must match the selected direction.")

    characters = _require_list(response.get("characters"), "characters")
    story_outline = _require_dict(response.get("story_outline"), "story_outline")
    episodes = _require_list(response.get("episodes"), "episodes")
    production_notes = _require_dict(response.get("production_notes"), "production_notes")
    video_plan = _require_dict(response.get("video_plan"), "video_plan")
    asset_usage_strategy = _require_dict(response.get("asset_usage_strategy"), "asset_usage_strategy")

    if len(episodes) != 1:
        raise NodeValidationError("episodes must contain exactly 1 item for standalone video generation.")

    total_duration_seconds = _require_int(video_plan.get("total_duration_seconds"), "video_plan.total_duration_seconds")
    if total_duration_seconds < 20 or total_duration_seconds > 45:
        raise NodeValidationError("video_plan.total_duration_seconds must stay in a reasonable 20-45 second range.")
    generation_segments = _require_list(video_plan.get("generation_segments"), "video_plan.generation_segments")
    if not 2 <= len(generation_segments) <= 4:
        raise NodeValidationError("video_plan.generation_segments must contain 2-4 segments.")
    normalized_segments: list[dict[str, Any]] = []
    for index, item in enumerate(generation_segments):
        segment = _require_dict(item, f"video_plan.generation_segments[{index}]")
        duration_seconds = _require_int(
            segment.get("duration_seconds"), f"video_plan.generation_segments[{index}].duration_seconds"
        )
        if duration_seconds > 10:
            raise NodeValidationError("Each generation segment must be 10 seconds or shorter.")
        if duration_seconds <= 0:
            raise NodeValidationError("Each generation segment must have a positive duration.")
        normalized_segments.append(
            {
                "segment_id": _require_str(segment.get("segment_id"), f"video_plan.generation_segments[{index}].segment_id"),
                "duration_seconds": duration_seconds,
                "purpose": _require_str(segment.get("purpose"), f"video_plan.generation_segments[{index}].purpose"),
                "coverage": _require_str(segment.get("coverage"), f"video_plan.generation_segments[{index}].coverage"),
            }
        )

    normalized_episodes: list[dict[str, Any]] = []
    for index, item in enumerate(episodes):
        episode = _require_dict(item, f"episodes[{index}]")
        episode_number = _require_int(episode.get("episode_number"), f"episodes[{index}].episode_number")
        if episode_number != 1:
            raise NodeValidationError("episodes[0].episode_number must be 1 for standalone video generation.")
        script_body = _require_str(episode.get("script_body"), f"episodes[{index}].script_body")
        if "△ " not in script_body:
            raise NodeValidationError(f"episodes[{index}].script_body must include lines starting with `△ `.")
        normalized_episodes.append(
            {
                "episode_number": episode_number,
                "title": _require_str(episode.get("title"), f"episodes[{index}].title"),
                "purpose": _require_str(episode.get("purpose"), f"episodes[{index}].purpose"),
                "emotion": _require_str(episode.get("emotion"), f"episodes[{index}].emotion"),
                "summary": _require_str(episode.get("summary"), f"episodes[{index}].summary"),
                "cliffhanger": _require_str(
                    episode.get("cliffhanger", ""), f"episodes[{index}].cliffhanger", allow_empty=True
                ),
                "script_body": script_body,
            }
        )

    first_video_title = normalized_episodes[0]["title"]
    fallback_series_title = (
        _require_str(response.get("series_title", ""), "series_title", allow_empty=True)
        or first_video_title
        or selected["name"]
        or analysis["source_overview"]["working_title"]
    )

    payload = {
        "direction_id": response_direction_id,
        "direction_name": selected["name"],
        "series_title": fallback_series_title,
        "core_hook": _require_str(response.get("core_hook"), "core_hook"),
        "one_line_sell": _require_str(response.get("one_line_sell"), "one_line_sell"),
        "audience_promise": _require_str(response.get("audience_promise"), "audience_promise"),
        "asset_usage_strategy": {
            "character_anchor": _require_str(asset_usage_strategy.get("character_anchor"), "asset_usage_strategy.character_anchor"),
            "supporting_references": _require_str_list(asset_usage_strategy.get("supporting_references"), "asset_usage_strategy.supporting_references"),
            "environment_anchor": _require_str(asset_usage_strategy.get("environment_anchor"), "asset_usage_strategy.environment_anchor"),
            "prop_discipline": _require_str(asset_usage_strategy.get("prop_discipline"), "asset_usage_strategy.prop_discipline"),
            "restraint_notes": _require_str(asset_usage_strategy.get("restraint_notes"), "asset_usage_strategy.restraint_notes"),
        },
        "video_plan": {
            "total_duration_seconds": total_duration_seconds,
            "planning_reason": _require_str(video_plan.get("planning_reason"), "video_plan.planning_reason"),
            "generation_segments": normalized_segments,
        },
        "analysis_anchor": {
            "working_title": analysis["source_overview"]["working_title"],
            "must_keep": analysis_must_keep(analysis),
        },
        "characters": [
            {
                "name": _require_str(_require_dict(item, f"characters[{index}]").get("name"), f"characters[{index}].name"),
                "visual_profile": _require_str(
                    _require_dict(item, f"characters[{index}]").get("visual_profile"),
                    f"characters[{index}].visual_profile",
                ),
                "identity": _require_str(_require_dict(item, f"characters[{index}]").get("identity"), f"characters[{index}].identity"),
                "traits": _require_str_list(_require_dict(item, f"characters[{index}]").get("traits"), f"characters[{index}].traits"),
                "catchphrase": _require_str(
                    _require_dict(item, f"characters[{index}]").get("catchphrase"),
                    f"characters[{index}].catchphrase",
                ),
            }
            for index, item in enumerate(characters)
        ],
        "story_outline": {
            "setup": _require_str(story_outline.get("setup"), "story_outline.setup"),
            "development": _require_str(story_outline.get("development"), "story_outline.development"),
            "climax": _require_str(story_outline.get("climax"), "story_outline.climax"),
            "resolution": _require_str(story_outline.get("resolution"), "story_outline.resolution"),
        },
        "episodes": normalized_episodes,
        "production_notes": {
            "visual_emphasis": _require_str_list(production_notes.get("visual_emphasis"), "production_notes.visual_emphasis"),
            "dialogue_style": _require_str(production_notes.get("dialogue_style"), "production_notes.dialogue_style"),
            "continuity_rules": _require_str_list(
                production_notes.get("continuity_rules"), "production_notes.continuity_rules"
            ),
        },
    }

    artifact = StageArtifact(
        stage="script_generator",
        generated_at=utc_now(),
        status="completed",
        summary="Standalone fission-video script ready for asset planning.",
        required_inputs=["Completed analysis artifact and selected direction artifact."],
        expected_outputs=["Single-video script package with production-ready markdown."],
        next_stage="asset_planner",
        payload=payload,
    )
    markdown = render_script_markdown(config, payload)
    return artifact, markdown


def render_script_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['series_title']} - 裂变视频脚本",
        "",
        f"- Project: `{config['project_name']}`",
        f"- Direction: `{payload['direction_id']}` {payload['direction_name']}",
        f"- 核心梗：{payload['core_hook']}",
        f"- 一句话卖点：{payload['one_line_sell']}",
        f"- 受众承诺：{payload['audience_promise']}",
        f"- 总时长规划：{payload['video_plan']['total_duration_seconds']}s",
        "",
        "## 素材使用策略",
        "",
        f"- 角色锚点：{payload['asset_usage_strategy']['character_anchor']}",
        f"- 辅助参考：{' / '.join(payload['asset_usage_strategy']['supporting_references'])}",
        f"- 环境锚点：{payload['asset_usage_strategy']['environment_anchor']}",
        f"- 道具策略：{payload['asset_usage_strategy']['prop_discipline']}",
        f"- 克制说明：{payload['asset_usage_strategy']['restraint_notes']}",
        "",
        "## 人物设定",
        "",
    ]
    for item in payload["characters"]:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- 视觉形象：{item['visual_profile']}",
                f"- 身份背景：{item['identity']}",
                f"- 核心标签：{' / '.join(item['traits'])}",
                f"- 金句：{item['catchphrase']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 视频结构",
            "",
            f"- 起：{payload['story_outline']['setup']}",
            f"- 承：{payload['story_outline']['development']}",
            f"- 转/高潮：{payload['story_outline']['climax']}",
            f"- 合：{payload['story_outline']['resolution']}",
            "",
            "## 生成分段规划",
            "",
            f"- 规划理由：{payload['video_plan']['planning_reason']}",
            "",
        ]
    )
    for item in payload["video_plan"]["generation_segments"]:
        lines.extend(
            [
                f"### {item['segment_id']}",
                "",
                f"- 时长：{item['duration_seconds']}s",
                f"- 作用：{item['purpose']}",
                f"- 覆盖内容：{item['coverage']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 生产提示",
            "",
        ]
    )
    lines.extend(f"- 视觉重点：{item}" for item in payload["production_notes"]["visual_emphasis"])
    lines.append(f"- 对白风格：{payload['production_notes']['dialogue_style']}")
    lines.extend(f"- 连贯规则：{item}" for item in payload["production_notes"]["continuity_rules"])
    lines.extend(["", "## 视频正文", ""])
    for item in payload["episodes"]:
        lines.extend(
            [
                f"### V{item['episode_number']:02d} {item['title']}",
                "",
                f"- 作用：{item['purpose']}",
                f"- 情绪：{item['emotion']}",
                f"- 摘要：{item['summary']}",
                f"- 收口：{item['cliffhanger']}",
                "",
                item["script_body"],
                "",
            ]
        )
    return "\n".join(lines)


def prepare_asset_planning(
    config: dict[str, Any],
    analysis: dict[str, Any],
    script: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> PreparedRequest:
    reference_summary = render_reference_library_summary(reference_library)
    material_summary = render_material_library_summary(material_library)
    prompt = f"""# Asset Planning Request

你是“AI 视频素材库规划”专家。现在你已经有视频解析和裂变剧本，请不要再改写剧情，而是把它们转成可执行的素材库规划。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Aspect ratio: `{config["aspect_ratio"]}`

## Analysis Anchor

- Working title: {analysis["source_overview"]["working_title"]}
- Visual style: {analysis["source_overview"]["visual_style"]}
- Must keep facts: {' / '.join(analysis_must_keep(analysis))}

## Script Anchor

- Series title: {script["series_title"]}
- Direction: {script["direction_name"]}
- Visual emphasis: {' / '.join(script["production_notes"]["visual_emphasis"])}
- Continuity rules: {' / '.join(script["production_notes"]["continuity_rules"])}

## External Reference Library

{reference_summary}

## Existing Material Library

{material_summary}

## Planning goals

1. 产出统一风格前缀和素材一致性规则。
2. 将素材拆成角色 C、场景 S、道具 P 三类。
3. 对于用户已提供参考素材的人物、表情包、IP 形象、固定道具，不要把它们当成纯原创资产，必须标注引用方式。
4. 如果现有素材库中已经有可直接复用的图像/视频/道具素材，优先绑定 `material_ids`，不要重复生成。
5. 每项素材都要写明用途、优先级、视觉描述、生成提示词，以及素材来源策略。
6. 给出“必须先生成”的最小素材集，方便开工。

## Output requirements

- 严格输出 JSON。
- `asset_id` 必须使用 `C` / `S` / `P` 前缀。
- `sourcing_mode` 只能是 `generate_fresh` / `generate_from_reference` / `use_reference_directly` / `use_material_library_directly`。
- 如果使用参考素材，`reference_ids` 必须引用 External Reference Library 里的 `reference_id`。
- `material_ids` 只能引用 Existing Material Library 里的 `material_id`。
- 每个 `characters[]` / `scenes[]` / `props[]` 条目都必须把所有字段填满；不要留空字符串，不要省略数组字段。
- `reference_notes`、`visual_description`、`generation_prompt` 都必须是非空字符串。
- 即使 `sourcing_mode = use_material_library_directly`，也必须填写 `generation_prompt`，把它当成“后续执行/复用说明 prompt”，说明这个现有素材该如何被使用、如何与整体风格保持一致。
- `generation_prompt` 用适合图像模型或执行阶段的完整描述，至少要写清主体/场景、风格、用途或使用方式。
- `must_generate_first` 只能引用已定义的素材 ID。
"""
    template = {
        "direction_id": script["direction_id"],
        "style_guide": {
            "visual_style": "",
            "style_prefix": "",
            "palette": [""],
            "rendering_notes": [""],
            "consistency_rules": [""],
        },
        "characters": [
            {
                "asset_id": "C01",
                "name": "",
                "purpose": "",
                "priority": "must",
                "sourcing_mode": "generate_fresh",
                "reference_ids": [],
                "material_ids": [],
                "reference_notes": "",
                "visual_description": "",
                "generation_prompt": "",
            }
        ],
        "scenes": [
            {
                "asset_id": "S01",
                "name": "",
                "purpose": "",
                "priority": "must",
                "sourcing_mode": "generate_fresh",
                "reference_ids": [],
                "material_ids": [],
                "reference_notes": "",
                "visual_description": "",
                "generation_prompt": "",
            }
        ],
        "props": [
            {
                "asset_id": "P01",
                "name": "",
                "purpose": "",
                "priority": "must",
                "sourcing_mode": "generate_fresh",
                "reference_ids": [],
                "material_ids": [],
                "reference_notes": "",
                "visual_description": "",
                "generation_prompt": "",
            }
        ],
        "reuse_plan": [
            {
                "assets": ["C01", "S01"],
                "episode_refs": ["E01", "E02"],
                "note": "",
            }
        ],
        "must_generate_first": ["C01", "S01", "P01"],
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare an asset planning request.")


def finalize_asset_planning(
    config: dict[str, Any],
    analysis: dict[str, Any],
    script: dict[str, Any],
    response: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> tuple[StageArtifact, str]:
    style_guide = _require_dict(response.get("style_guide"), "style_guide")
    characters = _require_list(response.get("characters"), "characters")
    scenes = _require_list(response.get("scenes"), "scenes")
    props = _require_list(response.get("props"), "props")
    reuse_plan = _require_list(response.get("reuse_plan"), "reuse_plan")
    must_generate_first = _require_str_list(response.get("must_generate_first"), "must_generate_first")
    direction_id = _require_str(response.get("direction_id"), "direction_id")
    if direction_id != script["direction_id"]:
        raise NodeValidationError("direction_id in asset response must match script direction_id.")
    available_reference_ids = {item["reference_id"] for item in reference_library["references"]}
    available_material_ids = {item["material_id"] for item in material_library["materials"]}
    allowed_sourcing_modes = {
        "generate_fresh",
        "generate_from_reference",
        "use_reference_directly",
        "use_material_library_directly",
    }

    style_prefix = _require_str(style_guide.get("style_prefix"), "style_guide.style_prefix")
    visual_style = _require_str(style_guide.get("visual_style"), "style_guide.visual_style")

    def fallback_generation_prompt(
        *,
        asset_id: str,
        name: str,
        purpose: str,
        sourcing_mode: str,
        material_ids: list[str],
        reference_notes: str,
        visual_description: str,
    ) -> str:
        material_hint = f" Materials: {', '.join(material_ids)}." if material_ids else ""
        notes_hint = f" Notes: {reference_notes}." if reference_notes else ""
        if sourcing_mode == "use_material_library_directly":
            return (
                f"{style_prefix} | Reuse the existing library asset for `{name}` ({asset_id}) as-is while keeping the "
                f"overall style consistent with {visual_style}. Purpose: {purpose}. Scene/asset description: "
                f"{visual_description}.{material_hint}{notes_hint} Use this asset as the direct base plate/reference in later generation."
            ).strip()
        return (
            f"{style_prefix} | {name} ({asset_id}). Purpose: {purpose}. Visual description: {visual_description}.{notes_hint}"
        ).strip()

    def normalize_assets(items: list[Any], prefix: str, path: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            asset = _require_dict(item, f"{path}[{index}]")
            asset_id = _require_str(asset.get("asset_id"), f"{path}[{index}].asset_id")
            if not asset_id.startswith(prefix):
                raise NodeValidationError(f"{path}[{index}].asset_id must start with `{prefix}`.")
            sourcing_mode = _require_str(asset.get("sourcing_mode"), f"{path}[{index}].sourcing_mode")
            if sourcing_mode not in allowed_sourcing_modes:
                raise NodeValidationError(
                    f"{path}[{index}].sourcing_mode must be one of generate_fresh / generate_from_reference / use_reference_directly / use_material_library_directly."
                )
            reference_ids = _require_str_list(asset.get("reference_ids", []), f"{path}[{index}].reference_ids")
            for reference_id in reference_ids:
                if reference_id not in available_reference_ids:
                    raise NodeValidationError(f"{path}[{index}] references undefined reference asset `{reference_id}`.")
            material_ids = _require_str_list(asset.get("material_ids", []), f"{path}[{index}].material_ids")
            for material_id in material_ids:
                if material_id not in available_material_ids:
                    raise NodeValidationError(f"{path}[{index}] references undefined material `{material_id}`.")
            if sourcing_mode != "generate_fresh" and not reference_ids:
                if sourcing_mode in {"generate_from_reference", "use_reference_directly"}:
                    raise NodeValidationError(f"{path}[{index}].reference_ids must not be empty for `{sourcing_mode}`.")
            if sourcing_mode == "use_material_library_directly" and not material_ids:
                raise NodeValidationError(f"{path}[{index}].material_ids must not be empty for `{sourcing_mode}`.")
            name = _require_str(asset.get("name"), f"{path}[{index}].name")
            purpose = _require_str(asset.get("purpose"), f"{path}[{index}].purpose")
            priority = _require_str(asset.get("priority"), f"{path}[{index}].priority")
            reference_notes = _require_str(
                asset.get("reference_notes", ""), f"{path}[{index}].reference_notes", allow_empty=True
            )
            visual_description = _require_str(
                asset.get("visual_description"), f"{path}[{index}].visual_description"
            )
            generation_prompt = _require_str(
                asset.get("generation_prompt", ""), f"{path}[{index}].generation_prompt", allow_empty=True
            )
            if not generation_prompt.strip():
                generation_prompt = fallback_generation_prompt(
                    asset_id=asset_id,
                    name=name,
                    purpose=purpose,
                    sourcing_mode=sourcing_mode,
                    material_ids=material_ids,
                    reference_notes=reference_notes,
                    visual_description=visual_description,
                )
            normalized.append(
                {
                    "asset_id": asset_id,
                    "name": name,
                    "purpose": purpose,
                    "priority": priority,
                    "sourcing_mode": sourcing_mode,
                    "reference_ids": reference_ids,
                    "material_ids": material_ids,
                    "reference_notes": reference_notes,
                    "visual_description": visual_description,
                    "generation_prompt": generation_prompt,
                }
            )
        return normalized

    normalized_characters = normalize_assets(characters, "C", "characters")
    normalized_scenes = normalize_assets(scenes, "S", "scenes")
    normalized_props = normalize_assets(props, "P", "props")
    all_ids = {item["asset_id"] for item in normalized_characters + normalized_scenes + normalized_props}
    for item in must_generate_first:
        if item not in all_ids:
            raise NodeValidationError("must_generate_first contains undefined asset IDs.")

    normalized_reuse: list[dict[str, Any]] = []
    for index, item in enumerate(reuse_plan):
        entry = _require_dict(item, f"reuse_plan[{index}]")
        assets_list = _require_str_list(entry.get("assets"), f"reuse_plan[{index}].assets")
        for asset_id in assets_list:
            if asset_id not in all_ids:
                raise NodeValidationError(f"reuse_plan[{index}] references undefined asset `{asset_id}`.")
        normalized_reuse.append(
            {
                "assets": assets_list,
                "episode_refs": _require_str_list(entry.get("episode_refs"), f"reuse_plan[{index}].episode_refs"),
                "note": _require_str(entry.get("note", ""), f"reuse_plan[{index}].note", allow_empty=True),
            }
        )

    payload = {
        "direction_id": direction_id,
        "series_title": script["series_title"],
        "analysis_anchor": {
            "working_title": analysis["source_overview"]["working_title"],
            "visual_style": analysis["source_overview"]["visual_style"],
        },
        "reference_library": reference_library["references"],
        "material_library": material_library["materials"],
        "style_guide": {
            "visual_style": _require_str(style_guide.get("visual_style"), "style_guide.visual_style"),
            "style_prefix": _require_str(style_guide.get("style_prefix"), "style_guide.style_prefix"),
            "palette": _require_str_list(style_guide.get("palette"), "style_guide.palette"),
            "rendering_notes": _require_str_list(style_guide.get("rendering_notes"), "style_guide.rendering_notes"),
            "consistency_rules": _require_str_list(style_guide.get("consistency_rules"), "style_guide.consistency_rules"),
        },
        "characters": normalized_characters,
        "scenes": normalized_scenes,
        "props": normalized_props,
        "reuse_plan": normalized_reuse,
        "must_generate_first": must_generate_first,
    }

    artifact = StageArtifact(
        stage="asset_planner",
        generated_at=utc_now(),
        status="completed",
        summary="Asset library plan completed.",
        required_inputs=["Completed analysis artifact and script artifact."],
        expected_outputs=["Character, scene, and prop asset registry."],
        next_stage="storyboard_generator",
        payload=payload,
    )
    markdown = render_asset_markdown(config, payload)
    return artifact, markdown


def render_asset_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['series_title']} - 素材库规划",
        "",
        f"- Project: `{config['project_name']}`",
        f"- Direction: `{payload['direction_id']}`",
        f"- 视觉风格：{payload['style_guide']['visual_style']}",
        f"- 风格前缀：{payload['style_guide']['style_prefix']}",
        "",
    ]
    if payload["reference_library"]:
        lines.extend(["## 外部参考素材", ""])
        for item in payload["reference_library"]:
            must_keep = " / ".join(item["must_keep"]) if item["must_keep"] else "无"
            lines.extend(
                [
                    f"### {item['reference_id']} {item['name']}",
                    "",
                    f"- 类别：{item['category']}",
                    f"- 来源：{item['source_type']} -> {item['source']}",
                    f"- 保留要点：{must_keep}",
                    f"- 使用说明：{item['usage_notes']}",
                    "",
                ]
            )
    if payload["material_library"]:
        lines.extend(["## 现有素材库", ""])
        for item in payload["material_library"]:
            tags = " / ".join(item["tags"]) if item["tags"] else "无"
            lines.extend(
                [
                    f"### {item['material_id']} {item['name']}",
                    "",
                    f"- 类别：{item['category']}",
                    f"- 介质：{item['media_type']}",
                    f"- 来源：{item['source_type'] if 'source_type' in item else item['source_kind']} -> {item['source']}",
                    f"- 标签：{tags}",
                    f"- 使用说明：{item['usage_notes']}",
                    "",
                ]
            )
    lines.extend(["## 风格规则", ""])
    lines.extend(f"- 配色：{item}" for item in payload["style_guide"]["palette"])
    lines.extend(f"- 渲染备注：{item}" for item in payload["style_guide"]["rendering_notes"])
    lines.extend(f"- 一致性规则：{item}" for item in payload["style_guide"]["consistency_rules"])

    def append_assets(title: str, items: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        for item in items:
            lines.extend(
                [
                    f"### {item['asset_id']} {item['name']}",
                    "",
                    f"- 用途：{item['purpose']}",
                    f"- 优先级：{item['priority']}",
                    f"- 素材来源：{item['sourcing_mode']}",
                    f"- 参考素材：{', '.join(item['reference_ids']) if item['reference_ids'] else '无'}",
                    f"- 绑定素材：{', '.join(item['material_ids']) if item['material_ids'] else '无'}",
                    f"- 参考备注：{item['reference_notes']}",
                    f"- 视觉描述：{item['visual_description']}",
                    "",
                    "```text",
                    item["generation_prompt"],
                    "```",
                    "",
                ]
            )

    append_assets("角色素材", payload["characters"])
    append_assets("场景素材", payload["scenes"])
    append_assets("道具素材", payload["props"])
    lines.extend(["## 复用计划", ""])
    for item in payload["reuse_plan"]:
        lines.append(f"- {', '.join(item['assets'])} -> {', '.join(item['episode_refs'])}：{item['note']}")
    lines.extend(["", "## 必须先生成", ""])
    lines.extend(f"- {item}" for item in payload["must_generate_first"])
    lines.append("")
    return "\n".join(lines)


def prepare_storyboard_generation(
    config: dict[str, Any],
    script: dict[str, Any],
    assets: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> PreparedRequest:
    reference_summary = render_reference_library_summary(reference_library)
    material_summary = render_material_library_summary(material_library)
    asset_source_lines = []
    for item in assets["characters"] + assets["scenes"] + assets["props"]:
        references = ", ".join(item["reference_ids"]) if item["reference_ids"] else "none"
        materials = ", ".join(item.get("material_ids", [])) if item.get("material_ids", []) else "none"
        asset_source_lines.append(
            f"- {item['asset_id']} {item['name']} | {item['sourcing_mode']} | refs: {references} | materials: {materials} | notes: {item['reference_notes']}"
        )
    asset_source_summary = "\n".join(asset_source_lines)
    segment_plan_lines = []
    for item in script["video_plan"]["generation_segments"]:
        segment_plan_lines.append(
            f"- {item['segment_id']} | {item['duration_seconds']}s | {item['purpose']} | {item['coverage']}"
        )
    segment_plan_summary = "\n".join(segment_plan_lines)
    prompt = f"""# Storyboard Generation Request

你是“Seedance / AI 视频分镜规划”专家。请基于已经完成的裂变视频脚本、video_plan 和素材库，生成可以直接用于分段视频生成的详细分镜提示词。

## Project Context

- Project: `{config["project_name"]}`
- Aspect ratio: `{config["aspect_ratio"]}`
- Target finished runtime: {script["video_plan"]["total_duration_seconds"]} seconds
- Generation segment count: {len(script["video_plan"]["generation_segments"])}

## Script Anchor

- Series title: {script["series_title"]}
- Direction: {script["direction_name"]}
- Core hook: {script["core_hook"]}
- Visual emphasis: {' / '.join(script["production_notes"]["visual_emphasis"])}
- Continuity rules: {' / '.join(script["production_notes"]["continuity_rules"])}

## Video Plan

- Planning reason: {script["video_plan"]["planning_reason"]}
- Generation segments:
{segment_plan_summary}

## Asset Anchor

- Style prefix: {assets["style_guide"]["style_prefix"]}
- Must-generate assets: {' / '.join(assets["must_generate_first"])}
- Available character assets: {' / '.join(item["asset_id"] for item in assets["characters"])}
- Available scene assets: {' / '.join(item["asset_id"] for item in assets["scenes"])}
- Available prop assets: {' / '.join(item["asset_id"] for item in assets["props"])}

## Asset Source Rules

{asset_source_summary}

## External Reference Library

{reference_summary}

## Existing Material Library

{material_summary}

## Generation goals

1. 为这 1 条独立视频输出 1 份 storyboard package，但 package 内必须按 `video_plan.generation_segments` 拆成多个 segment prompt。
2. 每个 generation segment 都要单独写：用途、聚焦素材、可直接喂给视频模型的 prompt、声音设计、尾帧状态、与下一段的衔接。
3. 如果脚本里存在明显的场地变化、镜头视角跳变、叙事阶段切换，必须在相应 segment prompt 中明确体现，不要把多个阶段揉成一段。
4. 所有 segment 共享同一套人物、场景、道具素材设定；不要在分段之间随意改名、换造型或重设身份。
5. 上传素材表仍然要给出，但它应该服务于整条视频，并且 segment 内通过 `focus_asset_ids` 指明当前段主要使用哪些资产。
6. Prompt 要具体到镜头、动作、情绪、景别、屏幕文字和声音，细到足以直接进入生成层。
7. 素材引用必须只使用素材库里已定义的 ID。
8. 如果某项素材依赖参考图，上传表里要明确写出 `reference_ids`，提醒执行时一并上传。
9. 如果某项素材已绑定现有素材库，上传表里要明确写出 `material_ids`。
10. 分镜里的人物命名必须沿用 script 已锁定的角色原型与名称。若脚本已选定 `Cheems / Balltze meme dog` 或 `Smudge the Cat meme`，分镜与后续 prompt 中都要保持这个身份锚点，不要换成别的猫狗 meme。
11. 如果环境锚点指向《三角洲行动》式工业据点/战术办公室氛围，分镜 prompt 要持续体现这种场景语气与视觉语言，把场景说清楚，但不要机械堆砌官方专有名词。

## Output requirements

- 严格输出 JSON。
- `episodes` 数量必须和剧本一致。
- `upload_slots[].material_type` 只能是 `asset` 或 `reference`。
- `upload_slots[].material_ids` 只能引用 Existing Material Library 中已定义的 `material_id`。
- `generation_segments` 数量必须与 Video Plan 完全一致。
- `generation_segments[].segment_id` 必须与 Video Plan 中的 ID 一一对应。
- `generation_segments[].duration_seconds` 必须与 Video Plan 对应段一致，且不得超过 10。
- 每个 `episodes[]`、`upload_slots[]`、`generation_segments[]` 条目都必须把所有字符串字段填满；不要留空字符串，不要用占位符。
- `upload_slots[].usage` 必须明确说明该素材在整条视频里的职责。
- 每个 `generation_segments[]` 都必须填写非空的 `purpose`、`focus_asset_ids`、`timeline_prompt`、`sound_design`、`end_frame`、`continuity_to_next`。
- 如果某段已经是最后一段，`continuity_to_next` 也要明确写成类似“本段结束，视频在此闭环收口”，不要留空。
- 每个 `generation_segments[].timeline_prompt` 必须是单段可执行 prompt，不要写整条视频总述。
"""
    template_segments = [
        {
            "segment_id": item["segment_id"],
            "duration_seconds": item["duration_seconds"],
            "purpose": "",
            "focus_asset_ids": ["C01", "S01"],
            "timeline_prompt": "",
            "sound_design": "",
            "end_frame": "",
            "continuity_to_next": "",
        }
        for item in script["video_plan"]["generation_segments"]
    ]
    template = {
        "direction_id": script["direction_id"],
        "series_style": "",
        "global_rules": [""],
        "episodes": [
            {
                "episode_number": 1,
                "title": "",
                "upload_slots": [
                    {
                        "slot": "@图片1",
                        "material_type": "asset",
                        "asset_id": "C01",
                        "reference_ids": [],
                        "material_ids": [],
                        "usage": "",
                    }
                ],
                "generation_segments": template_segments,
                "timeline_prompt": "",
                "sound_design": "",
                "end_frame": "",
                "continuity_to_next": "",
            }
        ],
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a storyboard generation request.")

def finalize_storyboard_generation(
    config: dict[str, Any],
    script: dict[str, Any],
    assets: dict[str, Any],
    response: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> tuple[StageArtifact, str]:
    direction_id = _require_str(response.get("direction_id"), "direction_id")
    if direction_id != script["direction_id"]:
        raise NodeValidationError("direction_id in storyboard response must match script direction_id.")
    global_rules = _require_str_list(response.get("global_rules"), "global_rules")
    episodes = _require_list(response.get("episodes"), "episodes")
    expected_episode_count = len(script["episodes"])
    if len(episodes) != expected_episode_count:
        raise NodeValidationError(
            f"episodes must contain exactly {expected_episode_count} items for the storyboard stage."
        )
    all_asset_ids = {item["asset_id"] for item in assets["characters"] + assets["scenes"] + assets["props"]}
    asset_reference_map = {
        item["asset_id"]: set(item["reference_ids"]) for item in assets["characters"] + assets["scenes"] + assets["props"]
    }
    asset_material_map = {
        item["asset_id"]: set(item.get("material_ids", []))
        for item in assets["characters"] + assets["scenes"] + assets["props"]
    }
    available_reference_ids = {item["reference_id"] for item in reference_library["references"]}
    available_material_ids = {item["material_id"] for item in material_library["materials"]}
    planned_segments = script["video_plan"]["generation_segments"]
    normalized_episodes: list[dict[str, Any]] = []
    for index, item in enumerate(episodes):
        episode = _require_dict(item, f"episodes[{index}]")
        episode_number = _require_int(episode.get("episode_number"), f"episodes[{index}].episode_number")
        if episode_number != index + 1:
            raise NodeValidationError("storyboard episodes must be ordered sequentially starting at 1.")
        upload_slots = _require_list(episode.get("upload_slots"), f"episodes[{index}].upload_slots")
        normalized_slots: list[dict[str, Any]] = []
        for slot_index, slot_item in enumerate(upload_slots):
            slot = _require_dict(slot_item, f"episodes[{index}].upload_slots[{slot_index}]")
            asset_id = _require_str(slot.get("asset_id"), f"episodes[{index}].upload_slots[{slot_index}].asset_id")
            if asset_id not in all_asset_ids:
                raise NodeValidationError(f"episodes[{index}] references undefined asset `{asset_id}`.")
            material_type = _require_str(
                slot.get("material_type"), f"episodes[{index}].upload_slots[{slot_index}].material_type"
            )
            if material_type not in {"asset", "reference"}:
                raise NodeValidationError(
                    f"episodes[{index}].upload_slots[{slot_index}].material_type must be `asset` or `reference`."
                )
            reference_ids = _require_str_list(
                slot.get("reference_ids", []), f"episodes[{index}].upload_slots[{slot_index}].reference_ids"
            )
            material_ids = _require_str_list(
                slot.get("material_ids", []), f"episodes[{index}].upload_slots[{slot_index}].material_ids"
            )
            for reference_id in reference_ids:
                if reference_id not in available_reference_ids:
                    raise NodeValidationError(
                        f"episodes[{index}].upload_slots[{slot_index}] references undefined reference `{reference_id}`."
                    )
            for material_id in material_ids:
                if material_id not in available_material_ids:
                    raise NodeValidationError(
                        f"episodes[{index}].upload_slots[{slot_index}] references undefined material `{material_id}`."
                    )
            if material_type == "reference" and not reference_ids:
                raise NodeValidationError(
                    f"episodes[{index}].upload_slots[{slot_index}].reference_ids must not be empty for reference slots."
                )
            if reference_ids and not set(reference_ids).issubset(asset_reference_map[asset_id]):
                raise NodeValidationError(
                    f"episodes[{index}].upload_slots[{slot_index}] includes reference_ids not bound to asset `{asset_id}`."
                )
            if material_ids and not set(material_ids).issubset(asset_material_map[asset_id]):
                raise NodeValidationError(
                    f"episodes[{index}].upload_slots[{slot_index}] includes material_ids not bound to asset `{asset_id}`."
                )
            normalized_slots.append(
                {
                    "slot": _require_str(slot.get("slot"), f"episodes[{index}].upload_slots[{slot_index}].slot"),
                    "material_type": material_type,
                    "asset_id": asset_id,
                    "reference_ids": reference_ids,
                    "material_ids": material_ids,
                    "usage": _require_str(slot.get("usage"), f"episodes[{index}].upload_slots[{slot_index}].usage"),
                }
            )

        generation_segments = _require_list(episode.get("generation_segments"), f"episodes[{index}].generation_segments")
        segment_map: dict[str, dict[str, Any]] = {}
        for segment_index, raw_segment in enumerate(generation_segments):
            segment = _require_dict(raw_segment, f"episodes[{index}].generation_segments[{segment_index}]")
            segment_id = _require_str(
                segment.get("segment_id"), f"episodes[{index}].generation_segments[{segment_index}].segment_id"
            )
            segment_map[segment_id] = segment
        missing_segment_ids = [item["segment_id"] for item in planned_segments if item["segment_id"] not in segment_map]
        if missing_segment_ids:
            raise NodeValidationError(
                f"episodes[{index}].generation_segments is missing planned segments: {', '.join(missing_segment_ids)}."
            )

        normalized_segments: list[dict[str, Any]] = []
        for segment_index, planned in enumerate(planned_segments):
            segment_id = planned["segment_id"]
            segment = segment_map[segment_id]
            duration_seconds = _require_int(
                segment.get("duration_seconds"),
                f"episodes[{index}].generation_segments[{segment_index}].duration_seconds",
            )
            if duration_seconds != planned["duration_seconds"]:
                raise NodeValidationError(
                    f"episodes[{index}].generation_segments[{segment_id}].duration_seconds must match script video_plan."
                )
            if duration_seconds > 10:
                raise NodeValidationError("Storyboard generation segments must not exceed 10 seconds.")
            focus_asset_ids = _require_str_list(
                segment.get("focus_asset_ids"), f"episodes[{index}].generation_segments[{segment_index}].focus_asset_ids"
            )
            for asset_id in focus_asset_ids:
                if asset_id not in all_asset_ids:
                    raise NodeValidationError(
                        f"episodes[{index}].generation_segments[{segment_id}] references undefined asset `{asset_id}`."
                    )
            normalized_segments.append(
                {
                    "segment_id": segment_id,
                    "duration_seconds": duration_seconds,
                    "purpose": _require_str(
                        segment.get("purpose"), f"episodes[{index}].generation_segments[{segment_index}].purpose"
                    ),
                    "coverage": planned["coverage"],
                    "focus_asset_ids": focus_asset_ids,
                    "timeline_prompt": _require_str(
                        segment.get("timeline_prompt"),
                        f"episodes[{index}].generation_segments[{segment_index}].timeline_prompt",
                    ),
                    "sound_design": _require_str(
                        segment.get("sound_design"),
                        f"episodes[{index}].generation_segments[{segment_index}].sound_design",
                    ),
                    "end_frame": _require_str(
                        segment.get("end_frame"),
                        f"episodes[{index}].generation_segments[{segment_index}].end_frame",
                    ),
                    "continuity_to_next": _require_str(
                        segment.get("continuity_to_next", ""),
                        f"episodes[{index}].generation_segments[{segment_index}].continuity_to_next",
                        allow_empty=True,
                    ),
                }
            )

        combined_prompt_blocks: list[str] = []
        for segment in normalized_segments:
            combined_prompt_blocks.extend(
                [
                    f"[{segment['segment_id']} | {segment['duration_seconds']}s | assets: {', '.join(segment['focus_asset_ids'])}]",
                    segment["timeline_prompt"],
                    "",
                ]
            )
        combined_sound = "\n".join(
            f"{segment['segment_id']}: {segment['sound_design']}" for segment in normalized_segments
        )
        normalized_episodes.append(
            {
                "episode_number": episode_number,
                "title": _require_str(episode.get("title"), f"episodes[{index}].title"),
                "upload_slots": normalized_slots,
                "generation_segments": normalized_segments,
                "timeline_prompt": "\n".join(combined_prompt_blocks).strip(),
                "sound_design": combined_sound,
                "end_frame": normalized_segments[-1]["end_frame"],
                "continuity_to_next": normalized_segments[-1]["continuity_to_next"],
            }
        )

    payload = {
        "direction_id": direction_id,
        "series_title": script["series_title"],
        "series_style": _require_str(response.get("series_style"), "series_style"),
        "global_rules": global_rules,
        "video_plan": script["video_plan"],
        "episodes": normalized_episodes,
    }

    artifact = StageArtifact(
        stage="storyboard_generator",
        generated_at=utc_now(),
        status="completed",
        summary="Storyboard plan completed.",
        required_inputs=["Completed script artifact and asset registry."],
        expected_outputs=["Segment-level storyboard prompts ready for generation."],
        next_stage="qa_reviewer",
        payload=payload,
    )
    markdown = render_storyboard_markdown(config, payload)
    return artifact, markdown

def render_storyboard_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['series_title']} - 分镜总览",
        "",
        f"- Project: `{config['project_name']}`",
        f"- Direction: `{payload['direction_id']}`",
        f"- 系列风格：{payload['series_style']}",
        f"- 总时长规划：{payload['video_plan']['total_duration_seconds']}s",
        "",
        "## 全局规则",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["global_rules"])
    lines.extend(["", "## 分段计划", "", f"- 规划理由：{payload['video_plan']['planning_reason']}", ""])
    for segment in payload["video_plan"]["generation_segments"]:
        lines.extend(
            [
                f"### {segment['segment_id']}",
                "",
                f"- 时长：{segment['duration_seconds']}s",
                f"- 作用：{segment['purpose']}",
                f"- 覆盖内容：{segment['coverage']}",
                "",
            ]
        )
    lines.extend(["## 视频分镜", ""])
    for episode in payload["episodes"]:
        lines.extend(
            [
                f"### V{episode['episode_number']:02d} {episode['title']}",
                "",
                "| 上传位置 | 类型 | 素材ID | 参考ID | 素材库ID | 用途 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for slot in episode["upload_slots"]:
            reference_ids = ", ".join(slot["reference_ids"]) if slot["reference_ids"] else "-"
            material_ids = ", ".join(slot.get("material_ids", [])) if slot.get("material_ids", []) else "-"
            lines.append(
                f"| {slot['slot']} | {slot['material_type']} | {slot['asset_id']} | {reference_ids} | {material_ids} | {slot['usage']} |"
            )
        lines.append("")
        for segment in episode["generation_segments"]:
            lines.extend(
                [
                    f"#### {segment['segment_id']} ({segment['duration_seconds']}s)",
                    "",
                    f"- 作用：{segment['purpose']}",
                    f"- 主要资产：{', '.join(segment['focus_asset_ids'])}",
                    f"- 覆盖内容：{segment['coverage']}",
                    "",
                    "```text",
                    segment["timeline_prompt"],
                    "```",
                    "",
                    f"- 声音设计：{segment['sound_design']}",
                    f"- 尾帧：{segment['end_frame']}",
                    f"- 承接：{segment['continuity_to_next'] or '无'}",
                    "",
                ]
            )
    return "\n".join(lines)

def render_storyboard_episode_markdown(series_title: str, episode: dict[str, Any]) -> str:
    lines = [
        f"# {series_title} - V{episode['episode_number']:02d} {episode['title']}",
        "",
        "## 素材上传清单",
        "",
        "| 上传位置 | 类型 | 素材ID | 参考ID | 素材库ID | 用途 |",
        "|---|---|---|---|---|---|",
    ]
    for slot in episode["upload_slots"]:
        reference_ids = ", ".join(slot["reference_ids"]) if slot["reference_ids"] else "-"
        material_ids = ", ".join(slot.get("material_ids", [])) if slot.get("material_ids", []) else "-"
        lines.append(
            f"| {slot['slot']} | {slot['material_type']} | {slot['asset_id']} | {reference_ids} | {material_ids} | {slot['usage']} |"
        )
    lines.extend(["", "## Generation Segments", ""])
    for segment in episode["generation_segments"]:
        lines.extend(
            [
                f"### {segment['segment_id']} ({segment['duration_seconds']}s)",
                "",
                f"- 作用：{segment['purpose']}",
                f"- 主要资产：{', '.join(segment['focus_asset_ids'])}",
                f"- 覆盖内容：{segment['coverage']}",
                "",
                "```text",
                segment["timeline_prompt"],
                "```",
                "",
                "## 声音设计",
                "",
                segment["sound_design"],
                "",
                "## 尾帧描述",
                "",
                segment["end_frame"],
                "",
                "## 承接说明",
                "",
                segment["continuity_to_next"] or "无",
                "",
            ]
        )
    return "\n".join(lines)

def build_execution_plan(project_root: Path, config: dict[str, Any]) -> StageArtifact:
    source_video = load_source_video_input(project_root, config)
    assets = load_artifact_payload(project_root, "asset_planner")
    storyboards = load_artifact_payload(project_root, "storyboard_generator")
    qa = load_artifact_payload(project_root, "qa_reviewer")
    reference_library = load_reference_library(project_root)
    material_library = load_material_library(project_root)

    assets_by_id = {
        item["asset_id"]: item for item in assets["characters"] + assets["scenes"] + assets["props"]
    }
    references_by_id = {item["reference_id"]: item for item in reference_library["references"]}
    materials_by_id = {item["material_id"]: item for item in material_library["materials"]}

    episodes: list[dict[str, Any]] = []
    workflow_blockers: list[str] = []
    if not qa["ready_for_generation"]:
        workflow_blockers.append(f"QA not ready for generation: {qa['summary']}")

    for episode in storyboards["episodes"]:
        resolved_inputs: list[dict[str, Any]] = []
        episode_blockers: list[str] = []
        for slot in episode["upload_slots"]:
            asset = assets_by_id[slot["asset_id"]]
            resolved_reference_ids = slot["reference_ids"] or asset["reference_ids"]
            resolved_material_ids = slot.get("material_ids", []) or asset.get("material_ids", [])
            source_files: list[dict[str, Any]] = []

            for reference_id in resolved_reference_ids:
                reference = references_by_id[reference_id]
                source_files.append(
                    {
                        "kind": "reference",
                        "source_id": reference_id,
                        "name": reference["name"],
                        "source": reference["source"],
                        "resolved_source": reference["resolved_source"],
                    }
                )

            for material_id in resolved_material_ids:
                material = materials_by_id[material_id]
                source_files.append(
                    {
                        "kind": "material",
                        "source_id": material_id,
                        "name": material["name"],
                        "source": material["source"],
                        "resolved_source": material["resolved_source"],
                    }
                )

            if asset["sourcing_mode"] == "use_material_library_directly" and not resolved_material_ids:
                episode_blockers.append(
                    f"E{episode['episode_number']:02d} {slot['slot']} requires material_library assets for {asset['asset_id']}."
                )
            if asset["sourcing_mode"] in {"generate_from_reference", "use_reference_directly"} and not resolved_reference_ids:
                episode_blockers.append(
                    f"E{episode['episode_number']:02d} {slot['slot']} requires reference assets for {asset['asset_id']}."
                )

            resolved_inputs.append(
                {
                    "slot": slot["slot"],
                    "asset_id": slot["asset_id"],
                    "asset_name": asset["name"],
                    "material_type": slot["material_type"],
                    "usage": slot["usage"],
                    "sourcing_mode": asset["sourcing_mode"],
                    "reference_ids": resolved_reference_ids,
                    "material_ids": resolved_material_ids,
                    "source_files": source_files,
                }
            )

        resolved_segments: list[dict[str, Any]] = []
        for segment in episode.get("generation_segments", []):
            focus_asset_ids = set(segment["focus_asset_ids"])
            segment_inputs = [item for item in resolved_inputs if item["asset_id"] in focus_asset_ids]
            if not segment_inputs:
                segment_inputs = resolved_inputs
            resolved_segments.append(
                {
                    "segment_id": segment["segment_id"],
                    "duration_seconds": segment["duration_seconds"],
                    "purpose": segment["purpose"],
                    "coverage": segment["coverage"],
                    "focus_asset_ids": segment["focus_asset_ids"],
                    "timeline_prompt": segment["timeline_prompt"],
                    "sound_design": segment["sound_design"],
                    "end_frame": segment["end_frame"],
                    "continuity_to_next": segment["continuity_to_next"],
                    "resolved_inputs": segment_inputs,
                }
            )

        ready_to_render = not episode_blockers and qa["ready_for_generation"]
        episodes.append(
            {
                "episode_number": episode["episode_number"],
                "title": episode["title"],
                "timeline_prompt": episode["timeline_prompt"],
                "sound_design": episode["sound_design"],
                "end_frame": episode["end_frame"],
                "continuity_to_next": episode["continuity_to_next"],
                "generation_segments": resolved_segments,
                "resolved_inputs": resolved_inputs,
                "blockers": episode_blockers,
                "ready_to_render": ready_to_render,
            }
        )
        workflow_blockers.extend(episode_blockers)

    payload = {
        "series_title": storyboards["series_title"],
        "direction_id": storyboards["direction_id"],
        "video_plan": storyboards.get("video_plan", {}),
        "qa_ready_for_generation": qa["ready_for_generation"],
        "qa_summary": qa["summary"],
        "source_video": source_video,
        "material_library": material_library["materials"],
        "episodes": episodes,
        "workflow_blockers": workflow_blockers,
    }

    return StageArtifact(
        stage="execution_planner",
        generated_at=utc_now(),
        status="completed",
        summary="Execution plan resolved against the source video, reference assets, and material library.",
        required_inputs=["QA report, source inputs, asset registry, and storyboards."],
        expected_outputs=["Per-video resolved upload list and render readiness report."],
        next_stage="video_renderer",
        payload=payload,
    )

def render_execution_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['series_title']} - 执行计划",
        "",
        f"- Project: `{config['project_name']}`",
        f"- Direction: `{payload['direction_id']}`",
        f"- QA Ready: `{payload['qa_ready_for_generation']}`",
        f"- QA Summary: {payload['qa_summary']}",
        "",
        "## Source Video",
        "",
    ]
    lines.extend(render_source_video_summary(payload["source_video"]).splitlines())
    lines.extend(["", "## Workflow Blockers", ""])
    if payload["workflow_blockers"]:
        lines.extend(f"- {item}" for item in payload["workflow_blockers"])
    else:
        lines.append("- None")
    lines.extend(["", "## Video Plans", ""])
    for episode in payload["episodes"]:
        lines.extend(
            [
                f"### V{episode['episode_number']:02d} {episode['title']}",
                "",
                f"- Ready to render: `{episode['ready_to_render']}`",
            ]
        )
        if episode["blockers"]:
            lines.extend(f"- Blocker: {item}" for item in episode["blockers"])
        lines.extend(
            [
                "",
                "| 上传位置 | 素材ID | 来源模式 | 参考ID | 素材库ID |",
                "|---|---|---|---|---|",
            ]
        )
        for item in episode["resolved_inputs"]:
            reference_ids = ", ".join(item["reference_ids"]) if item["reference_ids"] else "-"
            material_ids = ", ".join(item.get("material_ids", [])) if item.get("material_ids", []) else "-"
            lines.append(
                f"| {item['slot']} | {item['asset_id']} | {item['sourcing_mode']} | {reference_ids} | {material_ids} |"
            )
        lines.append("")
        for segment in episode.get("generation_segments", []):
            lines.extend(
                [
                    f"#### {segment['segment_id']} ({segment['duration_seconds']}s)",
                    "",
                    f"- 作用：{segment['purpose']}",
                    f"- 主要资产：{', '.join(segment['focus_asset_ids'])}",
                    "",
                    "```text",
                    segment["timeline_prompt"],
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)

def render_execution_episode_markdown(series_title: str, episode: dict[str, Any]) -> str:
    lines = [
        f"# {series_title} - V{episode['episode_number']:02d} {episode['title']} 执行计划",
        "",
        f"- Ready to render: `{episode['ready_to_render']}`",
        "",
        "| 上传位置 | 素材ID | 来源模式 | 参考ID | 素材库ID |",
        "|---|---|---|---|---|",
    ]
    for item in episode["resolved_inputs"]:
        reference_ids = ", ".join(item["reference_ids"]) if item["reference_ids"] else "-"
        material_ids = ", ".join(item.get("material_ids", [])) if item.get("material_ids", []) else "-"
        lines.append(
            f"| {item['slot']} | {item['asset_id']} | {item['sourcing_mode']} | {reference_ids} | {material_ids} |"
        )
    lines.extend(["", "## Generation Segments", ""])
    for segment in episode.get("generation_segments", []):
        lines.extend(
            [
                f"### {segment['segment_id']} ({segment['duration_seconds']}s)",
                "",
                f"- 作用：{segment['purpose']}",
                f"- 主要资产：{', '.join(segment['focus_asset_ids'])}",
                f"- 覆盖内容：{segment['coverage']}",
                "",
                "## Prompt",
                "",
                "```text",
                segment["timeline_prompt"],
                "```",
                "",
                "## 声音设计",
                "",
                segment["sound_design"],
                "",
                "## 尾帧描述",
                "",
                segment["end_frame"],
                "",
                "## 承接说明",
                "",
                segment["continuity_to_next"] or "无",
                "",
            ]
        )
    return "\n".join(lines)

def prepare_video_rendering(config: dict[str, Any], execution_plan: dict[str, Any]) -> PreparedRequest:
    ready_count = sum(1 for item in execution_plan["episodes"] if item["ready_to_render"])
    blockers = execution_plan["workflow_blockers"]
    blocker_summary = " / ".join(blockers) if blockers else "none"
    segment_count = sum(len(item.get("generation_segments", [])) for item in execution_plan["episodes"] if item["ready_to_render"])
    prompt = f"""# Video Rendering Request

你现在进入执行层。请根据执行计划逐段完成视频生成，并把实际产出结果记录成 JSON。每个生成段都应同时使用该段的文字 prompt 和为该段准备的参考图像/素材图像。

## Project Context

- Project: `{config["project_name"]}`
- Aspect ratio: `{config["aspect_ratio"]}`
- Planned total duration: {execution_plan.get("video_plan", {}).get("total_duration_seconds", config["episode_duration_seconds"])} seconds
- Series title: {execution_plan["series_title"]}
- Direction: {execution_plan["direction_id"]}

## Render readiness

- Episodes ready to render: {ready_count}/{len(execution_plan["episodes"])}
- Generation segments ready to render: {segment_count}
- Workflow blockers: {blocker_summary}

## Instructions

1. 仅对 `ready_to_render=true` 的视频执行生成。
2. 每条视频要按 `generation_segments` 逐段生成，不要把整条视频合并成一次请求。
3. 每个 segment 都要使用对应 segment 的 `timeline_prompt`，并同时附上该 segment 的参考图像输入。
4. 把每段最终视频的本地路径或远程 URL 写回 `segments`。
5. 如果所有段都已完成但尚未拼接整条视频，episode 级别可以没有 `output_video`，并在 `notes` 说明需要后续拼接。
6. 如果某段未生成成功，segment 的 `status` 写 `failed`，并填写 `error`.
"""
    template = {
        "provider": "",
        "run_label": "",
        "episodes": [
            {
                "episode_number": 1,
                "task_id": "",
                "status": "in_progress",
                "output_video": "",
                "duration_seconds": config["episode_duration_seconds"],
                "used_reference_ids": [""],
                "used_material_ids": [""],
                "notes": "",
                "error": "",
                "segments": [
                    {
                        "segment_id": "G01",
                        "task_id": "",
                        "status": "submitted",
                        "output_video": "",
                        "duration_seconds": 8,
                        "used_reference_ids": [""],
                        "used_material_ids": [""],
                        "notes": "",
                        "error": "",
                    }
                ],
            }
        ],
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a segment-level video rendering request.")


def build_render_request_files(project_root: Path, config: dict[str, Any]) -> None:
    execution_plan = load_artifact_payload(project_root, "execution_planner")
    output_root = _stage_output_root(project_root, "render-videos")
    for episode in execution_plan["episodes"]:
        path = output_root / f"E{episode['episode_number']:02d}_渲染请求.md"
        path.write_text(render_execution_episode_markdown(execution_plan["series_title"], episode), encoding="utf-8")


def finalize_video_rendering(
    config: dict[str, Any], execution_plan: dict[str, Any], response: dict[str, Any]
) -> tuple[StageArtifact, str]:
    provider = _require_str(response.get("provider"), "provider")
    run_label = _require_str(response.get("run_label"), "run_label")
    episodes = _require_list(response.get("episodes"), "episodes")
    expected_episode_count = len(execution_plan["episodes"])
    if len(episodes) != expected_episode_count:
        raise NodeValidationError(f"episodes must contain exactly {expected_episode_count} items for render-videos.")

    execution_by_number = {item["episode_number"]: item for item in execution_plan["episodes"]}
    normalized_episodes: list[dict[str, Any]] = []
    has_pending = False
    for index, item in enumerate(episodes):
        episode = _require_dict(item, f"episodes[{index}]")
        episode_number = _require_int(episode.get("episode_number"), f"episodes[{index}].episode_number")
        if episode_number not in execution_by_number:
            raise NodeValidationError(f"episodes[{index}].episode_number is not part of the execution plan.")
        planned_episode = execution_by_number[episode_number]
        status = _require_str(episode.get("status"), f"episodes[{index}].status")
        if status not in {"submitted", "queued", "in_progress", "completed", "failed", "skipped"}:
            raise NodeValidationError(
                "render episode status must be submitted, queued, in_progress, completed, failed, or skipped."
            )
        if status in {"submitted", "queued", "in_progress"}:
            has_pending = True
        output_video = _require_str(episode.get("output_video"), f"episodes[{index}].output_video", allow_empty=True)
        segment_items = _require_list(episode.get("segments", []), f"episodes[{index}].segments")
        planned_segments = planned_episode.get("generation_segments", [])
        planned_segment_ids = [segment["segment_id"] for segment in planned_segments]
        normalized_segments: list[dict[str, Any]] = []
        seen_segment_ids: set[str] = set()
        completed_segment_outputs = 0
        for segment_index, segment_item in enumerate(segment_items):
            segment = _require_dict(segment_item, f"episodes[{index}].segments[{segment_index}]")
            segment_id = _require_str(segment.get("segment_id"), f"episodes[{index}].segments[{segment_index}].segment_id")
            if segment_id not in planned_segment_ids:
                raise NodeValidationError(
                    f"episodes[{index}].segments[{segment_index}].segment_id is not part of the execution plan."
                )
            if segment_id in seen_segment_ids:
                raise NodeValidationError(
                    f"episodes[{index}].segments contains duplicate segment_id `{segment_id}`."
                )
            seen_segment_ids.add(segment_id)
            segment_status = _require_str(segment.get("status"), f"episodes[{index}].segments[{segment_index}].status")
            if segment_status not in {"submitted", "queued", "in_progress", "completed", "failed", "skipped"}:
                raise NodeValidationError(
                    "render segment status must be submitted, queued, in_progress, completed, failed, or skipped."
                )
            if segment_status in {"submitted", "queued", "in_progress"}:
                has_pending = True
            segment_output = _require_str(
                segment.get("output_video"),
                f"episodes[{index}].segments[{segment_index}].output_video",
                allow_empty=True,
            )
            if segment_status == "completed" and not segment_output:
                raise NodeValidationError(
                    f"episodes[{index}].segments[{segment_index}].output_video must be set for completed renders."
                )
            if segment_output and not segment_output.startswith(("http://", "https://")):
                output_path = Path(segment_output)
                if not output_path.is_absolute():
                    output_path = (Path(config["project_root"]) / output_path).resolve()
                else:
                    output_path = output_path.resolve()
                if segment_status == "completed" and not output_path.exists():
                    raise NodeValidationError(
                        f"episodes[{index}].segments[{segment_index}].output_video points to a missing local file: {output_path}"
                    )
            if segment_status == "completed" and segment_output:
                completed_segment_outputs += 1
            normalized_segments.append(
                {
                    "segment_id": segment_id,
                    "task_id": _require_str(
                        segment.get("task_id", ""),
                        f"episodes[{index}].segments[{segment_index}].task_id",
                        allow_empty=True,
                    ),
                    "status": segment_status,
                    "output_video": segment_output,
                    "duration_seconds": _require_int(
                        segment.get("duration_seconds"), f"episodes[{index}].segments[{segment_index}].duration_seconds"
                    ),
                    "used_reference_ids": _require_str_list(
                        segment.get("used_reference_ids", []),
                        f"episodes[{index}].segments[{segment_index}].used_reference_ids",
                    ),
                    "used_material_ids": _require_str_list(
                        segment.get("used_material_ids", []),
                        f"episodes[{index}].segments[{segment_index}].used_material_ids",
                    ),
                    "notes": _require_str(
                        segment.get("notes", ""), f"episodes[{index}].segments[{segment_index}].notes", allow_empty=True
                    ),
                    "error": _require_str(
                        segment.get("error", ""), f"episodes[{index}].segments[{segment_index}].error", allow_empty=True
                    ),
                }
            )

        missing_segment_ids = [segment_id for segment_id in planned_segment_ids if segment_id not in seen_segment_ids]
        if missing_segment_ids:
            raise NodeValidationError(
                f"episodes[{index}].segments is missing planned segment ids: {', '.join(missing_segment_ids)}."
            )
        if status == "completed" and not output_video and completed_segment_outputs == 0:
            raise NodeValidationError(
                f"episodes[{index}] must include either a final output_video or completed segment outputs."
            )

        episode_reference_ids = _require_str_list(
            episode.get("used_reference_ids", []), f"episodes[{index}].used_reference_ids"
        )
        episode_material_ids = _require_str_list(
            episode.get("used_material_ids", []), f"episodes[{index}].used_material_ids"
        )
        if not episode_reference_ids:
            episode_reference_ids = sorted(
                {reference_id for segment in normalized_segments for reference_id in segment["used_reference_ids"]}
            )
        if not episode_material_ids:
            episode_material_ids = sorted(
                {material_id for segment in normalized_segments for material_id in segment["used_material_ids"]}
            )

        normalized_episodes.append(
            {
                "episode_number": episode_number,
                "task_id": _require_str(episode.get("task_id", ""), f"episodes[{index}].task_id", allow_empty=True),
                "status": status,
                "output_video": output_video,
                "duration_seconds": _require_int(episode.get("duration_seconds"), f"episodes[{index}].duration_seconds"),
                "used_reference_ids": episode_reference_ids,
                "used_material_ids": episode_material_ids,
                "notes": _require_str(episode.get("notes", ""), f"episodes[{index}].notes", allow_empty=True),
                "error": _require_str(episode.get("error", ""), f"episodes[{index}].error", allow_empty=True),
                "segments": normalized_segments,
            }
        )

    payload = {
        "series_title": execution_plan["series_title"],
        "provider": provider,
        "run_label": run_label,
        "episodes": normalized_episodes,
    }
    artifact_status = "in_progress" if has_pending else "completed"
    artifact = StageArtifact(
        stage="video_renderer",
        generated_at=utc_now(),
        status=artifact_status,
        summary="Segment render results imported." if artifact_status == "completed" else "Segment render tasks submitted.",
        required_inputs=["Execution plan and renderer outputs."],
        expected_outputs=["Normalized segment-level video manifest."],
        next_stage=None,
        payload=payload,
    )
    return artifact, render_video_manifest_markdown(config, payload)


def render_video_manifest_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['series_title']} - 视频生成结果",
        "",
        f"- Project: `{config['project_name']}`",
        f"- Provider: `{payload['provider']}`",
        f"- Run label: `{payload['run_label']}`",
        "",
        "| 视频 | 状态 | 总输出视频 | 参考ID | 素材库ID |",
        "|---|---|---|---|---|",
    ]
    for item in payload["episodes"]:
        reference_ids = ", ".join(item["used_reference_ids"]) if item["used_reference_ids"] else "-"
        material_ids = ", ".join(item["used_material_ids"]) if item["used_material_ids"] else "-"
        lines.append(
            f"| V{item['episode_number']:02d} | {item['status']} | {item['output_video'] or '-'} | {reference_ids} | {material_ids} |"
        )
        if item["notes"]:
            lines.append(f"|  | notes | {item['notes']} |  |  |")
        if item["error"]:
            lines.append(f"|  | error | {item['error']} |  |  |")
        lines.extend(["", f"## V{item['episode_number']:02d} Segments", "", "| Segment | 任务ID | 状态 | 输出视频 | 参考ID | 素材库ID |", "|---|---|---|---|---|---|"])
        for segment in item.get("segments", []):
            reference_ids = ", ".join(segment["used_reference_ids"]) if segment["used_reference_ids"] else "-"
            material_ids = ", ".join(segment["used_material_ids"]) if segment["used_material_ids"] else "-"
            lines.append(
                f"| {segment['segment_id']} | {segment.get('task_id') or '-'} | {segment['status']} | {segment['output_video'] or '-'} | {reference_ids} | {material_ids} |"
            )
            if segment["notes"]:
                lines.append(f"|  | notes | {segment['notes']} |  |  |  |")
            if segment["error"]:
                lines.append(f"|  | error | {segment['error']} |  |  |  |")
        lines.append("")
    return "\n".join(lines)


def prepare_qa_review(
    config: dict[str, Any],
    analysis: dict[str, Any],
    directions: dict[str, Any],
    script: dict[str, Any],
    assets: dict[str, Any],
    storyboards: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> PreparedRequest:
    referenced_asset_count = sum(
        1 for item in assets["characters"] + assets["scenes"] + assets["props"] if item["reference_ids"]
    )
    material_bound_asset_count = sum(
        1 for item in assets["characters"] + assets["scenes"] + assets["props"] if item.get("material_ids", [])
    )
    segment_count = sum(len(item.get("generation_segments", [])) for item in storyboards["episodes"])
    upload_slot_lines: list[str] = []
    for episode in storyboards["episodes"]:
        for slot in episode["upload_slots"]:
            reference_ids = ", ".join(slot.get("reference_ids", [])) if slot.get("reference_ids", []) else "none"
            material_ids = ", ".join(slot.get("material_ids", [])) if slot.get("material_ids", []) else "none"
            upload_slot_lines.append(
                f"- V{episode['episode_number']:02d} {slot['slot']} | asset={slot['asset_id']} | refs={reference_ids} | materials={material_ids}"
            )
    upload_slot_summary = "\n".join(upload_slot_lines) if upload_slot_lines else "- none"
    prompt = f"""# QA Review Request

你是“AI 视频生产 QA 审查”专家。现在你需要审查整条链路是否可执行：视频解析、方向规划、裂变脚本、素材库、分镜与执行前条件是否一致。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`

## Current Workflow Mode

- 这不是多集短剧流程，而是“单条独立视频”流程。
- `storyboards.episodes` 当前允许只有 1 条视频；不要因为只有 1 条 video storyboard 就判定失败。
- 连续性应主要检查 `generation_segments` 是否覆盖完整视频计划，而不是检查“集数是否足够”。

## Audit anchors

- Working title: {analysis["source_overview"]["working_title"]}
- Selected direction: {directions["selected_direction_id"]}
- Series title: {script["series_title"]}
- Asset count: {len(assets["characters"]) + len(assets["scenes"]) + len(assets["props"])}
- Reference assets registered: {len(reference_library["references"])}
- Materials registered: {len(material_library["materials"])}
- Assets depending on references: {referenced_asset_count}
- Assets bound to existing materials: {material_bound_asset_count}
- Standalone video count: {len(storyboards["episodes"])}
- Generation segment count: {segment_count}

## Known storyboard bindings

这些绑定已经在 storyboards JSON 中显式存在，应视为已知事实，不要误判为“缺失引用”：
{upload_slot_summary}

## Review goals

1. 检查 must_keep 事实有没有被破坏，但允许标题做轻度裂变包装；只有在核心事件被改写成另一件事时才判高风险。
2. 检查剧本、素材、分镜之间的素材引用和叙事连贯性。
3. 审查素材引用时，以 storyboards JSON 里的 `upload_slots.reference_ids` / `upload_slots.material_ids` 为准；如果字段已存在且非空，不要再报告“缺失 ID”。
4. 检查 `generation_segments` 是否覆盖单条视频计划、镜头衔接是否合理。
5. 标出高/中/低风险问题。
6. 给出是否可进入实际生成阶段的判断。

## Output requirements

- 严格输出 JSON。
- `overall_status` 只能是 `pass`、`needs_revision`、`fail`。
- `checklist` 至少覆盖事实一致性、素材完整性、分段连续性、生成可执行性。
- 如果主要问题只是文案层轻微偏差、而素材引用与执行链路已经闭合，应优先给 `needs_revision` 或 `pass`，不要夸大为阻塞性失败。
"""
    template = {
        "overall_status": "needs_revision",
        "summary": "",
        "ready_for_generation": False,
        "checklist": [
            {
                "check": "fact_integrity",
                "status": "pass",
                "details": "",
            }
        ],
        "findings": [
            {
                "severity": "medium",
                "area": "",
                "issue": "",
                "recommendation": "",
            }
        ],
        "revision_targets": [
            {
                "stage": "gen-assets",
                "reason": "",
            }
        ],
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a QA review request.")


def finalize_qa_review(
    config: dict[str, Any],
    analysis: dict[str, Any],
    directions: dict[str, Any],
    script: dict[str, Any],
    assets: dict[str, Any],
    storyboards: dict[str, Any],
    response: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> tuple[StageArtifact, str]:
    overall_status = _require_str(response.get("overall_status"), "overall_status")
    if overall_status not in {"pass", "needs_revision", "fail"}:
        raise NodeValidationError("overall_status must be one of pass, needs_revision, fail.")
    summary = _require_str(response.get("summary"), "summary")
    ready_for_generation = _require_bool(response.get("ready_for_generation"), "ready_for_generation")
    checklist = _require_list(response.get("checklist"), "checklist")
    findings = _require_list(response.get("findings"), "findings")
    revision_targets = _require_list(response.get("revision_targets"), "revision_targets")

    payload = {
        "overall_status": overall_status,
        "summary": summary,
        "ready_for_generation": ready_for_generation,
        "audit_anchor": {
            "working_title": analysis["source_overview"]["working_title"],
            "direction_id": directions["selected_direction_id"],
            "series_title": script["series_title"],
            "asset_count": len(assets["characters"]) + len(assets["scenes"]) + len(assets["props"]),
            "reference_asset_count": len(reference_library["references"]),
            "material_library_count": len(material_library["materials"]),
            "storyboard_episode_count": len(storyboards["episodes"]),
        },
        "checklist": [
            {
                "check": _require_str(_require_dict(item, f"checklist[{index}]").get("check"), f"checklist[{index}].check"),
                "status": _require_str(_require_dict(item, f"checklist[{index}]").get("status"), f"checklist[{index}].status"),
                "details": _require_str(_require_dict(item, f"checklist[{index}]").get("details"), f"checklist[{index}].details"),
            }
            for index, item in enumerate(checklist)
        ],
        "findings": [
            {
                "severity": _require_str(_require_dict(item, f"findings[{index}]").get("severity"), f"findings[{index}].severity"),
                "area": _require_str(_require_dict(item, f"findings[{index}]").get("area"), f"findings[{index}].area"),
                "issue": _require_str(_require_dict(item, f"findings[{index}]").get("issue"), f"findings[{index}].issue"),
                "recommendation": _require_str(
                    _require_dict(item, f"findings[{index}]").get("recommendation"),
                    f"findings[{index}].recommendation",
                ),
            }
            for index, item in enumerate(findings)
        ],
        "revision_targets": [
            {
                "stage": _require_str(_require_dict(item, f"revision_targets[{index}]").get("stage"), f"revision_targets[{index}].stage"),
                "reason": _require_str(_require_dict(item, f"revision_targets[{index}]").get("reason"), f"revision_targets[{index}].reason"),
            }
            for index, item in enumerate(revision_targets)
        ],
    }

    artifact = StageArtifact(
        stage="qa_reviewer",
        generated_at=utc_now(),
        status="completed",
        summary="QA review completed.",
        required_inputs=["All upstream artifacts."],
        expected_outputs=["Readiness decision and revision targets."],
        next_stage="execution_planner",
        payload=payload,
    )
    markdown = render_qa_markdown(config, payload)
    return artifact, markdown


def build_local_qa_review(
    config: dict[str, Any],
    analysis: dict[str, Any],
    directions: dict[str, Any],
    script: dict[str, Any],
    assets: dict[str, Any],
    storyboards: dict[str, Any],
    reference_library: dict[str, Any],
    material_library: dict[str, Any],
) -> tuple[StageArtifact, str]:
    findings: list[dict[str, str]] = []
    revision_targets: list[dict[str, str]] = []
    checklist: list[dict[str, str]] = []

    reference_ids = {item["reference_id"] for item in reference_library["references"]}
    material_ids = {item["material_id"] for item in material_library["materials"]}
    assets_by_id = {item["asset_id"]: item for item in assets["characters"] + assets["scenes"] + assets["props"]}
    planned_segments = script["video_plan"]["generation_segments"]
    planned_segment_ids = [item["segment_id"] for item in planned_segments]
    planned_total_duration = sum(int(item["duration_seconds"]) for item in planned_segments)

    asset_binding_issues = 0
    for asset in assets_by_id.values():
        sourcing_mode = asset["sourcing_mode"]
        asset_refs = asset.get("reference_ids", [])
        asset_mats = asset.get("material_ids", [])
        if sourcing_mode in {"generate_from_reference", "use_reference_directly"} and not asset_refs:
            asset_binding_issues += 1
            findings.append(
                {
                    "severity": "high",
                    "area": "asset-binding",
                    "issue": f"Asset {asset['asset_id']} requires reference_ids for sourcing_mode={sourcing_mode}.",
                    "recommendation": "补全对应素材的 reference_ids 绑定。",
                }
            )
        if sourcing_mode == "use_material_library_directly" and not asset_mats:
            asset_binding_issues += 1
            findings.append(
                {
                    "severity": "high",
                    "area": "asset-binding",
                    "issue": f"Asset {asset['asset_id']} requires material_ids for sourcing_mode={sourcing_mode}.",
                    "recommendation": "补全对应素材的 material_ids 绑定。",
                }
            )
        for reference_id in asset_refs:
            if reference_id not in reference_ids:
                asset_binding_issues += 1
                findings.append(
                    {
                        "severity": "high",
                        "area": "asset-binding",
                        "issue": f"Asset {asset['asset_id']} references unknown reference_id {reference_id}.",
                        "recommendation": "修正 reference_assets.json 或素材绑定中的 reference_id。",
                    }
                )
        for material_id in asset_mats:
            if material_id not in material_ids:
                asset_binding_issues += 1
                findings.append(
                    {
                        "severity": "high",
                        "area": "asset-binding",
                        "issue": f"Asset {asset['asset_id']} references unknown material_id {material_id}.",
                        "recommendation": "修正 material_library.json 或素材绑定中的 material_id。",
                    }
                )

    slot_binding_issues = 0
    segment_issues = 0
    for episode in storyboards["episodes"]:
        for slot in episode["upload_slots"]:
            asset_id = slot["asset_id"]
            if asset_id not in assets_by_id:
                slot_binding_issues += 1
                findings.append(
                    {
                        "severity": "high",
                        "area": "storyboard-binding",
                        "issue": f"Storyboard slot {slot['slot']} references unknown asset_id {asset_id}.",
                        "recommendation": "修正分镜 upload_slots 中的 asset_id。",
                    }
                )
                continue
            asset = assets_by_id[asset_id]
            slot_refs = slot.get("reference_ids", [])
            slot_mats = slot.get("material_ids", [])
            for reference_id in slot_refs:
                if reference_id not in reference_ids:
                    slot_binding_issues += 1
                    findings.append(
                        {
                            "severity": "high",
                            "area": "storyboard-binding",
                            "issue": f"Storyboard slot {slot['slot']} references unknown reference_id {reference_id}.",
                            "recommendation": "修正分镜 upload_slots 的 reference_ids。",
                        }
                    )
                if reference_id not in asset.get("reference_ids", []):
                    slot_binding_issues += 1
                    findings.append(
                        {
                            "severity": "high",
                            "area": "storyboard-binding",
                            "issue": f"Storyboard slot {slot['slot']} uses reference_id {reference_id} not bound to asset {asset_id}.",
                            "recommendation": "让分镜引用仅使用该 asset 已绑定的 reference_ids。",
                        }
                    )
            for material_id in slot_mats:
                if material_id not in material_ids:
                    slot_binding_issues += 1
                    findings.append(
                        {
                            "severity": "high",
                            "area": "storyboard-binding",
                            "issue": f"Storyboard slot {slot['slot']} references unknown material_id {material_id}.",
                            "recommendation": "修正分镜 upload_slots 的 material_ids。",
                        }
                    )
                if material_id not in asset.get("material_ids", []):
                    slot_binding_issues += 1
                    findings.append(
                        {
                            "severity": "high",
                            "area": "storyboard-binding",
                            "issue": f"Storyboard slot {slot['slot']} uses material_id {material_id} not bound to asset {asset_id}.",
                            "recommendation": "让分镜引用仅使用该 asset 已绑定的 material_ids。",
                        }
                    )
            if asset["sourcing_mode"] in {"generate_from_reference", "use_reference_directly"} and not slot_refs:
                slot_binding_issues += 1
                findings.append(
                    {
                        "severity": "high",
                        "area": "storyboard-binding",
                        "issue": f"Storyboard slot {slot['slot']} for asset {asset_id} is missing required reference_ids.",
                        "recommendation": "在 upload_slots 中补全 reference_ids。",
                    }
                )
            if asset["sourcing_mode"] == "use_material_library_directly" and not slot_mats:
                slot_binding_issues += 1
                findings.append(
                    {
                        "severity": "high",
                        "area": "storyboard-binding",
                        "issue": f"Storyboard slot {slot['slot']} for asset {asset_id} is missing required material_ids.",
                        "recommendation": "在 upload_slots 中补全 material_ids。",
                    }
                )

        segment_ids = [segment["segment_id"] for segment in episode.get("generation_segments", [])]
        if segment_ids != planned_segment_ids:
            segment_issues += 1
            findings.append(
                {
                    "severity": "high",
                    "area": "segment-plan",
                    "issue": f"Storyboard generation_segments {segment_ids} do not match script video_plan {planned_segment_ids}.",
                    "recommendation": "让 storyboards 与 script.video_plan 使用完全一致的 segment_id 顺序。",
                }
            )
        episode_total_duration = sum(int(segment["duration_seconds"]) for segment in episode.get("generation_segments", []))
        if episode_total_duration != planned_total_duration:
            segment_issues += 1
            findings.append(
                {
                    "severity": "medium",
                    "area": "segment-plan",
                    "issue": f"Storyboard segment duration total {episode_total_duration}s does not match script plan {planned_total_duration}s.",
                    "recommendation": "统一 storyboards 与 script.video_plan 的总时长。",
                }
            )

    if asset_binding_issues == 0:
        checklist.append(
            {
                "check": "asset_binding_integrity",
                "status": "pass",
                "details": "素材层的 reference_ids / material_ids 绑定完整，且都指向已注册 ID。",
            }
        )
    else:
        checklist.append(
            {
                "check": "asset_binding_integrity",
                "status": "fail",
                "details": f"发现 {asset_binding_issues} 个素材绑定问题。",
            }
        )
        revision_targets.append({"stage": "gen-assets", "reason": "修正素材层 reference_ids / material_ids 绑定。"})

    if slot_binding_issues == 0:
        checklist.append(
            {
                "check": "storyboard_binding_integrity",
                "status": "pass",
                "details": "分镜上传表中的 asset_id / reference_ids / material_ids 与素材库绑定一致。",
            }
        )
    else:
        checklist.append(
            {
                "check": "storyboard_binding_integrity",
                "status": "fail",
                "details": f"发现 {slot_binding_issues} 个分镜绑定问题。",
            }
        )
        revision_targets.append({"stage": "gen-storyboards", "reason": "修正 upload_slots 中的 asset_id / reference_ids / material_ids。"})

    if segment_issues == 0:
        checklist.append(
            {
                "check": "segment_plan_integrity",
                "status": "pass",
                "details": "单条视频的 generation_segments 与 script.video_plan 一致。",
            }
        )
    else:
        checklist.append(
            {
                "check": "segment_plan_integrity",
                "status": "fail",
                "details": f"发现 {segment_issues} 个 generation_segments 结构问题。",
            }
        )
        revision_targets.append({"stage": "gen-storyboards", "reason": "让 generation_segments 与 script.video_plan 完全对齐。"})

    ready_for_generation = asset_binding_issues == 0 and slot_binding_issues == 0 and segment_issues == 0
    overall_status = "pass" if ready_for_generation else "needs_revision"
    summary = (
        "本地交叉校验通过：素材绑定、分镜引用和分段计划均已闭合，可进入执行层。"
        if ready_for_generation
        else "本地交叉校验发现阻塞项：请先修正素材绑定、分镜引用或分段计划不一致问题。"
    )

    payload = {
        "overall_status": overall_status,
        "summary": summary,
        "ready_for_generation": ready_for_generation,
        "audit_anchor": {
            "working_title": analysis["source_overview"]["working_title"],
            "direction_id": directions["selected_direction_id"],
            "series_title": script["series_title"],
            "asset_count": len(assets["characters"]) + len(assets["scenes"]) + len(assets["props"]),
            "reference_asset_count": len(reference_library["references"]),
            "material_library_count": len(material_library["materials"]),
            "storyboard_episode_count": len(storyboards["episodes"]),
            "generation_segment_count": sum(len(item.get("generation_segments", [])) for item in storyboards["episodes"]),
        },
        "checklist": checklist,
        "findings": findings,
        "revision_targets": revision_targets,
    }

    artifact = StageArtifact(
        stage="qa_reviewer",
        generated_at=utc_now(),
        status="completed",
        summary="Local QA cross-check completed.",
        required_inputs=["All upstream artifacts."],
        expected_outputs=["Readiness decision from local structural validation."],
        next_stage="execution_planner",
        payload=payload,
    )
    markdown = render_qa_markdown(config, payload)
    return artifact, markdown


def render_qa_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {config['project_name']} - QA 审查",
        "",
        f"- 总体状态：{payload['overall_status']}",
        f"- 可直接生成：{payload['ready_for_generation']}",
        "",
        "## 摘要",
        "",
        payload["summary"],
        "",
        "## 审查清单",
        "",
    ]
    for item in payload["checklist"]:
        lines.append(f"- {item['check']} [{item['status']}]：{item['details']}")
    lines.extend(["", "## 问题发现", ""])
    for item in payload["findings"]:
        lines.append(f"- {item['severity']} | {item['area']} | {item['issue']} | 建议：{item['recommendation']}")
    lines.extend(["", "## 返工目标", ""])
    for item in payload["revision_targets"]:
        lines.append(f"- {item['stage']}：{item['reason']}")
    lines.append("")
    return "\n".join(lines)
