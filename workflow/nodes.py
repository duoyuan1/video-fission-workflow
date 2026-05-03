"""Stage-specific logic for the first three workflow nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow.constants import STAGE_BY_COMMAND, STAGE_BY_NAME
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


def _require_str(data: Any, path: str) -> str:
    if not isinstance(data, str) or not data.strip():
        raise NodeValidationError(f"{path} must be a non-empty string.")
    return data.strip()


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
        prepared = prepare_video_analysis(config)
    elif command == "plan-directions":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        prepared = prepare_direction_planning(config, analysis)
    elif command == "gen-script":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        directions = load_artifact_payload(project_root, "direction_planner")
        prepared = prepare_script_generation(config, analysis, directions, direction_id=direction_id)
    elif command == "gen-assets":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        script = load_artifact_payload(project_root, "script_generator")
        reference_library = load_reference_library(project_root)
        prepared = prepare_asset_planning(config, analysis, script, reference_library)
    elif command == "gen-storyboards":
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        reference_library = load_reference_library(project_root)
        prepared = prepare_storyboard_generation(config, script, assets, reference_library)
    elif command == "qa":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        directions = load_artifact_payload(project_root, "direction_planner")
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        storyboards = load_artifact_payload(project_root, "storyboard_generator")
        reference_library = load_reference_library(project_root)
        prepared = prepare_qa_review(config, analysis, directions, script, assets, storyboards, reference_library)
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
        return finalize_asset_planning(config, analysis, script, response, reference_library)
    if command == "gen-storyboards":
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        reference_library = load_reference_library(project_root)
        return finalize_storyboard_generation(config, script, assets, response, reference_library)
    if command == "qa":
        analysis = load_artifact_payload(project_root, "video_analyzer")
        directions = load_artifact_payload(project_root, "direction_planner")
        script = load_artifact_payload(project_root, "script_generator")
        assets = load_artifact_payload(project_root, "asset_planner")
        storyboards = load_artifact_payload(project_root, "storyboard_generator")
        reference_library = load_reference_library(project_root)
        return finalize_qa_review(config, analysis, directions, script, assets, storyboards, response, reference_library)
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


def prepare_video_analysis(config: dict[str, Any]) -> PreparedRequest:
    prompt = f"""# Video Analysis Request

你是“视频内容理解与结构化拆解”专家。你的任务不是改写，不是创作新剧情，而是先忠实理解输入视频，为后续“方向规划 -> 裂变剧本 -> 素材库 -> 分镜”提供干净的结构化基础。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Target platform: `{config["platform"]}`
- Target aspect ratio: `{config["aspect_ratio"]}`
- Default episode duration: `{config["episode_duration_seconds"]}` seconds
- Target episode count: `{config["target_episode_count"]}`
- Locale: `{config["locale"]}`

## Your goals

1. 识别视频中的核心叙事：开场、冲突、转折、结局、主题。
2. 提取角色、场景、关键动作、情绪变化、视觉记忆点。
3. 标出适合做“内容裂变”的爆点：钩子、反差、反转、情绪峰值、视觉奇观。
4. 给后续阶段保留真实边界：哪些事实必须保留，哪些元素可以二次创作。

## Output requirements

- 严格输出 JSON。
- 保持忠于视频内容，不要提前编造完整新剧本。
- 角色、场景、镜头节拍尽量具体。
- `hook_score` 使用 1-5 分。
- `adaptation_facts` 里区分 `must_keep`、`flexible`、`avoid`。

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
        "viral_elements": [
            {
                "element": "",
                "reason": "",
                "recommended_use": "",
            }
        ],
        "adaptation_facts": {
            "must_keep": [""],
            "flexible": [""],
            "avoid": [""],
        },
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a structured video analysis request.")


def finalize_video_analysis(config: dict[str, Any], response: dict[str, Any]) -> tuple[StageArtifact, str]:
    source_overview = _require_dict(response.get("source_overview"), "source_overview")
    core_narrative = _require_dict(response.get("core_narrative"), "core_narrative")
    characters = _require_list(response.get("characters"), "characters")
    scenes = _require_list(response.get("scenes"), "scenes")
    beats = _require_list(response.get("beats"), "beats")
    viral_elements = _require_list(response.get("viral_elements"), "viral_elements")
    adaptation_facts = _require_dict(response.get("adaptation_facts"), "adaptation_facts")

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
        "viral_elements": [
            {
                "element": _require_str(_require_dict(item, f"viral_elements[{index}]").get("element"), f"viral_elements[{index}].element"),
                "reason": _require_str(_require_dict(item, f"viral_elements[{index}]").get("reason"), f"viral_elements[{index}].reason"),
                "recommended_use": _require_str(
                    _require_dict(item, f"viral_elements[{index}]").get("recommended_use"),
                    f"viral_elements[{index}].recommended_use",
                ),
            }
            for index, item in enumerate(viral_elements)
        ],
        "adaptation_facts": {
            "must_keep": _require_str_list(adaptation_facts.get("must_keep"), "adaptation_facts.must_keep"),
            "flexible": _require_str_list(adaptation_facts.get("flexible"), "adaptation_facts.flexible"),
            "avoid": _require_str_list(adaptation_facts.get("avoid"), "adaptation_facts.avoid"),
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
    lines.extend(["", "## 裂变机会点", ""])
    for item in payload["viral_elements"]:
        lines.extend([f"- {item['element']}：{item['reason']}；建议：{item['recommended_use']}"])
    lines.extend(["", "## 改编边界", ""])
    lines.extend(f"- 必须保留：{item}" for item in payload["adaptation_facts"]["must_keep"])
    lines.extend(f"- 可灵活处理：{item}" for item in payload["adaptation_facts"]["flexible"])
    lines.extend(f"- 避免误读：{item}" for item in payload["adaptation_facts"]["avoid"])
    lines.append("")
    return "\n".join(lines)


def prepare_direction_planning(config: dict[str, Any], analysis: dict[str, Any]) -> PreparedRequest:
    prompt = f"""# Direction Planning Request

你是“内容裂变方向规划”专家。现在你已经拿到了原视频的结构化解析，请不要重复做视频理解，而是基于这份解析，提出 3-5 个明确区分的改编方向。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Target platform: `{config["platform"]}`
- Target aspect ratio: `{config["aspect_ratio"]}`
- Target episode duration: `{config["episode_duration_seconds"]}` seconds
- Target episode count: `{config["target_episode_count"]}`

## Analysis Summary

- Working title: {analysis["source_overview"]["working_title"]}
- Story type: {analysis["source_overview"]["story_type"]}
- Theme: {analysis["core_narrative"]["theme"]}
- Must keep facts: {' / '.join(analysis["adaptation_facts"]["must_keep"])}
- Viral elements: {' / '.join(item["element"] for item in analysis["viral_elements"])}

## Planning goals

1. 给出 3-5 个明显不同的裂变方向，不要只是换同义词。
2. 每个方向都要说明目标受众、情绪曲线、内容钩子、适合的平台打法。
3. 给出推荐方向，并解释为什么它最适合进入剧本阶段。
4. 保持与原视频解析一致，不要违背 must_keep 事实。

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
                "episode_plan": {
                    "episode_count": config["target_episode_count"],
                    "episode_duration_seconds": config["episode_duration_seconds"],
                },
                "sample_title": "",
                "sample_opening": "",
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

    normalized_directions: list[dict[str, Any]] = []
    recommended_ids: list[str] = []
    for index, item in enumerate(directions):
        direction = _require_dict(item, f"directions[{index}]")
        episode_plan = _require_dict(direction.get("episode_plan"), f"directions[{index}].episode_plan")
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
            "episode_plan": {
                "episode_count": _require_int(episode_plan.get("episode_count"), f"directions[{index}].episode_plan.episode_count"),
                "episode_duration_seconds": _require_int(
                    episode_plan.get("episode_duration_seconds"),
                    f"directions[{index}].episode_plan.episode_duration_seconds",
                ),
            },
            "sample_title": _require_str(direction.get("sample_title"), f"directions[{index}].sample_title"),
            "sample_opening": _require_str(direction.get("sample_opening"), f"directions[{index}].sample_opening"),
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
            "must_keep": analysis["adaptation_facts"]["must_keep"],
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
                f"- 集数建议：{item['episode_plan']['episode_count']} x {item['episode_plan']['episode_duration_seconds']}s",
                f"- 示例标题：{item['sample_title']}",
                f"- 示例开头：{item['sample_opening']}",
                f"- 评分：{item['score']}",
                "",
            ]
        )
    lines.extend(["## 选择理由", "", payload["selection_reason"], ""])
    return "\n".join(lines)


def prepare_script_generation(
    config: dict[str, Any], analysis: dict[str, Any], directions: dict[str, Any], direction_id: str | None = None
) -> PreparedRequest:
    selected = choose_direction(directions, direction_id)
    prompt = f"""# Script Generation Request

你是“裂变短剧编剧”专家。请基于原视频解析和已选方向，输出一个可以直接进入素材库与分镜阶段的短剧脚本包。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Aspect ratio: `{config["aspect_ratio"]}`
- Episode duration: `{config["episode_duration_seconds"]}` seconds
- Episode count: `{config["target_episode_count"]}`

## Analysis Anchor

- Working title: {analysis["source_overview"]["working_title"]}
- Story theme: {analysis["core_narrative"]["theme"]}
- Must keep facts: {' / '.join(analysis["adaptation_facts"]["must_keep"])}
- Flexible facts: {' / '.join(analysis["adaptation_facts"]["flexible"])}

## Selected Direction

- Direction ID: {selected["direction_id"]}
- Name: {selected["name"]}
- Positioning: {selected["positioning"]}
- Hook: {selected["hook"]}
- Tone: {selected["tone"]}
- Emotion curve: {' -> '.join(selected["emotion_curve"])}
- Episode plan: {selected["episode_plan"]["episode_count"]} x {selected["episode_plan"]["episode_duration_seconds"]}s

## Writing requirements

1. 保留 must_keep 事实，不要脱离原视频根基。
2. 允许在 flexible 边界内强化戏剧张力和传播钩子。
3. 输出必须同时包含结构化信息和每集可读的脚本正文。
4. 每集正文参考现有 skill 的格式，至少包含：
   - `第X集`
   - `道具：`
   - `出场人物：`
   - 若干以 `△ ` 开头的镜头描述
5. 情绪弧线和方向定位要一致，不要写成普通流水账。

## Output requirements

- 严格输出 JSON。
- `episodes` 数量要与选定方向一致。
- 每集 `script_body` 必须是可直接写入 markdown 的多行文本。
"""

    template = {
        "direction_id": selected["direction_id"],
        "series_title": "",
        "core_hook": "",
        "one_line_sell": "",
        "audience_promise": "",
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
                "script_body": "第1集\n1-1 日 外 场景名\n道具：\n出场人物：\n\n△ 镜头描述",
            }
        ],
        "production_notes": {
            "visual_emphasis": [""],
            "dialogue_style": "",
            "continuity_rules": [""],
        },
    }
    return PreparedRequest(prompt=prompt, response_template=template, summary="Prepare a裂变剧本 generation request.")


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

    expected_episode_count = selected["episode_plan"]["episode_count"]
    if len(episodes) != expected_episode_count:
        raise NodeValidationError(
            f"episodes must contain exactly {expected_episode_count} items for the selected direction."
        )

    normalized_episodes: list[dict[str, Any]] = []
    for index, item in enumerate(episodes):
        episode = _require_dict(item, f"episodes[{index}]")
        episode_number = _require_int(episode.get("episode_number"), f"episodes[{index}].episode_number")
        if episode_number != index + 1:
            raise NodeValidationError("episodes must be ordered sequentially starting at 1.")
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
                "cliffhanger": _require_str(episode.get("cliffhanger"), f"episodes[{index}].cliffhanger"),
                "script_body": script_body,
            }
        )

    payload = {
        "direction_id": response_direction_id,
        "direction_name": selected["name"],
        "series_title": _require_str(response.get("series_title"), "series_title"),
        "core_hook": _require_str(response.get("core_hook"), "core_hook"),
        "one_line_sell": _require_str(response.get("one_line_sell"), "one_line_sell"),
        "audience_promise": _require_str(response.get("audience_promise"), "audience_promise"),
        "analysis_anchor": {
            "working_title": analysis["source_overview"]["working_title"],
            "must_keep": analysis["adaptation_facts"]["must_keep"],
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
        summary="裂变剧本 package ready for asset planning.",
        required_inputs=["Completed analysis artifact and selected direction artifact."],
        expected_outputs=["Series script package with episode-ready markdown."],
        next_stage="asset_planner",
        payload=payload,
    )
    markdown = render_script_markdown(config, payload)
    return artifact, markdown


def render_script_markdown(config: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['series_title']} - 裂变剧本",
        "",
        f"- Project: `{config['project_name']}`",
        f"- Direction: `{payload['direction_id']}` {payload['direction_name']}",
        f"- 核心梗：{payload['core_hook']}",
        f"- 一句话卖点：{payload['one_line_sell']}",
        f"- 受众承诺：{payload['audience_promise']}",
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
            "## 剧本大纲",
            "",
            f"- 起：{payload['story_outline']['setup']}",
            f"- 承：{payload['story_outline']['development']}",
            f"- 转/高潮：{payload['story_outline']['climax']}",
            f"- 合：{payload['story_outline']['resolution']}",
            "",
            "## 生产提示",
            "",
        ]
    )
    lines.extend(f"- 视觉重点：{item}" for item in payload["production_notes"]["visual_emphasis"])
    lines.append(f"- 对白风格：{payload['production_notes']['dialogue_style']}")
    lines.extend(f"- 连贯规则：{item}" for item in payload["production_notes"]["continuity_rules"])
    lines.extend(["", "## 分集正文", ""])
    for item in payload["episodes"]:
        lines.extend(
            [
                f"### E{item['episode_number']:02d} {item['title']}",
                "",
                f"- 作用：{item['purpose']}",
                f"- 情绪：{item['emotion']}",
                f"- 摘要：{item['summary']}",
                f"- 尾钩：{item['cliffhanger']}",
                "",
                item["script_body"],
                "",
            ]
        )
    return "\n".join(lines)


def prepare_asset_planning(
    config: dict[str, Any], analysis: dict[str, Any], script: dict[str, Any], reference_library: dict[str, Any]
) -> PreparedRequest:
    reference_summary = render_reference_library_summary(reference_library)
    prompt = f"""# Asset Planning Request

你是“AI 视频素材库规划”专家。现在你已经有视频解析和裂变剧本，请不要再改写剧情，而是把它们转成可执行的素材库规划。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`
- Aspect ratio: `{config["aspect_ratio"]}`

## Analysis Anchor

- Working title: {analysis["source_overview"]["working_title"]}
- Visual style: {analysis["source_overview"]["visual_style"]}
- Must keep facts: {' / '.join(analysis["adaptation_facts"]["must_keep"])}

## Script Anchor

- Series title: {script["series_title"]}
- Direction: {script["direction_name"]}
- Visual emphasis: {' / '.join(script["production_notes"]["visual_emphasis"])}
- Continuity rules: {' / '.join(script["production_notes"]["continuity_rules"])}

## External Reference Library

{reference_summary}

## Planning goals

1. 产出统一风格前缀和素材一致性规则。
2. 将素材拆成角色 C、场景 S、道具 P 三类。
3. 对于用户已提供参考素材的人物、表情包、IP 形象、固定道具，不要把它们当成纯原创资产，必须标注引用方式。
4. 每项素材都要写明用途、优先级、视觉描述、生成提示词，以及素材来源策略。
5. 给出“必须先生成”的最小素材集，方便开工。

## Output requirements

- 严格输出 JSON。
- `asset_id` 必须使用 `C` / `S` / `P` 前缀。
- `sourcing_mode` 只能是 `generate_fresh` / `generate_from_reference` / `use_reference_directly`。
- 如果使用参考素材，`reference_ids` 必须引用 External Reference Library 里的 `reference_id`。
- `generation_prompt` 用适合图像模型的完整描述。
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
    allowed_sourcing_modes = {"generate_fresh", "generate_from_reference", "use_reference_directly"}

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
                    f"{path}[{index}].sourcing_mode must be one of generate_fresh / generate_from_reference / use_reference_directly."
                )
            reference_ids = _require_str_list(asset.get("reference_ids", []), f"{path}[{index}].reference_ids")
            for reference_id in reference_ids:
                if reference_id not in available_reference_ids:
                    raise NodeValidationError(f"{path}[{index}] references undefined reference asset `{reference_id}`.")
            if sourcing_mode != "generate_fresh" and not reference_ids:
                raise NodeValidationError(f"{path}[{index}].reference_ids must not be empty for `{sourcing_mode}`.")
            normalized.append(
                {
                    "asset_id": asset_id,
                    "name": _require_str(asset.get("name"), f"{path}[{index}].name"),
                    "purpose": _require_str(asset.get("purpose"), f"{path}[{index}].purpose"),
                    "priority": _require_str(asset.get("priority"), f"{path}[{index}].priority"),
                    "sourcing_mode": sourcing_mode,
                    "reference_ids": reference_ids,
                    "reference_notes": _require_str(asset.get("reference_notes"), f"{path}[{index}].reference_notes"),
                    "visual_description": _require_str(
                        asset.get("visual_description"), f"{path}[{index}].visual_description"
                    ),
                    "generation_prompt": _require_str(
                        asset.get("generation_prompt"), f"{path}[{index}].generation_prompt"
                    ),
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
                "note": _require_str(entry.get("note"), f"reuse_plan[{index}].note"),
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
    config: dict[str, Any], script: dict[str, Any], assets: dict[str, Any], reference_library: dict[str, Any]
) -> PreparedRequest:
    reference_summary = render_reference_library_summary(reference_library)
    asset_source_lines = []
    for item in assets["characters"] + assets["scenes"] + assets["props"]:
        references = ", ".join(item["reference_ids"]) if item["reference_ids"] else "none"
        asset_source_lines.append(
            f"- {item['asset_id']} {item['name']} | {item['sourcing_mode']} | refs: {references} | notes: {item['reference_notes']}"
        )
    asset_source_summary = "\n".join(asset_source_lines)
    prompt = f"""# Storyboard Generation Request

你是“Seedance / AI 视频分镜规划”专家。请基于已经完成的裂变剧本和素材库，生成每一集可执行的分镜提示词。

## Project Context

- Project: `{config["project_name"]}`
- Aspect ratio: `{config["aspect_ratio"]}`
- Episode duration: `{config["episode_duration_seconds"]}` seconds
- Episode count: `{config["target_episode_count"]}`

## Script Anchor

- Series title: {script["series_title"]}
- Direction: {script["direction_name"]}
- Visual emphasis: {' / '.join(script["production_notes"]["visual_emphasis"])}
- Continuity rules: {' / '.join(script["production_notes"]["continuity_rules"])}

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

## Generation goals

1. 每集输出上传素材表、15 秒时间轴 prompt、尾帧描述。
2. 第 2 集开始，默认用 `将@视频1延长15s` 串联。
3. Prompt 要具体到镜头、动作、情绪和声音。
4. 素材引用必须只使用素材库里已定义的 ID。
5. 如果某项素材依赖参考图，上传表里要明确写出 `reference_ids`，提醒执行时一并上传。

## Output requirements

- 严格输出 JSON。
- `episodes` 数量必须和剧本一致。
- `upload_slots[].material_type` 只能是 `asset` 或 `reference`。
- `timeline_prompt` 必须包含 0-3 秒到 12-15 秒的分段。
- 第 2 集及之后若使用续接，需在 `timeline_prompt` 中体现 `将@视频1延长15s`。
"""
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
                        "usage": "",
                    }
                ],
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
    available_reference_ids = {item["reference_id"] for item in reference_library["references"]}
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
            for reference_id in reference_ids:
                if reference_id not in available_reference_ids:
                    raise NodeValidationError(
                        f"episodes[{index}].upload_slots[{slot_index}] references undefined reference `{reference_id}`."
                    )
            if material_type == "reference" and not reference_ids:
                raise NodeValidationError(
                    f"episodes[{index}].upload_slots[{slot_index}].reference_ids must not be empty for reference slots."
                )
            if reference_ids and not set(reference_ids).issubset(asset_reference_map[asset_id]):
                raise NodeValidationError(
                    f"episodes[{index}].upload_slots[{slot_index}] includes reference_ids not bound to asset `{asset_id}`."
                )
            normalized_slots.append(
                {
                    "slot": _require_str(slot.get("slot"), f"episodes[{index}].upload_slots[{slot_index}].slot"),
                    "material_type": material_type,
                    "asset_id": asset_id,
                    "reference_ids": reference_ids,
                    "usage": _require_str(slot.get("usage"), f"episodes[{index}].upload_slots[{slot_index}].usage"),
                }
            )
        timeline_prompt = _require_str(episode.get("timeline_prompt"), f"episodes[{index}].timeline_prompt")
        for required_marker in ["0-3", "3-6", "6-9", "9-12", "12-15"]:
            if required_marker not in timeline_prompt:
                raise NodeValidationError(
                    f"episodes[{index}].timeline_prompt must include the `{required_marker}` segment marker."
                )
        if episode_number > 1 and "将@视频1延长15s" not in timeline_prompt:
            raise NodeValidationError(
                f"episodes[{index}].timeline_prompt must include `将@视频1延长15s` for chained episodes."
            )
        normalized_episodes.append(
            {
                "episode_number": episode_number,
                "title": _require_str(episode.get("title"), f"episodes[{index}].title"),
                "upload_slots": normalized_slots,
                "timeline_prompt": timeline_prompt,
                "sound_design": _require_str(episode.get("sound_design"), f"episodes[{index}].sound_design"),
                "end_frame": _require_str(episode.get("end_frame"), f"episodes[{index}].end_frame"),
                "continuity_to_next": _require_str(
                    episode.get("continuity_to_next"), f"episodes[{index}].continuity_to_next"
                ),
            }
        )

    payload = {
        "direction_id": direction_id,
        "series_title": script["series_title"],
        "series_style": _require_str(response.get("series_style"), "series_style"),
        "global_rules": global_rules,
        "episodes": normalized_episodes,
    }

    artifact = StageArtifact(
        stage="storyboard_generator",
        generated_at=utc_now(),
        status="completed",
        summary="Storyboard plan completed.",
        required_inputs=["Completed script artifact and asset registry."],
        expected_outputs=["Episode storyboard prompts ready for generation."],
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
        "",
        "## 全局规则",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["global_rules"])
    lines.extend(["", "## 分集分镜", ""])
    for episode in payload["episodes"]:
        lines.extend(
            [
                f"### E{episode['episode_number']:02d} {episode['title']}",
                "",
                "| 上传位置 | 类型 | 素材ID | 参考ID | 用途 |",
                "|---|---|---|---|---|",
            ]
        )
        for slot in episode["upload_slots"]:
            reference_ids = ", ".join(slot["reference_ids"]) if slot["reference_ids"] else "-"
            lines.append(
                f"| {slot['slot']} | {slot['material_type']} | {slot['asset_id']} | {reference_ids} | {slot['usage']} |"
            )
        lines.extend(
            [
                "",
                "```text",
                episode["timeline_prompt"],
                "```",
                "",
                f"- 声音设计：{episode['sound_design']}",
                f"- 尾帧：{episode['end_frame']}",
                f"- 承接：{episode['continuity_to_next']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_storyboard_episode_markdown(series_title: str, episode: dict[str, Any]) -> str:
    lines = [
        f"# {series_title} - E{episode['episode_number']:02d} {episode['title']}",
        "",
        "## 素材上传清单",
        "",
        "| 上传位置 | 类型 | 素材ID | 参考ID | 用途 |",
        "|---|---|---|---|---|",
    ]
    for slot in episode["upload_slots"]:
        reference_ids = ", ".join(slot["reference_ids"]) if slot["reference_ids"] else "-"
        lines.append(
            f"| {slot['slot']} | {slot['material_type']} | {slot['asset_id']} | {reference_ids} | {slot['usage']} |"
        )
    lines.extend(
        [
            "",
            "## Timeline Prompt",
            "",
            "```text",
            episode["timeline_prompt"],
            "```",
            "",
            "## 声音设计",
            "",
            episode["sound_design"],
            "",
            "## 尾帧描述",
            "",
            episode["end_frame"],
            "",
            "## 承接说明",
            "",
            episode["continuity_to_next"],
            "",
        ]
    )
    return "\n".join(lines)


def prepare_qa_review(
    config: dict[str, Any],
    analysis: dict[str, Any],
    directions: dict[str, Any],
    script: dict[str, Any],
    assets: dict[str, Any],
    storyboards: dict[str, Any],
    reference_library: dict[str, Any],
) -> PreparedRequest:
    referenced_asset_count = sum(
        1 for item in assets["characters"] + assets["scenes"] + assets["props"] if item["reference_ids"]
    )
    prompt = f"""# QA Review Request

你是“AI 视频生产 QA 审查”专家。现在你需要审查整条链路是否可执行：视频解析、方向规划、裂变剧本、素材库、分镜是否一致。

## Project Context

- Project: `{config["project_name"]}`
- Source video: `{config["source_video"]}`

## Audit anchors

- Working title: {analysis["source_overview"]["working_title"]}
- Selected direction: {directions["selected_direction_id"]}
- Series title: {script["series_title"]}
- Asset count: {len(assets["characters"]) + len(assets["scenes"]) + len(assets["props"])}
- Reference assets registered: {len(reference_library["references"])}
- Assets depending on references: {referenced_asset_count}
- Storyboard episodes: {len(storyboards["episodes"])}

## Review goals

1. 检查 must_keep 事实有没有被破坏。
2. 检查剧本、素材、分镜之间的素材引用和叙事连贯性。
3. 检查所有依赖参考图的素材，是否在分镜上传表中明确给出了 reference_ids。
4. 标出高/中/低风险问题。
5. 给出是否可进入实际生成阶段的判断。

## Output requirements

- 严格输出 JSON。
- `overall_status` 只能是 `pass`、`needs_revision`、`fail`。
- `checklist` 至少覆盖事实一致性、素材完整性、分镜连续性、生成可执行性。
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
        next_stage=None,
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
