"""Shared constants for the workflow scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageDefinition:
    name: str
    label: str
    description: str
    input_summary: str
    output_summary: str
    command: str
    output_dir: str
    json_filename: str
    markdown_filename: str


STAGES: list[StageDefinition] = [
    StageDefinition(
        name="video_analyzer",
        label="Video Analyzer",
        description="Read the source video and produce a structured content summary.",
        input_summary="Source video path plus project-level workflow config.",
        output_summary="Structured video summary, roles, scenes, motions, and highlights.",
        command="analyze",
        output_dir="01_video_analysis",
        json_filename="video_summary.json",
        markdown_filename="video_summary.md",
    ),
    StageDefinition(
        name="direction_planner",
        label="Direction Planner",
        description="Turn the video summary into content directions and prioritization.",
        input_summary="Video analysis output and strategy preferences.",
        output_summary="Direction options, target audience, hook, mood curve, and recommendation.",
        command="plan-directions",
        output_dir="02_direction_planning",
        json_filename="directions.json",
        markdown_filename="directions.md",
    ),
    StageDefinition(
        name="script_generator",
        label="Script Generator",
        description="Create one standalone fission-video script from a chosen direction.",
        input_summary="Selected direction plus prior analysis artifacts.",
        output_summary="Single-video hook, script structure, and structured markdown script.",
        command="gen-script",
        output_dir="03_scripts",
        json_filename="script.json",
        markdown_filename="剧本.md",
    ),
    StageDefinition(
        name="asset_planner",
        label="Asset Planner",
        description="Plan characters, scenes, props, and reusable assets for production.",
        input_summary="Approved script and style rules.",
        output_summary="Asset registry with C/S/P IDs and prompt-ready material notes.",
        command="gen-assets",
        output_dir="04_asset_library",
        json_filename="assets.json",
        markdown_filename="素材清单.md",
    ),
    StageDefinition(
        name="storyboard_generator",
        label="Storyboard Generator",
        description="Generate per-episode storyboard prompts for downstream video models.",
        input_summary="Script plus asset registry.",
        output_summary="Episode storyboard prompts, upload slot tables, and end-frame notes.",
        command="gen-storyboards",
        output_dir="05_storyboards",
        json_filename="storyboards.json",
        markdown_filename="分镜总览.md",
    ),
    StageDefinition(
        name="qa_reviewer",
        label="QA Reviewer",
        description="Check consistency across script, assets, storyboards, and continuity.",
        input_summary="All prior artifacts.",
        output_summary="QA report covering missing references, continuity, and readiness.",
        command="qa",
        output_dir="06_qa",
        json_filename="qa_report.json",
        markdown_filename="qa_report.md",
    ),
    StageDefinition(
        name="execution_planner",
        label="Execution Planner",
        description="Resolve storyboards against source video, reference assets, and reusable materials.",
        input_summary="QA report plus all upstream planning artifacts and source libraries.",
        output_summary="Per-episode execution plan with resolved source files and render readiness.",
        command="prepare-execution",
        output_dir="07_execution_plan",
        json_filename="execution_plan.json",
        markdown_filename="执行计划.md",
    ),
    StageDefinition(
        name="video_renderer",
        label="Video Renderer",
        description="Prepare or import final episode video generation results.",
        input_summary="Execution plan plus downstream render outputs.",
        output_summary="Episode video manifest with local or remote output locations.",
        command="render-videos",
        output_dir="08_video_generation",
        json_filename="video_manifest.json",
        markdown_filename="视频生成结果.md",
    ),
]

STAGE_BY_COMMAND = {stage.command: stage for stage in STAGES}
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
