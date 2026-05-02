"""Schema definitions for workflow state and per-stage artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from workflow.constants import STAGES


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StageArtifactSpec:
    stage: str
    label: str
    description: str
    input_summary: str
    output_summary: str
    output_dir: str
    json_filename: str
    markdown_filename: str


@dataclass
class ProjectConfig:
    project_name: str
    source_video: str
    project_root: str
    platform: str = "seedance"
    aspect_ratio: str = "9:16"
    episode_duration_seconds: int = 15
    target_episode_count: int = 6
    locale: str = "zh-CN"
    notes: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageState:
    stage: str
    status: str = "pending"
    last_run_at: str | None = None
    json_output: str | None = None
    markdown_output: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowState:
    current_stage: str = "video_analyzer"
    updated_at: str = field(default_factory=utc_now)
    stages: list[StageState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "updated_at": self.updated_at,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass
class StageArtifact:
    stage: str
    generated_at: str
    status: str
    summary: str
    required_inputs: list[str]
    expected_outputs: list[str]
    next_stage: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_stage_states() -> list[StageState]:
    return [StageState(stage=stage.name) for stage in STAGES]


def stage_specs() -> list[StageArtifactSpec]:
    return [
        StageArtifactSpec(
            stage=stage.name,
            label=stage.label,
            description=stage.description,
            input_summary=stage.input_summary,
            output_summary=stage.output_summary,
            output_dir=stage.output_dir,
            json_filename=stage.json_filename,
            markdown_filename=stage.markdown_filename,
        )
        for stage in STAGES
    ]
