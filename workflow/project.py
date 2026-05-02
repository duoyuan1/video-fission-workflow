"""Project and filesystem helpers for the workflow scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from workflow.constants import STAGES, STAGE_BY_COMMAND
from workflow.nodes import (
    NodeValidationError,
    finalize_response,
    prepare_request,
    render_storyboard_episode_markdown,
)
from workflow.provider import (
    ProviderError,
    resolve_provider_settings,
    call_openai_compatible_json,
    write_provider_response,
)
from workflow.schemas import (
    ProjectConfig,
    StageArtifact,
    WorkflowState,
    default_stage_states,
    stage_specs,
    utc_now,
)


PROJECTS_DIR = Path("projects")
CONFIG_FILE = "config.json"
STATE_FILE = "state.json"
SCHEMA_FILE = "schemas.json"
REQUEST_FILE = "request.md"
RESPONSE_TEMPLATE_FILE = "response_template.json"
RAW_RESPONSE_FILE = "raw_response.json"
PROVIDER_RESPONSE_FILE = "provider_response.json"


class WorkflowError(RuntimeError):
    """Raised when the workflow command cannot continue."""


def slugify(name: str) -> str:
    result = []
    for char in name.strip().lower():
        if char.isalnum():
            result.append(char)
        elif char in {" ", "-", "_"}:
            result.append("-")
    slug = "".join(result).strip("-")
    return slug or "project"


def project_path(project_name: str) -> Path:
    return PROJECTS_DIR / slugify(project_name)


def ensure_project(path: Path) -> None:
    if not path.exists():
        raise WorkflowError(f"Project not found: {path}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def init_project(project_name: str, source_video: str, notes: str = "") -> Path:
    root = project_path(project_name)
    if root.exists():
        raise WorkflowError(f"Project already exists: {root}")

    (root / "source").mkdir(parents=True)
    (root / "outputs").mkdir(parents=True)

    config = ProjectConfig(
        project_name=project_name,
        source_video=source_video,
        project_root=str(root),
        notes=notes,
    )
    state = WorkflowState(stages=default_stage_states())

    write_json(root / CONFIG_FILE, config.to_dict())
    write_json(root / STATE_FILE, state.to_dict())
    write_json(root / SCHEMA_FILE, {"stages": [spec.__dict__ for spec in stage_specs()]})

    for stage in STAGES:
        (root / "outputs" / stage.output_dir).mkdir(parents=True, exist_ok=True)

    readme = build_project_readme(config.project_name, source_video)
    (root / "README.md").write_text(readme, encoding="utf-8")

    return root


def build_project_readme(project_name: str, source_video: str) -> str:
    return (
        f"# {project_name}\n\n"
        "This workflow project was created by `video-fission-workflow init`.\n\n"
        "## Source\n\n"
        f"- Source video: `{source_video}`\n\n"
        "## Stages\n\n"
        + "\n".join(f"- `{stage.command}` -> {stage.label}" for stage in STAGES)
        + "\n"
    )


def load_state(root: Path) -> dict:
    ensure_project(root)
    return read_json(root / STATE_FILE)


def load_config(root: Path) -> dict:
    ensure_project(root)
    return read_json(root / CONFIG_FILE)


def save_state(root: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    write_json(root / STATE_FILE, state)


def stage_output_paths(root: Path, command: str) -> tuple[Path, Path]:
    stage = STAGE_BY_COMMAND[command]
    output_root = root / "outputs" / stage.output_dir
    return output_root / stage.json_filename, output_root / stage.markdown_filename


def previous_stage_completed(state: dict, command: str) -> bool:
    stage_names = [stage.name for stage in STAGES]
    stage = STAGE_BY_COMMAND[command]
    index = stage_names.index(stage.name)
    if index == 0:
        return True
    previous = state["stages"][index - 1]
    return previous["status"] == "completed"


def load_response_file(path: Path) -> dict:
    if not path.exists():
        raise WorkflowError(f"Response file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Response file is not valid JSON: {path}") from exc


def set_stage_state(
    root: Path,
    stage_name: str,
    *,
    status: str,
    note: str,
    json_output: str | None = None,
    markdown_output: str | None = None,
) -> None:
    state = load_state(root)
    for item in state["stages"]:
        if item["stage"] == stage_name:
            item["status"] = status
            item["last_run_at"] = utc_now()
            if json_output is not None:
                item["json_output"] = json_output
            if markdown_output is not None:
                item["markdown_output"] = markdown_output
            item["note"] = note
            break
    save_state(root, state)


def run_stage(
    root: Path,
    command: str,
    *,
    response_file: Path | None = None,
    direction_id: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    timeout_seconds: int = 180,
    use_json_mode: bool = True,
) -> tuple[Path, Path]:
    ensure_project(root)
    state = load_state(root)
    config = load_config(root)

    if not previous_stage_completed(state, command):
        raise WorkflowError(f"Previous stage must be completed before `{command}`.")

    stage = STAGE_BY_COMMAND[command]
    stage_names = [item.name for item in STAGES]
    current_index = stage_names.index(stage.name)
    json_path, markdown_path = stage_output_paths(root, command)
    next_stage = stage_names[current_index + 1] if current_index + 1 < len(stage_names) else None

    if command in {"analyze", "plan-directions", "gen-script", "gen-assets", "gen-storyboards", "qa"}:
        try:
            if response_file is not None and model is not None:
                raise WorkflowError("Use either --response-file or --model, not both.")
            if response_file is None:
                request_path, template_path = prepare_request(root, command, config, direction_id=direction_id)
                settings = resolve_provider_settings(
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    use_json_mode=use_json_mode,
                )
                if settings is None:
                    stage_state = state["stages"][current_index]
                    stage_state["status"] = "in_progress"
                    stage_state["last_run_at"] = utc_now()
                    stage_state["note"] = "Request package prepared. Import a model JSON response to complete this stage."
                    save_state(root, state)
                    return (request_path, template_path)

                prompt = request_path.read_text(encoding="utf-8")
                response_template = read_json(template_path)
                provider_response, response = call_openai_compatible_json(
                    prompt=prompt,
                    response_template=response_template,
                    settings=settings,
                )
                provider_response_path = root / "outputs" / stage.output_dir / PROVIDER_RESPONSE_FILE
                write_provider_response(provider_response_path, provider_response)
            else:
                response = load_response_file(response_file)
            artifact, markdown = finalize_response(root, command, config, response, direction_id=direction_id)
        except NodeValidationError as exc:
            raise WorkflowError(str(exc)) from exc
        except ProviderError as exc:
            raise WorkflowError(str(exc)) from exc

        write_json(json_path, artifact.to_dict())
        markdown_path.write_text(markdown, encoding="utf-8")
        raw_response_path = root / "outputs" / stage.output_dir / RAW_RESPONSE_FILE
        write_json(raw_response_path, response)
        if command == "gen-storyboards":
            write_storyboard_episode_files(root, artifact.payload)

        stage_state = state["stages"][current_index]
        stage_state["status"] = "completed"
        stage_state["last_run_at"] = artifact.generated_at
        stage_state["json_output"] = str(json_path)
        stage_state["markdown_output"] = str(markdown_path)
        stage_state["note"] = "Stage completed from imported model response."
        state["current_stage"] = next_stage or stage.name
        save_state(root, state)
        return json_path, markdown_path

    artifact = StageArtifact(
        stage=stage.name,
        generated_at=utc_now(),
        status="placeholder",
        summary=stage.description,
        required_inputs=[stage.input_summary],
        expected_outputs=[stage.output_summary],
        next_stage=next_stage,
        payload={
            "project_name": config["project_name"],
            "source_video": config["source_video"],
            "notes": "Replace this placeholder payload with real model output in the next implementation phase.",
        },
    )
    write_json(json_path, artifact.to_dict())
    markdown_path.write_text(build_stage_markdown(stage.label, artifact, next_stage), encoding="utf-8")

    stage_state = state["stages"][current_index]
    stage_state["status"] = "completed"
    stage_state["last_run_at"] = artifact.generated_at
    stage_state["json_output"] = str(json_path)
    stage_state["markdown_output"] = str(markdown_path)
    stage_state["note"] = "Placeholder artifact generated by CLI scaffold."
    state["current_stage"] = next_stage or stage.name
    save_state(root, state)

    return json_path, markdown_path


def write_storyboard_episode_files(root: Path, payload: dict) -> None:
    stage = STAGE_BY_COMMAND["gen-storyboards"]
    output_root = root / "outputs" / stage.output_dir
    for episode in payload.get("episodes", []):
        filename = f"E{episode['episode_number']:02d}_分镜.md"
        episode_markdown = render_storyboard_episode_markdown(payload["series_title"], episode)
        (output_root / filename).write_text(episode_markdown, encoding="utf-8")


def build_stage_markdown(stage_label: str, artifact: StageArtifact, next_stage: str | None) -> str:
    lines = [
        f"# {stage_label}",
        "",
        f"- Stage: `{artifact.stage}`",
        f"- Generated at: `{artifact.generated_at}`",
        f"- Status: `{artifact.status}`",
        "",
        "## Summary",
        "",
        artifact.summary,
        "",
        "## Required Inputs",
        "",
    ]
    lines.extend(f"- {item}" for item in artifact.required_inputs)
    lines.extend(
        [
            "",
            "## Expected Outputs",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in artifact.expected_outputs)
    lines.extend(
        [
            "",
            "## Payload",
            "",
            "```json",
            json.dumps(artifact.payload, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Next",
            "",
            f"- `{next_stage}`" if next_stage else "- Workflow complete",
            "",
        ]
    )
    return "\n".join(lines)


def status_table(root: Path) -> str:
    state = load_state(root)
    rows = ["stage\tstatus\tlast_run_at"]
    rows.extend(
        f"{stage['stage']}\t{stage['status']}\t{stage['last_run_at'] or '-'}"
        for stage in state["stages"]
    )
    rows.append(f"current_stage\t{state['current_stage']}\t{state['updated_at']}")
    return "\n".join(rows)
