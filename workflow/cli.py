"""CLI entrypoint for the lightweight workflow scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from workflow.constants import STAGES, STAGE_BY_COMMAND
from workflow.env_loader import load_dotenv
from workflow.project import (
    WorkflowError,
    init_project,
    load_state,
    project_path,
    run_stage,
    status_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-fission-workflow",
        description="Lightweight workflow scaffold for video analysis, content fission, script, asset, storyboard, and QA pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new workflow project scaffold.")
    init_parser.add_argument("project_name", help="Human-readable project name.")
    init_parser.add_argument("--source-video", required=True, help="Path or URL to the source video.")
    init_parser.add_argument("--notes", default="", help="Optional notes for project setup.")
    init_parser.set_defaults(handler=handle_init)

    status_parser = subparsers.add_parser("status", help="Show stage completion status for a project.")
    status_parser.add_argument("project_name", help="Project name or slug.")
    status_parser.set_defaults(handler=handle_status)

    run_all_parser = subparsers.add_parser("run-all", help="Run the full workflow chain, skipping already-completed stages by default.")
    run_all_parser.add_argument("project_name", help="Project name or slug.")
    run_all_parser.add_argument(
        "--direction-id",
        help="Optional direction ID to use when running gen-script. Defaults to the recommended direction from plan-directions.",
    )
    run_all_parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Re-run completed stages instead of skipping them.",
    )
    run_all_parser.set_defaults(handler=handle_run_all)

    for stage in STAGES:
        stage_parser = subparsers.add_parser(stage.command, help=stage.description)
        stage_parser.add_argument("project_name", help="Project name or slug.")
        if stage.command in {"analyze", "plan-directions", "gen-script"}:
            stage_parser.add_argument(
                "--model",
                help="OpenAI-compatible model name. If provided, the workflow calls the API directly instead of waiting for --response-file.",
            )
            stage_parser.add_argument(
                "--api-base",
                help="OpenAI-compatible API base URL, for example https://api.openai.com/v1 . Falls back to OPENAI_BASE_URL.",
            )
            stage_parser.add_argument(
                "--api-key",
                help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.",
            )
            stage_parser.add_argument(
                "--temperature",
                type=float,
                default=0.2,
                help="Sampling temperature for direct API calls.",
            )
            stage_parser.add_argument(
                "--timeout-seconds",
                type=int,
                default=180,
                help="HTTP timeout in seconds for direct API calls.",
            )
            stage_parser.add_argument(
                "--disable-json-mode",
                action="store_true",
                help="Disable response_format json_object for providers that do not support JSON mode.",
            )
            stage_parser.add_argument(
                "--response-file",
                help="Path to a model-generated JSON response for this stage. Without it, the command prepares request files only.",
            )
        if stage.command in {"gen-assets", "gen-storyboards", "qa"}:
            stage_parser.add_argument(
                "--model",
                help="OpenAI-compatible model name. If provided, the workflow calls the API directly instead of waiting for --response-file.",
            )
            stage_parser.add_argument(
                "--api-base",
                help="OpenAI-compatible API base URL, for example https://api.openai.com/v1 . Falls back to OPENAI_BASE_URL.",
            )
            stage_parser.add_argument(
                "--api-key",
                help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.",
            )
            stage_parser.add_argument(
                "--temperature",
                type=float,
                default=0.2,
                help="Sampling temperature for direct API calls.",
            )
            stage_parser.add_argument(
                "--timeout-seconds",
                type=int,
                default=180,
                help="HTTP timeout in seconds for direct API calls.",
            )
            stage_parser.add_argument(
                "--disable-json-mode",
                action="store_true",
                help="Disable response_format json_object for providers that do not support JSON mode.",
            )
            stage_parser.add_argument(
                "--response-file",
                help="Path to a model-generated JSON response for this stage. Without it, the command prepares request files only.",
            )
        if stage.command == "render-videos":
            stage_parser.add_argument(
                "--model",
                help="Optional render model override. Falls back to RENDER_MODEL or stage-specific .env settings.",
            )
            stage_parser.add_argument(
                "--api-base",
                help="Optional render API base override. Falls back to RENDER_API_BASE or stage-specific .env settings.",
            )
            stage_parser.add_argument(
                "--api-key",
                help="Optional render API key override. Falls back to RENDER_API_KEY or stage-specific .env settings.",
            )
            stage_parser.add_argument(
                "--timeout-seconds",
                type=int,
                default=180,
                help="HTTP timeout in seconds for direct render provider calls.",
            )
            stage_parser.add_argument(
                "--response-file",
                help="Path to a render-result JSON manifest. Without it, the command prepares render request files only.",
            )
        if stage.command == "gen-script":
            stage_parser.add_argument(
                "--direction-id",
                help="Optional direction ID to script against. Defaults to the recommended direction from plan-directions.",
            )
        stage_parser.set_defaults(handler=make_stage_handler(stage.command))

    return parser


def handle_init(args: argparse.Namespace) -> int:
    root = init_project(args.project_name, args.source_video, notes=args.notes)
    print(f"Created workflow project at {root}")
    return 0


def handle_status(args: argparse.Namespace) -> int:
    root = project_path(args.project_name)
    print(status_table(root))
    return 0


def handle_run_all(args: argparse.Namespace) -> int:
    root = project_path(args.project_name)
    rerun_completed = bool(getattr(args, "rerun_completed", False))
    direction_id = getattr(args, "direction_id", None)

    for stage in STAGES:
        state = load_state(root)
        stage_state = next(item for item in state["stages"] if item["stage"] == stage.name)
        status = stage_state.get("status", "")

        if status == "completed" and not rerun_completed:
            print(f"Skipping {stage.command} (already completed)")
            continue

        stage_direction_id = direction_id if stage.command == "gen-script" else None
        first_path, second_path = run_stage(
            root,
            stage.command,
            direction_id=stage_direction_id,
        )
        print(f"Wrote {first_path}")
        print(f"Wrote {second_path}")

        state = load_state(root)
        stage_state = next(item for item in state["stages"] if item["stage"] == stage.name)
        if stage_state.get("status") == "in_progress":
            print(f"Stopped at {stage.command}: {stage_state.get('note', 'Stage is still in progress.')}")
            return 0

    return 0


def make_stage_handler(command: str) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        root = project_path(args.project_name)
        response_file = Path(args.response_file) if getattr(args, "response_file", None) else None
        direction_id = getattr(args, "direction_id", None)
        model = getattr(args, "model", None)
        api_base = getattr(args, "api_base", None)
        api_key = getattr(args, "api_key", None)
        temperature = getattr(args, "temperature", 0.2)
        timeout_seconds = getattr(args, "timeout_seconds", 180)
        use_json_mode = not getattr(args, "disable_json_mode", False)
        first_path, second_path = run_stage(
            root,
            command,
            response_file=response_file,
            direction_id=direction_id,
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            use_json_mode=use_json_mode,
        )
        if response_file is None and command in {
            "analyze",
            "plan-directions",
            "gen-script",
            "gen-assets",
            "gen-storyboards",
            "qa",
            "render-videos",
        }:
            state = load_state(root)
            stage_name = STAGE_BY_COMMAND[command].name
            stage_state = next(item for item in state["stages"] if item["stage"] == stage_name)
            if stage_state["status"] == "in_progress":
                print(f"Wrote {first_path}")
                print(f"Wrote {second_path}")
                if command == "render-videos":
                    print("Render request prepared. Import a render-result JSON with --response-file to complete this stage.")
                else:
                    print("Model request prepared. Import a JSON response with --response-file to complete this stage.")
                return 0
        print(f"Wrote {first_path}")
        print(f"Wrote {second_path}")
        return 0

    return handler


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler")
    try:
        return handler(args)
    except WorkflowError as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
