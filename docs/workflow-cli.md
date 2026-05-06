# Workflow CLI Scaffold

This subproject includes a lightweight workflow scaffold for the customized pipeline:

`video -> direction planning -> script -> assets -> storyboard -> QA`

## Why this shape

- Zero runtime dependencies
- File-based project state
- One command per stage
- Easy to replace placeholder outputs with real model calls later

## Project layout

After `init`, each workflow project is created under `projects/<slug>/`:

```text
projects/<slug>/
  README.md
  config.json
  state.json
  schemas.json
  source/
    assets/
    source_video.json
    reference_assets.json
    material_library.json
  outputs/
    01_video_analysis/
    02_direction_planning/
    03_scripts/
    04_asset_library/
    05_storyboards/
    06_qa/
    07_execution_plan/
    08_video_generation/
```

## Commands

Change into the subproject and install in editable mode:

```bash
cd video-fission-workflow
python3 -m pip install -e .
```

The CLI now auto-loads `.env` from the repo root, so you do not need `set -a; source .env` for normal usage.

Create a project:

```bash
video-fission-workflow init "司马光裂变" --source-video "./input.mp4"
```

For reference-driven assets such as meme characters, iconic props, or fixed IP faces:

1. Put the files under `projects/<slug>/source/assets/`
2. Fill `projects/<slug>/source/reference_assets.json`
3. Run `gen-assets` and `gen-storyboards` as usual

Suggested `reference_assets.json` shape:

```json
{
  "references": [
    {
      "reference_id": "R01",
      "name": "Sad Frog Meme",
      "category": "character",
      "source_type": "image",
      "source": "source/assets/sad-frog.png",
      "usage_notes": "Use as face and expression reference. Do not redesign into another character.",
      "must_keep": ["green frog face", "downturned eyes", "flat meme composition"]
    }
  ]
}
```

Suggested `source_video.json` shape:

```json
{
  "source": "/absolute/or/relative/path/to/source.mp4",
  "title": "Original Campaign Video",
  "analysis_notes": "Focus on the transformation beat and the final reveal.",
  "transcript_excerpt": "",
  "key_moments": ["open on the close-up", "cut to crowd reaction"],
  "frame_reference_images": ["source/assets/frame-01.png", "source/assets/frame-02.png"]
}
```

For `qwen3-omni-flash`, prefer a public `https://...mp4` URL in `source`.
If the source video is a large local file, keep `frame_reference_images` populated so `analyze` still has visual context.

Suggested `material_library.json` shape:

```json
{
  "materials": [
    {
      "material_id": "M01",
      "name": "Temple Courtyard Plate",
      "category": "scene",
      "media_type": "image",
      "source": "source/assets/temple-courtyard.png",
      "usage_notes": "Use as the recurring wide shot background.",
      "tags": ["courtyard", "daytime", "ancient"],
      "available_for": ["gen-assets", "gen-storyboards"],
      "linked_reference_ids": []
    }
  ]
}
```

Check status:

```bash
video-fission-workflow status "司马光裂变"
```

Every stage now supports two modes:

- `prepare/import` mode: write `request.md` + `response_template.json`, then import a model JSON file
- `direct API` mode: call an OpenAI-compatible `/v1/chat/completions` endpoint directly

## Prepare/import mode

```bash
video-fission-workflow analyze "司马光裂变"
```

This writes:

- `request.md`
- `response_template.json`

Feed the source video plus `request.md` to your chosen model, save its JSON reply locally, then import it:

```bash
video-fission-workflow analyze "司马光裂变" --response-file "./responses/analyze.json"

video-fission-workflow plan-directions "司马光裂变"
video-fission-workflow plan-directions "司马光裂变" --response-file "./responses/directions.json"

video-fission-workflow gen-script "司马光裂变"
video-fission-workflow gen-script "司马光裂变" --response-file "./responses/script.json"

video-fission-workflow gen-assets "司马光裂变"
video-fission-workflow gen-assets "司马光裂变" --response-file "./responses/assets.json"

video-fission-workflow gen-storyboards "司马光裂变"
video-fission-workflow gen-storyboards "司马光裂变" --response-file "./responses/storyboards.json"

video-fission-workflow qa "司马光裂变"
video-fission-workflow qa "司马光裂变" --response-file "./responses/qa.json"

video-fission-workflow prepare-execution "司马光裂变"

video-fission-workflow render-videos "司马光裂变"
video-fission-workflow render-videos "司马光裂变" --response-file "./responses/render.json"
```

## Direct API mode

The workflow uses an OpenAI-compatible Chat Completions shape:

- `POST /v1/chat/completions`
- `messages`
- optional `response_format: {"type":"json_object"}`

Set env vars:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4.1-mini"
```

Then run stages directly:

```bash
video-fission-workflow analyze "司马光裂变" --model "$OPENAI_MODEL"
video-fission-workflow plan-directions "司马光裂变" --model "$OPENAI_MODEL"
video-fission-workflow gen-script "司马光裂变" --model "$OPENAI_MODEL"
video-fission-workflow gen-assets "司马光裂变" --model "$OPENAI_MODEL"
video-fission-workflow gen-storyboards "司马光裂变" --model "$OPENAI_MODEL"
video-fission-workflow qa "司马光裂变" --model "$OPENAI_MODEL"
```

If your compatible provider does not support JSON mode:

```bash
video-fission-workflow analyze "司马光裂变" --model "$OPENAI_MODEL" --disable-json-mode
```

## Qwen Omni Flash

The workflow can call DashScope's OpenAI-compatible endpoint directly for `analyze`.

Recommended env vars:

```bash
export OPENAI_MODEL="qwen3-omni-flash"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="your-dashscope-key"
```

Then run:

```bash
video-fission-workflow analyze "司马光裂变" --model "$OPENAI_MODEL"
```

Notes:

- `qwen3-omni-flash` is handled in streaming mode automatically because Qwen Omni requires streaming.
- For a remote source video URL, the workflow sends the video directly as `video_url`.
- For a local source video, the workflow only inlines the video when it is small enough for a safe Base64 payload; otherwise it falls back to the source manifest plus any `frame_reference_images`.
- `gen-assets`, `gen-storyboards`, and `qa` can still use the same OpenAI-compatible path for text-plus-image stages.

## Stage-specific .env setup

You can now configure different providers per stage directly in `.env`.

Typical setup:

```bash
ANALYZE_MODEL=qwen3-omni-flash
ANALYZE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
ANALYZE_API_KEY=your-dashscope-key
ANALYZE_AUTH_SCHEME=bearer

PLANNING_MODEL=deepseek-v4-pro
PLANNING_API_BASE=https://api.deepseek.com
PLANNING_API_KEY=your-deepseek-key
PLANNING_AUTH_SCHEME=bearer
```

If you use a yunwu.ai OpenAI-compatible gateway for planning:

```bash
PLANNING_MODEL=deepseek-v4-pro
PLANNING_API_BASE=https://yunwu.ai/v1
PLANNING_API_KEY=your-yunwu-token
PLANNING_AUTH_SCHEME=raw
```

Priority order:

- CLI flags like `--model`, `--api-base`, `--api-key`
- Per-stage env vars like `ANALYZE_*`, `PLAN_*`, `SCRIPT_*`, `ASSET_*`, `STORYBOARD_*`, `QA_*`
- Shared planning env vars like `PLANNING_*`
- Global fallback env vars like `OPENAI_*`

## Seedance 1.5 Render Setup

`render-videos` can now call an async video provider directly.

Recommended `.env` setup for yunwu.ai + Seedance 1.5:

```bash
RENDER_PROVIDER=yunwu-seedance-1.5
RENDER_MODEL=doubao-seedance-1-5-pro-251215
RENDER_API_BASE=https://yunwu.ai
RENDER_API_KEY=your-yunwu-token
RENDER_AUTH_SCHEME=raw
RENDER_CREATE_PATH=/v1/videos
RENDER_STATUS_PATH=/v1/videos/{task_id}
RENDER_REQUEST_FORMAT=content
RENDER_DURATION=8
RENDER_GENERATE_AUDIO=true
RENDER_AUTO_POLL=true
RENDER_POLL_SECONDS=20
RENDER_MAX_POLLS=90
```

How it behaves:

- First `render-videos` call prepares request markdown files and submits ready episodes to the provider.
- If polling finishes, the stage becomes `completed`.
- If some async jobs are still running, the stage stays `in_progress`.
- Re-run `render-videos` later and it will keep polling the saved task IDs instead of importing a manual manifest.

Run it:

```bash
python3 -m workflow.cli render-videos "司马光裂变"
```

If your gateway uses different async endpoints, adjust:

```bash
RENDER_CREATE_PATH=/your/create/path
RENDER_STATUS_PATH=/your/status/path/{task_id}
```

## What is implemented now

- Stage definitions and output locations
- `config.json` / `state.json` project state
- Stage ordering and dependency checks
- Real request/response flow for all planning stages plus render-result importing
- Direct OpenAI-compatible API execution for the planning stages
- Per-episode storyboard markdown export under `05_storyboards/`
- Reference-aware asset planning and storyboard upload tables
- Source video manifest and material-library ingestion
- Execution plan generation and render-result importing

## What comes next

- Optionally add a minimal web UI on top of the same project files
