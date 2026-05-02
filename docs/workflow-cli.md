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
  outputs/
    01_video_analysis/
    02_direction_planning/
    03_scripts/
    04_asset_library/
    05_storyboards/
    06_qa/
```

## Commands

Change into the subproject and install in editable mode:

```bash
cd video-fission-workflow
python3 -m pip install -e .
```

Create a project:

```bash
video-fission-workflow init "司马光裂变" --source-video "./input.mp4"
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

## What is implemented now

- Stage definitions and output locations
- `config.json` / `state.json` project state
- Stage ordering and dependency checks
- Real request/response flow for all six stages
- Direct OpenAI-compatible API execution for all six stages
- Per-episode storyboard markdown export under `05_storyboards/`

## What comes next

- Optionally add a minimal web UI on top of the same project files
