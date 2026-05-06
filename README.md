# Video Fission Workflow

A standalone workflow subproject for:

`video analysis -> direction planning -> fission script -> asset planning -> storyboard generation -> QA`

This project is independent from the Seedance-specific example content in the parent repository. It is meant to be copied into its own GitHub repository when you are ready.

## What it includes

- A lightweight Python CLI
- OpenAI-compatible API calling support
- Request/import mode for providers that need manual handoff
- Structured `json + markdown` outputs across planning and execution stages
- Episode-level storyboard markdown export
- Local reference-asset manifest for meme characters, fixed props, and other must-match visuals
- Source video intake manifest and reusable material library manifest
- Execution-layer workflow stages for render planning and video-result importing

## Quick start

```bash
cd video-fission-workflow
python3 -m pip install -e .
video-fission-workflow --help
```

Qwen Omni Flash can be used directly for `analyze` through DashScope's OpenAI-compatible API:

```bash
cp .env.example .env
# edit .env and fill ANALYZE_API_KEY / PLANNING_API_KEY
video-fission-workflow analyze "司马光裂变"
```

The CLI auto-loads `.env` from the repo root and supports stage-specific settings like `ANALYZE_*`, `PLANNING_*`, `SCRIPT_*`, `ASSET_*`, `STORYBOARD_*`, and `QA_*`.
It also supports `RENDER_*` for direct async video generation, including a yunwu.ai + Seedance 1.5 starting configuration.

Create a project:

```bash
video-fission-workflow init "司马光裂变" --source-video "./input.mp4"
```

Detailed usage:

- [docs/workflow-cli.md](./docs/workflow-cli.md)

## GitHub-ready notes

- Keep this folder as the root when creating the new repository
- Do not commit real `.env` files or source videos
- Generated project outputs under `projects/` are ignored by default

## Reference asset flow

For assets that cannot rely on text prompts alone, such as meme characters or fixed-expression IP:

- Put the files under `projects/<slug>/source/assets/`
- Register them in `projects/<slug>/source/reference_assets.json`
- The asset planner will bind production assets to `reference_id`s
- Storyboard upload tables will then surface `reference_ids` so the operator knows what must be uploaded alongside generated assets

## Material library flow

For reusable local or remote materials such as pre-made background plates, prop photos, or stock clips:

- Register them in `projects/<slug>/source/material_library.json`
- Bind them to planned assets through `material_ids`
- Storyboard upload tables and execution plans will surface those `material_ids`
- `render-videos` will prepare per-episode execution request files and expect a render-result JSON import

## Suggested next files

- `LICENSE`
- GitHub Actions workflow for `py_compile` or smoke tests
- `.github/ISSUE_TEMPLATE/` if you want to track product changes in GitHub
