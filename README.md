# Video Fission Workflow

A standalone workflow subproject for:

`video analysis -> direction planning -> fission script -> asset planning -> storyboard generation -> QA`

This project is independent from the Seedance-specific example content in the parent repository. It is meant to be copied into its own GitHub repository when you are ready.

## What it includes

- A lightweight Python CLI
- OpenAI-compatible API calling support
- Request/import mode for providers that need manual handoff
- Structured `json + markdown` outputs for all six stages
- Episode-level storyboard markdown export

## Quick start

```bash
cd video-fission-workflow
python3 -m pip install -e .
video-fission-workflow --help
```

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

## Suggested next files

- `LICENSE`
- GitHub Actions workflow for `py_compile` or smoke tests
- `.github/ISSUE_TEMPLATE/` if you want to track product changes in GitHub
