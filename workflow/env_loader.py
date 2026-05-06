"""Minimal .env loader for local workflow usage."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> bool:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists() or not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
    return True
