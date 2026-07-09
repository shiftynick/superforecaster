"""Tiny ``.env`` loader shared across the CLI and the LLM client.

We keep secrets (the Metaculus token, the Anthropic API key) in an
``engine/.env`` file of plain ``KEY=VALUE`` lines rather than committing them.
This helper reads such a file into the process environment. It is deliberately
dependency-free: no ``python-dotenv``, just a few lines of parsing.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Read simple ``KEY=VALUE`` lines from ``path`` into ``os.environ``.

    Blank lines and ``#`` comments are ignored. Existing environment variables
    win, so a real shell export always overrides the file. Missing files are a
    no-op, so callers can point at several candidate locations safely.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
