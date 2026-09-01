"""Tiny .env loader -- stdlib only, no python-dotenv dependency.

Reads KEY=VALUE lines from the repo root's .env (gitignored, personal) and
puts them into os.environ WITHOUT overriding variables already set in the
shell: the environment always wins over the file.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path=None):
    path = Path(path) if path else REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
