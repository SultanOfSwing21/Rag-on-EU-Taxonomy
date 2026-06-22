"""Project-root paths shared across the application."""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_FAQ_RELATIVE = Path("data") / "taxonomy_faqs_cleaned.md"


def _resolve_project_root() -> Path:
    """Resolve the repository root, including non-editable installs and Docker."""
    env_root = os.environ.get("EU_TAXONOMY_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidates: list[Path] = [
        _PACKAGE_DIR.parents[2],
        _PACKAGE_DIR.parents[1],
        Path.cwd(),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / _FAQ_RELATIVE).exists():
            return resolved

    return _PACKAGE_DIR.parents[2]


def _resolve_eval_db() -> Path:
    env_path = os.environ.get("EU_TAXONOMY_EVAL_DB", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return PROJECT_ROOT / ".cache" / "generation_eval.db"


PROJECT_ROOT = _resolve_project_root()

DEFAULT_FAQ_PATH = PROJECT_ROOT / _FAQ_RELATIVE
DEFAULT_CHUNKS_CACHE = PROJECT_ROOT / ".cache" / "chunks.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / ".cache" / "index"
DEFAULT_EVAL_DB = _resolve_eval_db()
DOCS_DOCUMENTATION_DIR = PROJECT_ROOT / "docs" / "documentation"
