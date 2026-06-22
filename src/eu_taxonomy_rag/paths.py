"""Project-root paths shared across the application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FAQ_PATH = PROJECT_ROOT / "data" / "taxonomy_faqs_cleaned.md"
DEFAULT_CHUNKS_CACHE = PROJECT_ROOT / ".cache" / "chunks.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / ".cache" / "index"
DEFAULT_EVAL_DB = PROJECT_ROOT / ".cache" / "generation_eval.db"
