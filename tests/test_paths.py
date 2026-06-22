from pathlib import Path

from eu_taxonomy_rag.paths import DEFAULT_CHUNKS_CACHE, DEFAULT_FAQ_PATH, DEFAULT_INDEX_DIR, PROJECT_ROOT


def test_project_paths_point_to_repo_root() -> None:
    assert PROJECT_ROOT.exists()
    assert DEFAULT_FAQ_PATH == PROJECT_ROOT / "data" / "taxonomy_faqs_cleaned.md"
    assert DEFAULT_CHUNKS_CACHE == PROJECT_ROOT / ".cache" / "chunks.jsonl"
    assert DEFAULT_INDEX_DIR == PROJECT_ROOT / ".cache" / "index"
