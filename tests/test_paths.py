from pathlib import Path

from eu_taxonomy_rag.paths import (
    DEFAULT_CHUNKS_CACHE,
    DEFAULT_EVAL_DB,
    DEFAULT_FAQ_PATH,
    DEFAULT_INDEX_DIR,
    PROJECT_ROOT,
)


def test_project_paths_point_to_repo_root() -> None:
    assert PROJECT_ROOT.exists()
    assert DEFAULT_FAQ_PATH == PROJECT_ROOT / "data" / "taxonomy_faqs_cleaned.md"
    assert DEFAULT_CHUNKS_CACHE == PROJECT_ROOT / ".cache" / "chunks.jsonl"
    assert DEFAULT_INDEX_DIR == PROJECT_ROOT / ".cache" / "index"
    assert DEFAULT_EVAL_DB == PROJECT_ROOT / ".cache" / "generation_eval.db"


def test_project_root_respects_env_override(tmp_path: Path, monkeypatch) -> None:
    faq_dir = tmp_path / "data"
    faq_dir.mkdir()
    (faq_dir / "taxonomy_faqs_cleaned.md").write_text("# FAQ\n", encoding="utf-8")

    monkeypatch.setenv("EU_TAXONOMY_PROJECT_ROOT", str(tmp_path))
    import importlib

    import eu_taxonomy_rag.paths as paths_module

    importlib.reload(paths_module)
    assert paths_module.PROJECT_ROOT == tmp_path.resolve()
    assert paths_module.DEFAULT_FAQ_PATH == tmp_path / "data" / "taxonomy_faqs_cleaned.md"

    monkeypatch.delenv("EU_TAXONOMY_PROJECT_ROOT", raising=False)
    importlib.reload(paths_module)
