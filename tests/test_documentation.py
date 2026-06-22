from pathlib import Path

import pytest

from eu_taxonomy_rag.paths import DOCS_DOCUMENTATION_DIR, PROJECT_ROOT


def test_docs_documentation_dir_exists() -> None:
    assert DOCS_DOCUMENTATION_DIR.is_dir()


@pytest.mark.parametrize(
    "section_key",
    [
        "home",
        "ingestion",
        "indexing",
        "retrieval",
        "benchmarking",
        "evaluation_datasets",
        "generation",
        "faithfulness",
        "kpi_tracking",
        "tradeoffs",
        "roadmap",
    ],
)
def test_documentation_section_files_exist(section_key: str) -> None:
    path = DOCS_DOCUMENTATION_DIR / f"{section_key}.md"
    assert path.is_file(), f"Missing documentation file: {path}"


def test_load_documentation_section_from_app_helper() -> None:
    import sys

    app_dir = PROJECT_ROOT / "app"
    sys.path.insert(0, str(app_dir))
    from documentation_page import load_documentation_section

    content = load_documentation_section("home")
    assert "### Why this project exists" in content
