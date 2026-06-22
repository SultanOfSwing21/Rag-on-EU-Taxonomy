"""In-app documentation — guided walkthrough loaded from markdown files."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from eu_taxonomy_rag.paths import DOCS_DOCUMENTATION_DIR

DOC_SECTION_KEY = "documentation_section"

SECTIONS: tuple[tuple[str, str], ...] = (
    ("ingestion", "Ingestion"),
    ("indexing", "Indexing"),
    ("retrieval", "Retrieval"),
    ("benchmarking", "Benchmarking"),
    ("evaluation_datasets", "Eval datasets"),
    ("generation", "Generation"),
    ("faithfulness", "Faithfulness"),
    ("kpi_tracking", "KPI tracking"),
    ("tradeoffs", "Trade-offs"),
    ("roadmap", "Roadmap"),
)

SECTION_LABELS = {key: label for key, label in SECTIONS}


def documentation_section_path(section_key: str, *, docs_dir: Path = DOCS_DOCUMENTATION_DIR) -> Path:
    """Return the markdown file path for a documentation section."""
    return docs_dir / f"{section_key}.md"


def load_documentation_section(section_key: str, *, docs_dir: Path = DOCS_DOCUMENTATION_DIR) -> str:
    """Load markdown content for a documentation section."""
    path = documentation_section_path(section_key, docs_dir=docs_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Documentation file not found: {path}")
    return path.read_text(encoding="utf-8")


def _set_section(section_key: str) -> None:
    st.session_state[DOC_SECTION_KEY] = section_key


def _render_section_nav() -> str:
    if DOC_SECTION_KEY not in st.session_state:
        st.session_state[DOC_SECTION_KEY] = SECTIONS[0][0]

    active = st.session_state[DOC_SECTION_KEY]
    st.caption("Select a topic to explore the design and methodology behind this project.")

    row_size = 4
    for row_start in range(0, len(SECTIONS), row_size):
        cols = st.columns(row_size)
        for col, (key, label) in zip(cols, SECTIONS[row_start : row_start + row_size]):
            with col:
                st.button(
                    label,
                    key=f"doc_nav_{key}",
                    use_container_width=True,
                    type="primary" if key == active else "secondary",
                    on_click=_set_section,
                    args=(key,),
                )

    st.divider()
    return st.session_state[DOC_SECTION_KEY]


def _render_section(section_key: str) -> None:
    try:
        content = load_documentation_section(section_key)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.markdown(content, unsafe_allow_html=False)


def render_documentation_page() -> None:
    """Render the guided documentation page."""
    st.header("Documentation")
    st.markdown(
        "Architecture, methodology, and metrics — a guided walkthrough of how this RAG "
        "application was designed and evaluated."
    )

    section_key = _render_section_nav()
    if section_key not in SECTION_LABELS:
        section_key = SECTIONS[0][0]
    _render_section(section_key)
