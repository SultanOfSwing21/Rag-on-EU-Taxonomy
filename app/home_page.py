"""Welcome / home page — first impression when users open the app."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from documentation_page import load_documentation_section

_HERO_STYLE = """
<style>
.eu-taxonomy-hero {
    padding: 2.75rem 2.25rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #0f766e 100%);
    color: #f8fafc;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 40px rgba(15, 23, 42, 0.18);
}
.eu-taxonomy-hero h1 {
    margin: 0;
    font-size: 2.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.eu-taxonomy-hero p {
    margin: 0.85rem 0 0;
    font-size: 1.12rem;
    opacity: 0.93;
    max-width: 52rem;
    line-height: 1.55;
}
.eu-taxonomy-hero .badge-row {
    margin-top: 1.35rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.eu-taxonomy-hero .badge {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
}
.eu-taxonomy-metric-card {
    padding: 1.1rem 1rem;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    text-align: center;
    min-height: 5.5rem;
}
.eu-taxonomy-metric-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f766e;
    line-height: 1.1;
}
.eu-taxonomy-metric-card .label {
    margin-top: 0.35rem;
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.3;
}
.eu-taxonomy-pipeline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 0.35rem 0.25rem;
    padding: 1.25rem 0.75rem;
    margin: 0.5rem 0 1rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: #ffffff;
}
.eu-taxonomy-pipeline-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 0.35rem 0.25rem;
    width: 100%;
}
.eu-taxonomy-pipeline-step {
    padding: 0.55rem 0.85rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #0f172a;
    background: #ecfeff;
    border: 1px solid #99f6e4;
    white-space: nowrap;
}
.eu-taxonomy-pipeline-step.eval {
    background: #f0fdf4;
    border-color: #86efac;
}
.eu-taxonomy-pipeline-step.gen {
    background: #eff6ff;
    border-color: #93c5fd;
}
.eu-taxonomy-pipeline-arrow {
    color: #64748b;
    font-size: 1rem;
    font-weight: 700;
    padding: 0 0.1rem;
}
.eu-taxonomy-pipeline-branch {
    width: 100%;
    margin-top: 0.65rem;
    padding-top: 0.75rem;
    border-top: 1px dashed rgba(148, 163, 184, 0.55);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 0.35rem 0.25rem;
}
.eu-taxonomy-pipeline-caption {
    width: 100%;
    text-align: center;
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.35rem;
}
</style>
"""


def _render_hero() -> None:
    st.markdown(
        """
<div class="eu-taxonomy-hero">
  <h1>EU Taxonomy RAG</h1>
  <p>
    Grounded Q&amp;A on official EU Taxonomy FAQs — retrieval you can benchmark,
    answers you can trace, faithfulness you can measure.
  </p>
  <div class="badge-row">
    <span class="badge">RAG</span>
    <span class="badge">Hybrid retrieval</span>
    <span class="badge">Recall@K · MRR</span>
    <span class="badge">NLI faithfulness</span>
    <span class="badge">No API key required for benchmarks</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_cards(*, num_chunks: int | None = None) -> None:
    chunk_value = str(num_chunks) if num_chunks is not None else "324"
    cards = [
        (chunk_value, "Official FAQ chunks"),
        ("5", "Retrieval methods"),
        ("1,448", "Labelled eval queries"),
        ("2", "Retrieval + answer eval"),
    ]
    cols = st.columns(4)
    for col, (value, label) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
<div class="eu-taxonomy-metric-card">
  <div class="value">{value}</div>
  <div class="label">{label}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def _render_pipeline_diagram() -> None:
    st.markdown(
        """
<div class="eu-taxonomy-pipeline">
  <div class="eu-taxonomy-pipeline-row">
    <span class="eu-taxonomy-pipeline-step">FAQ Markdown</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step">324 Chunks</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step">BM25 + Dense</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step">Hybrid retrieval</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step gen">RAG prompt</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step gen">LLM answer</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step eval">NLI faithfulness</span>
  </div>
  <div class="eu-taxonomy-pipeline-branch">
    <span class="eu-taxonomy-pipeline-caption">Parallel evaluation path</span>
    <span class="eu-taxonomy-pipeline-step">324 Chunks</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step eval">Benchmark datasets</span>
    <span class="eu-taxonomy-pipeline-arrow">→</span>
    <span class="eu-taxonomy-pipeline-step eval">Recall@K · MRR</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_quick_actions(navigate_to: Callable[[str], None]) -> None:
    st.markdown("#### Get started")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.button(
            "Ask a question",
            use_container_width=True,
            on_click=navigate_to,
            args=("Chatbot",),
        )
    with c2:
        st.button(
            "Run a benchmark",
            use_container_width=True,
            on_click=navigate_to,
            args=("Benchmark",),
        )
    with c3:
        st.button(
            "Compare retrieval",
            use_container_width=True,
            on_click=navigate_to,
            args=("Interactive test",),
        )
    with c4:
        st.button(
            "Read the docs",
            use_container_width=True,
            on_click=navigate_to,
            args=("Documentation",),
        )


def render_home_page(
    *,
    navigate_to: Callable[[str], None],
    num_chunks: int | None = None,
) -> None:
    """Render the welcome landing page."""
    st.markdown(_HERO_STYLE, unsafe_allow_html=True)
    _render_hero()
    _render_metric_cards(num_chunks=num_chunks)
    st.markdown("<br>", unsafe_allow_html=True)

    try:
        body = load_documentation_section("home")
    except FileNotFoundError:
        st.warning("Welcome content not found at `docs/documentation/home.md`.")
        return

    pipeline_heading = "### Pipeline at a glance"
    under_the_hood_heading = "### Under the hood"
    if pipeline_heading in body:
        before_pipeline, rest = body.split(pipeline_heading, 1)
        st.markdown(before_pipeline)
        st.markdown(pipeline_heading)
        _render_pipeline_diagram()
        if under_the_hood_heading in rest:
            _, after_pipeline = rest.split(under_the_hood_heading, 1)
            st.markdown(under_the_hood_heading + after_pipeline)
        else:
            st.markdown(rest)
    else:
        st.markdown(body)
    st.divider()
    _render_quick_actions(navigate_to)
