#!/usr/bin/env python3
"""Streamlit dashboard — EU Taxonomy RAG retrieval evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot_page import ensure_chatbot_settings_loaded, render_chatbot_page

from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.evaluation.dashboard import (
    AVAILABLE_DATASETS,
    METRIC_COLUMNS,
    PROJECT_ROOT,
    RESULTS_DIR,
    build_segment_comparison_df,
    build_segment_filtered_comparison_df,
    collect_segment_values,
    filter_benchmark_results,
    get_dataset_spec,
    list_saved_results,
    load_latest_benchmark_results,
    load_saved_result,
    method_label,
    pivot_metric,
    saved_result_to_df,
)
from eu_taxonomy_rag.evaluation.runner import (
    build_indexes_for_methods,
    default_output_path,
    run_retrieval_evaluation,
)
from eu_taxonomy_rag.pipelines.index_manager import DEFAULT_INDEX_DIR
from eu_taxonomy_rag.retrieval.embeddings import is_sentence_transformers_available
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod, available_retrieval_methods
from eu_taxonomy_rag.retrieval.retriever import Retriever

st.set_page_config(
    page_title="EU Taxonomy RAG",
    page_icon="📊",
    layout="wide",
)

AVAILABLE_METHODS = available_retrieval_methods()
METHOD_OPTIONS = [(method.value, method_label(method.value)) for method in AVAILABLE_METHODS]
DEFAULT_METHOD_VALUES = [method.value for method in AVAILABLE_METHODS]
INDEX_DIR = PROJECT_ROOT / DEFAULT_INDEX_DIR


def render_environment_notice() -> None:
    import sys

    from eu_taxonomy_rag.retrieval.dense_index import dense_index_backend, is_faiss_available

    st.sidebar.caption(f"Dense backend: **{dense_index_backend()}**")

    if is_sentence_transformers_available():
        from eu_taxonomy_rag.retrieval.embeddings import embedding_device

        st.sidebar.caption(f"Embedding device: **{embedding_device()}**")
        if not is_faiss_available():
            st.sidebar.info(
                "FAISS is not installed. Dense retrieval uses a NumPy cosine-similarity index, "
                "which is sufficient for this FAQ corpus."
            )
        return

    st.sidebar.warning(
        "Dense/hybrid retrieval is unavailable: `sentence-transformers` is not installed "
        f"(Python {sys.version_info.major}.{sys.version_info.minor}). "
        "Only **BM25** can run. Use Python 3.10–3.12 and run: "
        "`pip install -e \".[ui]\"`"
    )


def filter_selected_methods(method_values: list[str]) -> list[RetrievalMethod]:
    allowed = {method.value for method in AVAILABLE_METHODS}
    selected = [RetrievalMethod(value) for value in method_values if value in allowed]
    if method_values and not selected:
        st.error(
            "Selected methods require sentence-transformers. "
            "Switch to BM25 or install dependencies with Python 3.10–3.12."
        )
    return selected


@st.cache_resource(show_spinner="Loading FAQ chunks…")
def get_chunks():
    return load_or_build_chunks()


@st.cache_resource(show_spinner="Warming retrieval indexes…")
def warm_indexes(method_values: tuple[str, ...]) -> str:
    """Build/load indexes once per Streamlit session."""
    chunks = load_or_build_chunks()
    methods = [RetrievalMethod(value) for value in method_values]
    build_indexes_for_methods(chunks, methods, base_dir=INDEX_DIR)
    return "ready"


def _render_metric_heatmap(df: pd.DataFrame, metric_key: str, metric_label: str) -> None:
    pivot = pivot_metric(df, metric_key)
    if pivot.empty:
        st.info("No data to display.")
        return

    styled = pivot.style.format("{:.3f}").background_gradient(cmap="Greens", axis=None, vmin=0, vmax=1)
    st.dataframe(styled, use_container_width=True)


def _render_grouped_bars(df: pd.DataFrame, metric_key: str, metric_label: str) -> None:
    if df.empty:
        return
    chart_df = df.pivot_table(
        index="method_label",
        columns="dataset",
        values=metric_key,
        aggfunc="first",
    ).reset_index()
    st.caption(metric_label)
    st.bar_chart(chart_df, x="method_label", y=list(chart_df.columns[1:]), stack=False)


def _ensure_benchmark_results_loaded(selected_specs: list) -> None:
    """Load latest saved JSON files for datasets not yet in memory."""
    cache: dict = st.session_state.setdefault("benchmark_results", {})
    sources: dict = st.session_state.setdefault("benchmark_sources", {})

    missing_specs = [spec for spec in selected_specs if spec.key not in cache]
    if not missing_specs:
        return

    loaded, new_sources = load_latest_benchmark_results(missing_specs)
    if not loaded:
        return

    cache.update(loaded)
    for key, path in new_sources.items():
        sources[key] = str(path)
    st.session_state["benchmark_from_disk"] = True


def _filter_benchmark_view(
    dataset_keys: list[str],
    method_values: list[str],
    *,
    difficulties: list[str] | None = None,
    personas: list[str] | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    results = st.session_state.get("benchmark_results") or {}
    filtered = filter_benchmark_results(
        results,
        dataset_keys=dataset_keys,
        method_values=method_values,
    )
    if not filtered:
        return {}, None
    return filtered, build_segment_filtered_comparison_df(
        filtered,
        difficulties=difficulties,
        personas=personas,
    )


def page_benchmark() -> None:
    st.header("Multi-dataset benchmark")
    st.caption(
        "Compare Recall@K and MRR across evaluation datasets and retrieval methods "
        "(BM25, dense MiniLM/MPNet, hybrid RRF)."
    )

    available = [spec for spec in AVAILABLE_DATASETS if spec.exists]
    if not available:
        st.error("No evaluation datasets found in data/evaluation/.")
        return

    col_cfg, col_run = st.columns([2, 1])

    with col_cfg:
        dataset_keys = st.multiselect(
            "Datasets",
            options=[spec.key for spec in available],
            default=[spec.key for spec in available],
            format_func=lambda key: next(spec.label for spec in available if spec.key == key),
        )
        for spec in available:
            if spec.key in dataset_keys:
                st.caption(f"**{spec.label}** — {spec.description}")

        method_values = st.multiselect(
            "Retrieval methods",
            options=[value for value, _ in METHOD_OPTIONS],
            default=DEFAULT_METHOD_VALUES,
            format_func=method_label,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            k = st.number_input("Top-k retrieval", min_value=1, max_value=20, value=5)
        with c2:
            candidate_k = st.number_input("Candidate-k (hybrid)", min_value=5, max_value=50, value=20)
        with c3:
            limit = st.number_input("Question limit (0 = all)", min_value=0, value=0)
        limit_value = None if limit == 0 else int(limit)

    with col_run:
        st.subheader("Run")
        run_button = st.button("Run benchmark", type="primary", use_container_width=True)
        save_results = st.checkbox("Save JSON results", value=True)
        st.caption(
            "Results update when you change dataset, method, or segment filters. "
            "Missing datasets are loaded from the latest saved JSON on disk. "
            "**Run benchmark** forces a fresh evaluation."
        )

    selected_specs = [spec for spec in available if spec.key in dataset_keys]
    _ensure_benchmark_results_loaded(selected_specs)

    preview_results = filter_benchmark_results(
        st.session_state.get("benchmark_results") or {},
        dataset_keys=dataset_keys,
        method_values=method_values,
    )
    available_difficulties = collect_segment_values(preview_results, "difficulty")
    available_personas = collect_segment_values(preview_results, "persona")

    with col_cfg:
        c4, c5 = st.columns(2)
        with c4:
            difficulty_filter = st.multiselect(
                "Difficulty",
                options=available_difficulties,
                default=[],
                help="Leave empty for overall metrics. Filter by simple or complex questions.",
                disabled=not available_difficulties,
            )
        with c5:
            persona_filter = st.multiselect(
                "Persona / role",
                options=available_personas,
                default=[],
                help="Leave empty for overall metrics. Filter by user role (natural dataset).",
                disabled=not available_personas,
            )

    if difficulty_filter and persona_filter:
        st.warning(
            "Difficulty and persona filters cannot be combined. "
            "The persona filter is applied; clear it to filter by difficulty."
        )
        difficulty_filter = []

    if st.session_state.get("benchmark_from_disk") and st.session_state.get("benchmark_sources"):
        sources = st.session_state["benchmark_sources"]
        lines = [
            f"- **{get_dataset_spec(key).label}**: `{path}`"
            for key, path in sources.items()
            if key in dataset_keys
        ]
        if lines:
            st.info("In-memory results (latest saved JSON per dataset):\n" + "\n".join(lines))

    if run_button:
        if not dataset_keys or not method_values:
            st.warning("Select at least one dataset and one method.")
            return

        selected_specs = [spec for spec in available if spec.key in dataset_keys]
        methods = filter_selected_methods(method_values)
        if not methods:
            return

        status = st.empty()
        progress = st.progress(0.0, text="Warming indexes…")
        warm_indexes(tuple(sorted(method_values)))
        chunks = get_chunks()

        results: dict = {}
        total_steps = len(selected_specs) * len(methods)
        completed_steps = 0

        def on_progress(dataset_name: str, method: RetrievalMethod, step: int, total: int) -> None:
            nonlocal completed_steps
            completed_steps += 1
            progress.progress(
                completed_steps / total_steps,
                text=f"{dataset_name} · {method_label(method.value)} ({completed_steps}/{total_steps})",
            )

        for spec in selected_specs:
            status.info(f"Evaluating **{spec.label}** ({len(methods)} methods, batch mode)…")
            results[spec.key] = run_retrieval_evaluation(
                dataset_path=spec.path,
                methods=methods,
                k=int(k),
                candidate_k=int(candidate_k),
                index_dir=INDEX_DIR,
                limit=limit_value,
                chunks=chunks,
                build_indexes=False,
                progress_callback=on_progress,
                dataset_label=spec.label,
            )
            if save_results:
                output_path = default_output_path(spec.path, results_dir=RESULTS_DIR)
                results[spec.key].save(output_path)

        progress.progress(1.0, text="Done.")
        status.empty()
        cache = st.session_state.setdefault("benchmark_results", {})
        cache.update(results)
        sources = st.session_state.setdefault("benchmark_sources", {})
        if save_results:
            sources.update(
                {
                    spec.key: str(default_output_path(spec.path, results_dir=RESULTS_DIR))
                    for spec in selected_specs
                }
            )
        st.session_state["benchmark_from_disk"] = False
        st.success(f"Benchmark completed on {len(selected_specs)} dataset(s).")

    if not dataset_keys or not method_values:
        st.info("Select at least one dataset and one retrieval method.")
        return

    results, df = _filter_benchmark_view(
        dataset_keys,
        method_values,
        difficulties=difficulty_filter or None,
        personas=persona_filter or None,
    )

    if df is None or df.empty:
        cached = st.session_state.get("benchmark_results") or {}
        missing_datasets = [key for key in dataset_keys if key not in cached]
        if missing_datasets:
            st.info(
                "No saved results yet for: "
                + ", ".join(get_dataset_spec(key).label for key in missing_datasets)
                + ". Run a benchmark or check `data/evaluation/results/`."
            )
        else:
            if difficulty_filter or persona_filter:
                st.info("No results for the selected difficulty/persona filters.")
            else:
                st.info("No results for the current dataset/method filters.")
        return

    st.subheader("Overview")
    if difficulty_filter:
        st.caption(f"Metrics filtered by difficulty: **{', '.join(difficulty_filter)}**")
    elif persona_filter:
        st.caption(f"Metrics filtered by persona: **{', '.join(persona_filter)}**")

    overview_columns = ["dataset", "method_label", "recall@1", "recall@3", "recall@5", "mrr", "num_queries"]
    overview_labels = ["Dataset", "Method", "Recall@1", "Recall@3", "Recall@5", "MRR", "Questions"]
    if "segment_label" in df.columns:
        overview_columns.insert(1, "segment_label")
        overview_labels.insert(1, "Segment")

    display_df = df[overview_columns].copy()
    display_df.columns = overview_labels
    st.dataframe(
        display_df.style.format(
            {"Recall@1": "{:.3f}", "Recall@3": "{:.3f}", "Recall@5": "{:.3f}", "MRR": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    tab_heat, tab_bars, tab_segments = st.tabs(["Heatmaps", "Bar charts", "Breakdowns"])

    with tab_heat:
        for metric_key, metric_label in METRIC_COLUMNS:
            st.markdown(f"**{metric_label}** — method × dataset")
            _render_metric_heatmap(df, metric_key, metric_label)

    with tab_bars:
        for metric_key, metric_label in METRIC_COLUMNS:
            _render_grouped_bars(df, metric_key, metric_label)

    with tab_segments:
        if results is None:
            return

        segment = st.selectbox(
            "Break down by",
            options=["difficulty", "persona", "query_type"],
            format_func=lambda value: {
                "difficulty": "Difficulty (simple / complex)",
                "persona": "Persona (natural datasets)",
                "query_type": "Query type",
            }[value],
        )
        segment_df = build_segment_comparison_df(results, segment)
        if segment_df.empty:
            st.info("No metadata available for this breakdown on the selected datasets.")
        else:
            selected_dataset = st.selectbox(
                "Dataset",
                options=sorted(segment_df["dataset"].unique()),
            )
            filtered = segment_df[segment_df["dataset"] == selected_dataset]
            for metric_key, metric_label in METRIC_COLUMNS:
                st.markdown(f"**{metric_label}** — {selected_dataset}")
                pivot = filtered.pivot_table(
                    index="method_label",
                    columns="segment",
                    values=metric_key,
                    aggfunc="first",
                )
                if not pivot.empty:
                    st.dataframe(
                        pivot.style.format("{:.3f}").background_gradient(cmap="Greens", axis=None, vmin=0, vmax=1),
                        use_container_width=True,
                    )


def page_interactive() -> None:
    from eu_taxonomy_rag.evaluation.explorer import (
        QuestionOption,
        build_eval_question_options,
        build_faq_question_options,
        truncate,
    )

    st.header("Interactive query test")
    st.caption("Compare retrieval results side by side. Pick a FAQ question, an evaluation query, or type your own.")

    chunks = get_chunks()
    available_eval_specs = [spec for spec in AVAILABLE_DATASETS if spec.exists]

    source_mode = st.radio(
        "Question source",
        options=["custom", "faq", "eval"],
        horizontal=True,
        format_func=lambda value: {
            "custom": "Custom question",
            "faq": "Original FAQ",
            "eval": "Evaluation dataset",
        }[value],
    )

    selected_option: QuestionOption | None = None
    query = ""

    if source_mode == "custom":
        query = st.text_area(
            "Question",
            placeholder="e.g. How should undertakings report Taxonomy-aligned CapEx?",
            height=100,
            key="interactive_custom_query",
        )
    elif source_mode == "faq":
        faq_options = build_faq_question_options(chunks)
        faq_sections = sorted({opt.metadata.get("section", "Unknown") for opt in faq_options})
        section_filter = st.multiselect("Filter by section", options=faq_sections)
        filtered_faq = [
            opt
            for opt in faq_options
            if not section_filter or opt.metadata.get("section") in section_filter
        ]
        if not filtered_faq:
            st.warning("No FAQ questions match the selected filters.")
            return
        selected_key = st.selectbox(
            "FAQ question",
            options=[opt.key for opt in filtered_faq],
            format_func=lambda key: next(opt.label for opt in filtered_faq if opt.key == key),
        )
        selected_option = next(opt for opt in filtered_faq if opt.key == selected_key)
        query = selected_option.question
        if selected_option:
            st.markdown(f"## Selected question: \n\n### {query}\n\n")
            st.caption(f"Expected chunks: `{', '.join(selected_option.expected_chunk_ids)}`")
    else:
        if not available_eval_specs:
            st.error("No evaluation datasets found.")
            return
        dataset_key = st.selectbox(
            "Dataset",
            options=[spec.key for spec in available_eval_specs],
            format_func=lambda key: next(spec.label for spec in available_eval_specs if spec.key == key),
        )
        spec = next(spec for spec in available_eval_specs if spec.key == dataset_key)
        eval_options = build_eval_question_options(spec)
        difficulty_filter = st.selectbox("Difficulty", options=["All", "simple", "complex"])
        if difficulty_filter != "All":
            eval_options = [opt for opt in eval_options if opt.metadata.get("difficulty") == difficulty_filter]
        search_filter = st.text_input("Search in questions", placeholder="Filter by keyword…")
        if search_filter.strip():
            needle = search_filter.strip().lower()
            eval_options = [opt for opt in eval_options if needle in opt.question.lower()]
        if not eval_options:
            st.warning("No questions match the selected filters.")
            return
        selected_key = st.selectbox(
            "Evaluation question",
            options=[opt.key for opt in eval_options],
            format_func=lambda key: next(opt.label for opt in eval_options if opt.key == key),
        )
        selected_option = next(opt for opt in eval_options if opt.key == selected_key)
        query = selected_option.question
        #st.text_area("Selected question", value=query, height=100, disabled=True)

        if selected_option:
            st.markdown(f"## Selected question: \n\n### {query}\n\n")
            st.caption(f"Expected chunks: `{', '.join(selected_option.expected_chunk_ids)}`")

    method_values = st.multiselect(
        "Methods",
        options=[value for value, _ in METHOD_OPTIONS],
        default=DEFAULT_METHOD_VALUES[:3] if len(DEFAULT_METHOD_VALUES) >= 3 else DEFAULT_METHOD_VALUES,
        format_func=method_label,
    )
    k = st.slider("Top-k", min_value=1, max_value=10, value=5)

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Enter or select a question.")
            return
        if not method_values:
            st.warning("Select at least one method.")
            return

        methods = filter_selected_methods(method_values)
        if not methods:
            return
        warm_indexes(tuple(sorted(method_values)))
        expected = set(selected_option.expected_chunk_ids) if selected_option else set()

        columns = st.columns(len(methods))
        for column, method in zip(columns, methods):
            with column:
                st.subheader(method_label(method.value))
                result = Retriever(
                    chunks=chunks,
                    method=method,
                    base_dir=INDEX_DIR,
                ).retrieve(query.strip(), k=k)

                if not result.chunks:
                    st.info("No results.")
                    continue

                if expected:
                    hits = {item.chunk.chunk_id for item in result.chunks} & expected
                    st.caption(f"Expected hit in top-{k}: **{len(hits)}/{len(expected)}**")

                for item in result.chunks:
                    is_expected = item.chunk.chunk_id in expected
                    hit = "✅ expected" if is_expected else ""
                    st.markdown(f"**#{item.rank}** `{item.chunk.chunk_id}` — score `{item.score:.4f}` {hit}")
                    st.markdown(
                        f"*{truncate(item.chunk.question, 120)}*"
                    )
                    with st.expander("Answer"):
                        st.write(item.chunk.answer[:800] + ("…" if len(item.chunk.answer) > 800 else ""))


def page_data_explorer() -> None:
    from eu_taxonomy_rag.evaluation.explorer import (
        chunk_sections,
        chunks_to_dataframe,
        eval_items_to_dataframe,
        filter_chunks,
        filter_eval_dataframe,
        truncate,
    )

    st.header("Data explorer")
    st.caption("Browse indexed FAQ chunks and evaluation dataset questions with metadata.")

    view = st.radio(
        "Explore",
        options=["chunks", "eval"],
        horizontal=True,
        format_func=lambda value: "FAQ chunks" if value == "chunks" else "Evaluation datasets",
    )

    if view == "chunks":
        chunks = get_chunks()
        st.metric("Total chunks", len(chunks))
        sections = chunk_sections(chunks)
        st.caption(f"Sections covered: **{len(sections)}**")

        c1, c2 = st.columns(2)
        with c1:
            section_filter = st.multiselect("Section", options=sections)
        with c2:
            search = st.text_input("Search chunks", placeholder="Question, answer, or chunk id…")

        filtered = filter_chunks(chunks, sections=section_filter or None, search=search)
        st.caption(f"Showing **{len(filtered)}** / {len(chunks)} chunks")
        st.dataframe(chunks_to_dataframe(filtered), use_container_width=True, hide_index=True)

        if filtered:
            chunk_ids = [chunk.chunk_id for chunk in filtered]
            selected_id = st.selectbox("Inspect chunk", options=chunk_ids)
            chunk = next(item for item in filtered if item.chunk_id == selected_id)
            st.markdown(f"### `{chunk.chunk_id}`")
            meta_cols = st.columns(3)
            meta_cols[0].metric("Section", chunk.metadata.get("section", "Unknown"))
            meta_cols[1].metric("FAQ index", chunk.metadata.get("index", "—"))
            meta_cols[2].metric("Answer length", len(chunk.answer))
            st.markdown("**Question**")
            st.write(chunk.question)
            st.markdown("**Answer**")
            st.write(chunk.answer)
            with st.expander("Embedding text"):
                st.code(chunk.text)
    else:
        available = [spec for spec in AVAILABLE_DATASETS if spec.exists]
        if not available:
            st.error("No evaluation datasets found.")
            return

        spec = st.selectbox(
            "Dataset",
            options=available,
            format_func=lambda item: item.label,
        )
        df = eval_items_to_dataframe(spec)
        if df.empty:
            st.warning("Dataset is empty.")
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Questions", len(df))
        m2.metric("Simple", int((df["difficulty"] == "simple").sum()))
        m3.metric("Complex", int((df["difficulty"] == "complex").sum()))

        c1, c2 = st.columns(2)
        with c1:
            difficulty = st.selectbox("Difficulty", options=["All", "simple", "complex"])
        with c2:
            search = st.text_input("Search questions", placeholder="Keyword, persona, chunk id…")

        filtered_df = filter_eval_dataframe(df, difficulty=difficulty, search=search)
        display_cols = [col for col in filtered_df.columns if col != "row"]
        st.caption(f"Showing **{len(filtered_df)}** / {len(df)} questions")
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        if not filtered_df.empty:
            row_options = filtered_df["row"].tolist()
            selected_row = st.selectbox(
                "Inspect question",
                options=row_options,
                format_func=lambda row: truncate(
                    filtered_df.loc[filtered_df["row"] == row, "question"].iloc[0],
                    100,
                ),
            )
            row = filtered_df.loc[filtered_df["row"] == selected_row].iloc[0]
            st.markdown("**Question**")
            st.write(row["question"])
            st.markdown("**Metadata**")
            meta = {col: row[col] for col in filtered_df.columns if col not in {"question", "row"}}
            st.json({key: (None if pd.isna(value) else value) for key, value in meta.items()})



def page_saved_results() -> None:
    st.header("Saved results")
    st.caption("Load and compare JSON exports from the benchmark or CLI script.")

    saved_files = list_saved_results()
    if not saved_files:
        st.info(f"No files in `{RESULTS_DIR}`. Run a benchmark with saving enabled.")
        return

    selected_files = st.multiselect(
        "JSON files",
        options=saved_files,
        default=saved_files[: min(3, len(saved_files))],
        format_func=lambda path: path.name,
    )

    if not selected_files:
        return

    frames: list[pd.DataFrame] = []
    for path in selected_files:
        payload = load_saved_result(path)
        frame = saved_result_to_df(payload)
        frame["source_file"] = path.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    st.dataframe(
        combined[
            ["source_file", "dataset", "method_label", "recall@1", "recall@3", "recall@5", "mrr", "num_queries"]
        ].style.format({"recall@1": "{:.3f}", "recall@3": "{:.3f}", "recall@5": "{:.3f}", "mrr": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    for metric_key, metric_label in METRIC_COLUMNS:
        st.markdown(f"**{metric_label}**")
        pivot = combined.pivot_table(
            index="method_label",
            columns="source_file",
            values=metric_key,
            aggfunc="first",
        )
        if not pivot.empty:
            st.dataframe(
                pivot.style.format("{:.3f}").background_gradient(cmap="Greens", axis=None, vmin=0, vmax=1),
                use_container_width=True,
            )


def main() -> None:
    ensure_chatbot_settings_loaded()

    st.title("EU Taxonomy RAG")
    st.markdown(
        "Retrieval evaluation, data exploration, and **RAG chatbot** for EU Taxonomy FAQs."
    )

    render_environment_notice()

    page = st.sidebar.radio(
        "Navigation",
        options=["Chatbot", "Benchmark", "Interactive test", "Data explorer"] #, "Saved results"],
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Available datasets**")
    for spec in AVAILABLE_DATASETS:
        status = "✅" if spec.exists else "❌"
        st.sidebar.caption(f"{status} {spec.label}")

    if page == "Chatbot":
        render_chatbot_page(
            get_chunks=get_chunks,
            warm_indexes=warm_indexes,
            method_label=method_label,
            filter_selected_methods=filter_selected_methods,
        )
    elif page == "Benchmark":
        page_benchmark()
    elif page == "Interactive test":
        page_interactive()
    elif page == "Data explorer":
        page_data_explorer()
    else:
        page_saved_results()


if __name__ == "__main__":
    main()
