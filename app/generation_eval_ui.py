"""Streamlit UI helpers for generation groundedness evaluation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from eu_taxonomy_rag.evaluation.generation_eval import (
    GenerationEvaluationResult,
    evaluate_generation,
    is_generation_eval_enabled,
)
from eu_taxonomy_rag.pipelines.rag_pipeline import RAGAnswer
from eu_taxonomy_rag.storage.evaluation_store import (
    GENERATION_EVAL_KPI_COLUMNS,
    StoredGenerationEvaluation,
    compute_generation_metrics_summary_from_records,
    kpi_value,
    load_recent_evaluations,
    records_to_kpi_dataframe,
    save_generation_evaluation,
)

METRICS_WINDOW_PRESETS = (10, 50, 100, 200, 500)


@st.cache_resource(show_spinner="Loading NLI model for groundedness evaluation…")
def _cached_nli_classifier():
    from eu_taxonomy_rag.evaluation.generation_eval import get_nli_classifier

    return get_nli_classifier()


def run_and_store_generation_evaluation(
    result: RAGAnswer,
    *,
    retrieval_method: str,
    top_k: int,
    candidate_k: int,
) -> GenerationEvaluationResult | None:
    """Evaluate groundedness for one answer and persist the interaction."""
    if not is_generation_eval_enabled():
        return None

    try:
        classifier = _cached_nli_classifier()
    except Exception as exc:
        st.warning(f"Groundedness evaluation unavailable: {exc}")
        return None

    try:
        evaluation = evaluate_generation(
            result.answer,
            [item.chunk for item in result.retrieval.chunks],
            classifier=classifier,
        )
    except Exception as exc:
        st.warning(f"Groundedness evaluation failed: {exc}")
        return None

    if evaluation.evaluation_failed:
        st.warning(evaluation.warning or "Groundedness evaluation failed.")
        return evaluation

    try:
        save_generation_evaluation(
            user_question=result.question,
            generated_answer=result.answer,
            retrieval_method=retrieval_method,
            top_k=top_k,
            candidate_k=candidate_k,
            retrieved_chunk_ids=result.chunk_ids,
            retrieved_chunk_texts=[item.chunk.text for item in result.retrieval.chunks],
            evaluation=evaluation,
        )
    except Exception as exc:
        st.warning(f"Could not save evaluation to SQLite: {exc}")

    return evaluation


def render_generation_evaluation(evaluation: GenerationEvaluationResult) -> None:
    """Display per-answer groundedness metrics and claim table."""
    st.markdown("### Groundedness evaluation")
    st.caption(
        "Diagnostic NLI-based faithfulness check against retrieved chunks. "
        "This is an approximate monitor, not a perfect judge."
    )

    if evaluation.abstention_response:
        st.info(evaluation.warning or "Abstention response — evaluation skipped.")
        st.caption("Faithfulness metrics are not applicable when the model declines to answer.")
        return

    if evaluation.warning:
        st.warning(evaluation.warning)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness", f"{evaluation.faithfulness_score:.0%}")
    c2.metric("Supported", evaluation.supported_claims)
    c3.metric("Contradicted", evaluation.contradicted_claims)
    c4.metric("Unsupported", evaluation.unsupported_claims)

    c5, c6, c7 = st.columns(3)
    c5.metric("Best claim score", f"{evaluation.best_claim_score:.3f}")
    c6.metric("Average claim score", f"{evaluation.avg_claim_score:.3f}")
    c7.metric("Score range", f"{evaluation.score_range:.3f}")

    if not evaluation.claims:
        return

    claims_df = pd.DataFrame(
        [
            {
                "claim": claim.claim,
                "label": claim.label,
                "confidence": claim.confidence,
                "claim_score": claim.claim_score,
                "best_chunk_id": claim.best_chunk_id or "",
            }
            for claim in evaluation.claims
        ]
    )
    st.dataframe(
        claims_df.style.format({"confidence": "{:.3f}", "claim_score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


def render_history_tab() -> None:
    st.subheader("Evaluation history")
    st.caption("Previous chat interactions with groundedness scores.")

    records = load_recent_evaluations(limit=50)
    if not records:
        st.info("No stored evaluations yet. Ask a question in the Chat tab.")
        return

    overview = pd.DataFrame(
        [
            {
                "id": record.id,
                "time": record.created_at,
                "question": record.user_question,
                "faithfulness": "N/A" if record.abstention_response else record.faithfulness_score,
                "claims": record.num_claims,
                "method": record.retrieval_method,
            }
            for record in records
        ]
    )
    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
    )

    selected_id = st.selectbox(
        "Inspect evaluation",
        options=[record.id for record in records],
        format_func=lambda evaluation_id: _history_label(
            next(record for record in records if record.id == evaluation_id)
        ),
    )
    selected = next(record for record in records if record.id == selected_id)
    _render_stored_evaluation(selected)


def render_metrics_tab() -> None:
    st.subheader("Generation metrics")
    st.caption("Aggregate groundedness KPIs over a sliding window of recent chat answers.")

    records = load_recent_evaluations(limit=1)
    if not records:
        st.info("No stored evaluations yet.")
        return

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        window_size = st.selectbox(
            "Answer window",
            options=list(METRICS_WINDOW_PRESETS),
            index=2,
            format_func=lambda value: f"Last {value} answers",
        )
    with c2:
        include_abstention = st.checkbox("Include abstention replies", value=False)
    with c3:
        include_failed = st.checkbox("Include failed evaluations", value=False)

    window_records = load_recent_evaluations(limit=window_size)
    chart_records = _evaluable_records_for_metrics(
        window_records,
        include_abstention=include_abstention,
        include_failed=include_failed,
    )

    if not chart_records:
        st.info("No evaluations match the selected window and filters.")
        return

    summary = compute_generation_metrics_summary_from_records(
        window_records,
        include_abstention=include_abstention,
        include_failed=include_failed,
    )
    st.caption(
        f"Showing **{summary.num_evaluations}** evaluable answer(s) "
        f"out of the last **{len(window_records)}** stored interaction(s)."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Evaluated answers", summary.num_evaluations)
    m2.metric("Average faithfulness", f"{summary.average_faithfulness:.0%}")
    m3.metric("Best faithfulness", f"{summary.best_faithfulness:.0%}")
    m4.metric("Worst faithfulness", f"{summary.worst_faithfulness:.0%}")

    m5, m6, m7 = st.columns(3)
    m5.metric("Faithfulness range", f"{summary.score_range:.0%}")
    m6.metric("Avg contradiction rate", f"{summary.average_contradiction_rate:.0%}")
    m7.metric("Avg unsupported rate", f"{summary.average_unsupported_rate:.0%}")

    with st.expander("All KPI averages for this window", expanded=False):
        summary_table = _kpi_summary_table(chart_records)
        if summary_table.empty:
            st.caption("No numeric KPIs available for this selection.")
        else:
            st.dataframe(
                summary_table.style.format({"Average": "{:.3f}", "Min": "{:.3f}", "Max": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )

    selected_kpis = st.multiselect(
        "KPIs to plot over time",
        options=list(GENERATION_EVAL_KPI_COLUMNS.keys()),
        default=["faithfulness_score", "contradiction_rate", "unsupported_rate"],
        format_func=lambda key: GENERATION_EVAL_KPI_COLUMNS[key],
    )

    if not selected_kpis:
        st.info("Select at least one KPI to display the trend chart.")
        return

    if len(chart_records) < 2:
        st.info("At least two evaluations are required to plot KPI trends.")
        return

    chart_df = records_to_kpi_dataframe(chart_records, selected_kpis)
    chart_df = chart_df[selected_kpis].rename(columns=GENERATION_EVAL_KPI_COLUMNS)
    st.markdown("**KPI evolution**")
    st.line_chart(chart_df, use_container_width=True)


def _evaluable_records_for_metrics(
    records: list[StoredGenerationEvaluation],
    *,
    include_abstention: bool,
    include_failed: bool,
) -> list[StoredGenerationEvaluation]:
    filtered: list[StoredGenerationEvaluation] = []
    for record in records:
        if record.evaluation_failed and not include_failed:
            continue
        if record.abstention_response and not include_abstention:
            continue
        filtered.append(record)
    return filtered


def _kpi_summary_table(records: list[StoredGenerationEvaluation]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for kpi_key, label in GENERATION_EVAL_KPI_COLUMNS.items():
        values = [
            value
            for record in records
            if (value := kpi_value(record, kpi_key)) is not None
        ]
        if not values:
            continue
        rows.append(
            {
                "KPI": label,
                "Average": sum(values) / len(values),
                "Min": min(values),
                "Max": max(values),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame


def _history_label(record: StoredGenerationEvaluation) -> str:
    question = record.user_question.strip().replace("\n", " ")
    if len(question) > 80:
        question = question[:77] + "…"
    return f"#{record.id} · {record.created_at[:19]} · {question}"


def _render_stored_evaluation(record: StoredGenerationEvaluation) -> None:
    from eu_taxonomy_rag.evaluation.generation_eval import ClaimEvaluation

    st.markdown("**Question**")
    st.write(record.user_question)
    st.markdown("**Answer**")
    st.write(record.generated_answer)
    st.caption(
        f"Method: `{record.retrieval_method}` · top-k={record.top_k} · "
        f"candidate-k={record.candidate_k}"
    )

    claims = tuple(
        ClaimEvaluation(
            claim=str(item["claim"]),
            label=str(item["label"]),
            confidence=float(item["confidence"]),
            claim_score=float(item.get("claim_score", 0.0)),
            best_chunk_id=item.get("best_chunk_id"),
            best_chunk_text=item.get("best_chunk_text"),
        )
        for item in record.claims
    )
    evaluation = GenerationEvaluationResult(
        faithfulness_score=record.faithfulness_score,
        contradiction_rate=record.contradiction_rate,
        unsupported_rate=record.unsupported_rate,
        num_claims=record.num_claims,
        supported_claims=record.supported_claims,
        contradicted_claims=record.contradicted_claims,
        unsupported_claims=record.unsupported_claims,
        best_claim_score=record.best_claim_score,
        avg_claim_score=record.avg_claim_score,
        score_range=record.score_range,
        claims=claims,
        warning=record.warning,
        evaluation_failed=record.evaluation_failed,
        abstention_response=record.abstention_response,
    )
    render_generation_evaluation(evaluation)

    with st.expander("Retrieved chunks"):
        for chunk_id, chunk_text in zip(record.retrieved_chunk_ids, record.retrieved_chunk_texts):
            st.markdown(f"`{chunk_id}`")
            st.text(chunk_text[:500] + ("…" if len(chunk_text) > 500 else ""))
