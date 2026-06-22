from pathlib import Path

import pytest

from eu_taxonomy_rag.evaluation.generation_eval import (
    ClaimEvaluation,
    GenerationEvaluationResult,
    build_generation_evaluation_result,
)
from eu_taxonomy_rag.storage.evaluation_store import (
    compute_generation_metrics_summary,
    init_evaluation_db,
    load_evaluation_by_id,
    load_recent_evaluations,
    records_to_kpi_dataframe,
    save_generation_evaluation,
)


def _sample_evaluation() -> GenerationEvaluationResult:
    return build_generation_evaluation_result(
        [
            ClaimEvaluation("Supported claim.", "supported", 0.9, 0.9, "faq-0001"),
            ClaimEvaluation("Unsupported claim.", "not_enough_info", 0.6, 0.2, "faq-0002"),
        ]
    )


def test_save_and_load_generation_evaluation(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    evaluation = _sample_evaluation()

    row_id = save_generation_evaluation(
        user_question="What is the EU Taxonomy?",
        generated_answer="It is a classification system.",
        retrieval_method="hybrid_minilm",
        top_k=5,
        candidate_k=20,
        retrieved_chunk_ids=["faq-0001"],
        retrieved_chunk_texts=["Question: Q\nAnswer: A"],
        evaluation=evaluation,
        db_path=db_path,
    )

    loaded = load_evaluation_by_id(row_id, db_path=db_path)
    assert loaded is not None
    assert loaded.user_question == "What is the EU Taxonomy?"
    assert loaded.faithfulness_score == pytest.approx(0.5)
    assert loaded.num_claims == 2
    assert len(loaded.claims) == 2


def test_load_recent_evaluations_returns_newest_first(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    evaluation = _sample_evaluation()

    first_id = save_generation_evaluation(
        user_question="First question",
        generated_answer="First answer with enough text for storage.",
        retrieval_method="bm25",
        top_k=3,
        candidate_k=10,
        retrieved_chunk_ids=["faq-0001"],
        retrieved_chunk_texts=["chunk"],
        evaluation=evaluation,
        db_path=db_path,
    )
    second_id = save_generation_evaluation(
        user_question="Second question",
        generated_answer="Second answer with enough text for storage.",
        retrieval_method="bm25",
        top_k=3,
        candidate_k=10,
        retrieved_chunk_ids=["faq-0002"],
        retrieved_chunk_texts=["chunk"],
        evaluation=evaluation,
        db_path=db_path,
    )

    recent = load_recent_evaluations(limit=10, db_path=db_path)
    assert [record.id for record in recent] == [second_id, first_id]


def test_compute_generation_metrics_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    high = build_generation_evaluation_result(
        [ClaimEvaluation("High quality claim.", "supported", 0.95, 0.95)]
    )
    from eu_taxonomy_rag.evaluation.generation_eval import build_abstention_evaluation_result

    abstention = build_abstention_evaluation_result()

    save_generation_evaluation(
        user_question="Q1",
        generated_answer="Answer one with enough claim text.",
        retrieval_method="bm25",
        top_k=5,
        candidate_k=20,
        retrieved_chunk_ids=["faq-0001"],
        retrieved_chunk_texts=["chunk"],
        evaluation=high,
        db_path=db_path,
    )
    save_generation_evaluation(
        user_question="Q2",
        generated_answer="I cannot answer this question from the available context.",
        retrieval_method="bm25",
        top_k=5,
        candidate_k=20,
        retrieved_chunk_ids=["faq-0002"],
        retrieved_chunk_texts=["chunk"],
        evaluation=abstention,
        db_path=db_path,
    )

    summary = compute_generation_metrics_summary(db_path=db_path)
    assert summary.num_evaluations == 1
    assert summary.average_faithfulness == pytest.approx(1.0)


def test_compute_generation_metrics_summary_with_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    high = build_generation_evaluation_result(
        [ClaimEvaluation("High quality claim.", "supported", 0.95, 0.95)]
    )
    low = build_generation_evaluation_result(
        [ClaimEvaluation("Low quality claim here.", "not_enough_info", 0.7, 0.2)]
    )

    for index in range(3):
        save_generation_evaluation(
            user_question=f"Q{index}",
            generated_answer=f"Answer {index} with enough claim text.",
            retrieval_method="bm25",
            top_k=5,
            candidate_k=20,
            retrieved_chunk_ids=[f"faq-{index:04d}"],
            retrieved_chunk_texts=["chunk"],
            evaluation=high if index < 2 else low,
            db_path=db_path,
        )

    summary = compute_generation_metrics_summary(db_path=db_path, limit=2)
    assert summary.num_evaluations == 2
    assert summary.average_faithfulness == pytest.approx(0.5)


def test_records_to_kpi_dataframe_orders_chronologically(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    evaluation = build_generation_evaluation_result(
        [ClaimEvaluation("Supported claim here.", "supported", 0.9, 0.9)]
    )
    save_generation_evaluation(
        user_question="Q1",
        generated_answer="Answer one with enough claim text.",
        retrieval_method="bm25",
        top_k=5,
        candidate_k=20,
        retrieved_chunk_ids=["faq-0001"],
        retrieved_chunk_texts=["chunk"],
        evaluation=evaluation,
        db_path=db_path,
    )
    save_generation_evaluation(
        user_question="Q2",
        generated_answer="Answer two with enough claim text.",
        retrieval_method="hybrid_minilm",
        top_k=3,
        candidate_k=15,
        retrieved_chunk_ids=["faq-0002"],
        retrieved_chunk_texts=["chunk"],
        evaluation=evaluation,
        db_path=db_path,
    )

    records = load_recent_evaluations(limit=2, db_path=db_path)
    frame = records_to_kpi_dataframe(records, ["faithfulness_score", "top_k", "candidate_k"])

    assert len(frame) == 2
    assert list(frame.columns) == ["faithfulness_score", "top_k", "candidate_k"]
    assert frame["top_k"].tolist() == [5.0, 3.0]


def test_init_evaluation_db_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    init_evaluation_db(db_path)
    init_evaluation_db(db_path)
    assert db_path.exists()
