import pytest

from eu_taxonomy_rag.core.models import Chunk, RetrievalResult, RetrievedChunk
from eu_taxonomy_rag.evaluation.metrics import (
    compute_evaluation_report,
    compute_recall_by_difficulty,
    compute_recall_by_field,
    compute_recall_metrics,
    compute_retrieval_metrics,
    mean_reciprocal_rank,
    mean_recall_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieved_chunk_ids,
)
from eu_taxonomy_rag.evaluation.golden_dataset import GoldenQuestion
from eu_taxonomy_rag.evaluation.natural_dataset import NaturalQuery


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        question=f"Q {chunk_id}",
        answer=f"A {chunk_id}",
    )


def test_recall_at_k_single_hit() -> None:
    assert recall_at_k(["faq-0001", "faq-0002"], ["faq-0001"], k=1) == 1.0
    assert recall_at_k(["faq-0002", "faq-0001"], ["faq-0001"], k=2) == 1.0


def test_recall_at_k_single_miss() -> None:
    assert recall_at_k(["faq-0002", "faq-0003"], ["faq-0001"], k=3) == 0.0


def test_recall_at_k_partial_multi_chunk() -> None:
    expected = ["faq-0001", "faq-0002", "faq-0003"]
    retrieved = ["faq-0001", "faq-0009", "faq-0002", "faq-0010"]

    assert recall_at_k(retrieved, expected, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, expected, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, expected, k=4) == pytest.approx(2 / 3)


def test_recall_at_k_empty_expected() -> None:
    assert recall_at_k(["faq-0001"], [], k=5) == 0.0


def test_recall_at_k_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(["faq-0001"], ["faq-0001"], k=0)


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank(["faq-0001", "faq-0002"], ["faq-0001"]) == 1.0


def test_reciprocal_rank_later_position() -> None:
    assert reciprocal_rank(["faq-0009", "faq-0001"], ["faq-0001"]) == pytest.approx(0.5)
    assert reciprocal_rank(["faq-0009", "faq-0008", "faq-0001"], ["faq-0001"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_miss() -> None:
    assert reciprocal_rank(["faq-0009", "faq-0010"], ["faq-0001"]) == 0.0
    assert reciprocal_rank([], ["faq-0001"]) == 0.0


def test_reciprocal_rank_multi_expected_uses_first_hit() -> None:
    retrieved = ["faq-0009", "faq-0002", "faq-0001"]
    expected = ["faq-0001", "faq-0002"]

    assert reciprocal_rank(retrieved, expected) == pytest.approx(0.5)


def test_mean_reciprocal_rank() -> None:
    retrieved = [
        ["faq-0001"],
        ["faq-0009", "faq-0002"],
        ["faq-0009", "faq-0010"],
    ]
    expected = [
        ["faq-0001"],
        ["faq-0002"],
        ["faq-0003"],
    ]

    assert mean_reciprocal_rank(retrieved, expected) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mean_recall_at_k() -> None:
    retrieved = [
        ["faq-0001", "faq-0002"],
        ["faq-0003", "faq-0004"],
    ]
    expected = [
        ["faq-0001"],
        ["faq-0003"],
    ]

    assert mean_recall_at_k(retrieved, expected, k=1) == 1.0
    assert mean_recall_at_k(retrieved, expected, k=2) == 1.0


def test_mean_recall_at_k_mixed_scores() -> None:
    retrieved = [
        ["faq-0001"],
        ["faq-0009"],
    ]
    expected = [
        ["faq-0001", "faq-0002"],
        ["faq-0003"],
    ]

    assert mean_recall_at_k(retrieved, expected, k=1) == pytest.approx(0.25)


def test_compute_recall_metrics() -> None:
    retrieved = [
        ["faq-0001", "faq-0002", "faq-0003"],
        ["faq-0010", "faq-0011", "faq-0012"],
    ]
    expected = [
        ["faq-0001"],
        ["faq-0010", "faq-0011"],
    ]

    metrics = compute_recall_metrics(retrieved, expected)

    assert metrics.recall_at_1 == pytest.approx(0.75)
    assert metrics.recall_at_3 == pytest.approx(1.0)
    assert metrics.recall_at_5 == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx(1.0)
    assert metrics.num_queries == 2
    assert metrics.to_dict()["recall@3"] == pytest.approx(1.0)
    assert metrics.to_dict()["mrr"] == pytest.approx(1.0)


def test_compute_retrieval_metrics_includes_mrr() -> None:
    retrieved = [
        ["faq-0009", "faq-0001"],
        ["faq-0010", "faq-0011"],
    ]
    expected = [
        ["faq-0001"],
        ["faq-0010", "faq-0011"],
    ]

    metrics = compute_retrieval_metrics(retrieved, expected)

    assert metrics.mrr == pytest.approx((0.5 + 1.0) / 2)


def test_retrieved_chunk_ids_from_result() -> None:
    result = RetrievalResult(
        query="test",
        chunks=(
            RetrievedChunk(chunk=_chunk("faq-0002"), score=0.9, rank=1),
            RetrievedChunk(chunk=_chunk("faq-0001"), score=0.8, rank=2),
        ),
    )

    assert retrieved_chunk_ids(result) == ["faq-0002", "faq-0001"]


def test_compute_recall_by_difficulty() -> None:
    questions = [
        GoldenQuestion(question="q1", expected_chunk_ids=["faq-0001"], difficulty="simple"),
        GoldenQuestion(question="q2", expected_chunk_ids=["faq-0002"], difficulty="simple"),
        GoldenQuestion(
            question="q3",
            expected_chunk_ids=["faq-0003", "faq-0004"],
            difficulty="complex",
        ),
    ]
    retrieved = [
        ["faq-0001"],
        ["faq-0009"],
        ["faq-0003", "faq-0009", "faq-0004"],
    ]

    report = compute_recall_by_difficulty(questions, retrieved)

    assert report["overall"].recall_at_1 == pytest.approx(0.5)
    assert report["by_difficulty"]["simple"].recall_at_1 == pytest.approx(0.5)
    assert report["by_difficulty"]["complex"].recall_at_3 == pytest.approx(1.0)


def test_compute_recall_by_field_persona() -> None:
    questions = [
        NaturalQuery(
            question="q1",
            expected_chunk_ids=["faq-0001"],
            difficulty="simple",
            query_type="natural_simple",
            persona="CFO",
        ),
        NaturalQuery(
            question="q2",
            expected_chunk_ids=["faq-0002"],
            difficulty="simple",
            query_type="natural_simple",
            persona="CFO",
        ),
        NaturalQuery(
            question="q3",
            expected_chunk_ids=["faq-0003"],
            difficulty="simple",
            query_type="natural_simple",
            persona="auditor",
        ),
    ]
    retrieved = [
        ["faq-0001"],
        ["faq-0009"],
        ["faq-0003"],
    ]

    by_persona = compute_recall_by_field(questions, retrieved, "persona")

    assert by_persona["CFO"].recall_at_1 == pytest.approx(0.5)
    assert by_persona["CFO"].num_queries == 2
    assert by_persona["auditor"].recall_at_1 == pytest.approx(1.0)


def test_compute_evaluation_report_with_metadata() -> None:
    questions = [
        NaturalQuery(
            question="q1",
            expected_chunk_ids=["faq-0001"],
            difficulty="simple",
            query_type="natural_simple",
            persona="CFO",
        ),
        NaturalQuery(
            question="q2",
            expected_chunk_ids=["faq-0002", "faq-0003"],
            difficulty="complex",
            query_type="natural_multihop",
            persona="auditor",
            similarity_score=0.91,
        ),
    ]
    retrieved = [
        ["faq-0001"],
        ["faq-0002", "faq-0009", "faq-0003"],
    ]

    report = compute_evaluation_report(questions, retrieved)

    assert report.overall.recall_at_1 == pytest.approx(0.75)
    assert report.overall.mrr == pytest.approx(1.0)
    assert report.by_difficulty["simple"].recall_at_1 == pytest.approx(1.0)
    assert report.by_difficulty["complex"].recall_at_3 == pytest.approx(1.0)
    assert report.by_persona["CFO"].num_queries == 1
    assert report.by_query_type["natural_multihop"].recall_at_3 == pytest.approx(1.0)
    assert report.to_dict()["overall"]["mrr"] == pytest.approx(1.0)
    assert report.to_dict()["by_persona"]["CFO"]["recall@1"] == pytest.approx(1.0)
