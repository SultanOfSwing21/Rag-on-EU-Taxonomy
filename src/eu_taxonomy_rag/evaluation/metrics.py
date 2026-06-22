"""Métriques d'évaluation du retrieval (Recall@K, MRR, etc.)."""

from dataclasses import dataclass
from typing import Any, Callable

from eu_taxonomy_rag.core.models import RetrievalResult

DEFAULT_RECALL_K_VALUES: tuple[int, ...] = (1, 3, 5)
DEFAULT_EVAL_GROUP_FIELDS: tuple[str, ...] = ("difficulty", "persona", "query_type")


def retrieved_chunk_ids(result: RetrievalResult) -> list[str]:
    """Extrait les chunk IDs dans l'ordre de ranking."""
    return [item.chunk.chunk_id for item in result.chunks]


def recall_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int,
) -> float:
    """Recall@K pour une requête.

    Recall = |expected ∩ top_k| / |expected|.
    Pour une question simple (1 chunk attendu), vaut 1.0 ou 0.0.
    Pour une question complexe (2–3 chunks), mesure la fraction retrouvée.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if not expected_ids:
        return 0.0

    expected = set(expected_ids)
    top_k = set(retrieved_ids[:k])
    hits = len(expected & top_k)
    return hits / len(expected)


def mean_recall_at_k(
    retrieved_ids_list: list[list[str]],
    expected_ids_list: list[list[str]],
    k: int,
) -> float:
    """Recall@K moyen sur un ensemble de requêtes."""
    if len(retrieved_ids_list) != len(expected_ids_list):
        raise ValueError("retrieved_ids_list and expected_ids_list must have the same length")
    if not retrieved_ids_list:
        return 0.0

    scores = [
        recall_at_k(retrieved, expected, k)
        for retrieved, expected in zip(retrieved_ids_list, expected_ids_list, strict=True)
    ]
    return sum(scores) / len(scores)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Reciprocal Rank pour une requête.

    Rang du premier chunk pertinent retrouvé : 1/rank, ou 0.0 si aucun hit.
    """
    if not expected_ids:
        return 0.0

    expected = set(expected_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_ids_list: list[list[str]],
    expected_ids_list: list[list[str]],
) -> float:
    """MRR (Mean Reciprocal Rank) sur un ensemble de requêtes."""
    if len(retrieved_ids_list) != len(expected_ids_list):
        raise ValueError("retrieved_ids_list and expected_ids_list must have the same length")
    if not retrieved_ids_list:
        return 0.0

    scores = [
        reciprocal_rank(retrieved, expected)
        for retrieved, expected in zip(retrieved_ids_list, expected_ids_list, strict=True)
    ]
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class RetrievalMetrics:
    """Recall@1/@3/@5 et MRR agrégés."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    num_queries: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "mrr": self.mrr,
            "num_queries": self.num_queries,
        }


# Alias rétrocompatible
RecallMetrics = RetrievalMetrics


def compute_retrieval_metrics(
    retrieved_ids_list: list[list[str]],
    expected_ids_list: list[list[str]],
) -> RetrievalMetrics:
    """Calcule Recall@1/@3/@5 et MRR sur un batch de requêtes."""
    return RetrievalMetrics(
        recall_at_1=mean_recall_at_k(retrieved_ids_list, expected_ids_list, k=1),
        recall_at_3=mean_recall_at_k(retrieved_ids_list, expected_ids_list, k=3),
        recall_at_5=mean_recall_at_k(retrieved_ids_list, expected_ids_list, k=5),
        mrr=mean_reciprocal_rank(retrieved_ids_list, expected_ids_list),
        num_queries=len(retrieved_ids_list),
    )


def compute_recall_metrics(
    retrieved_ids_list: list[list[str]],
    expected_ids_list: list[list[str]],
) -> RetrievalMetrics:
    """Alias de compute_retrieval_metrics."""
    return compute_retrieval_metrics(retrieved_ids_list, expected_ids_list)


def _validate_aligned_inputs(questions: list, retrieved_ids_list: list[list[str]]) -> None:
    if len(questions) != len(retrieved_ids_list):
        raise ValueError("questions and retrieved_ids_list must have the same length")


def _subset_metrics(
    questions: list,
    retrieved_ids_list: list[list[str]],
    indices: list[int],
) -> RetrievalMetrics:
    subset_retrieved = [retrieved_ids_list[i] for i in indices]
    subset_expected = [questions[i].expected_chunk_ids for i in indices]
    return compute_retrieval_metrics(subset_retrieved, subset_expected)


def compute_recall_by_field(
    questions: list,
    retrieved_ids_list: list[list[str]],
    field: str,
    *,
    key_fn: Callable[[Any], str | None] | None = None,
) -> dict[str, RetrievalMetrics]:
    """Recall@K groupé par un attribut des questions (ex. difficulty, persona, query_type)."""
    _validate_aligned_inputs(questions, retrieved_ids_list)

    grouped: dict[str, list[int]] = {}
    for index, question in enumerate(questions):
        value = key_fn(question) if key_fn is not None else getattr(question, field, None)
        if value is None:
            continue
        grouped.setdefault(str(value), []).append(index)

    return {
        group_value: _subset_metrics(questions, retrieved_ids_list, indices)
        for group_value, indices in sorted(grouped.items())
    }


@dataclass(frozen=True)
class EvaluationReport:
    """Rapport Recall@K + MRR global et ventilé par métadonnées."""

    overall: RetrievalMetrics
    by_difficulty: dict[str, RetrievalMetrics]
    by_persona: dict[str, RetrievalMetrics]
    by_query_type: dict[str, RetrievalMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_difficulty": {key: value.to_dict() for key, value in self.by_difficulty.items()},
            "by_persona": {key: value.to_dict() for key, value in self.by_persona.items()},
            "by_query_type": {key: value.to_dict() for key, value in self.by_query_type.items()},
        }


def compute_evaluation_report(
    questions: list,
    retrieved_ids_list: list[list[str]],
) -> EvaluationReport:
    """Rapport Recall@K complet, avec ventilations par métadonnées si disponibles."""
    _validate_aligned_inputs(questions, retrieved_ids_list)

    expected_ids_list = [q.expected_chunk_ids for q in questions]
    overall = compute_retrieval_metrics(retrieved_ids_list, expected_ids_list)

    return EvaluationReport(
        overall=overall,
        by_difficulty=compute_recall_by_field(questions, retrieved_ids_list, "difficulty"),
        by_persona=compute_recall_by_field(questions, retrieved_ids_list, "persona"),
        by_query_type=compute_recall_by_field(questions, retrieved_ids_list, "query_type"),
    )


def compute_recall_by_difficulty(
    questions: list,
    retrieved_ids_list: list[list[str]],
) -> dict[str, RetrievalMetrics | dict]:
    """Recall@K global et par difficulté (simple / complex)."""
    report = compute_evaluation_report(questions, retrieved_ids_list)
    return {
        "overall": report.overall,
        "by_difficulty": report.by_difficulty,
    }
