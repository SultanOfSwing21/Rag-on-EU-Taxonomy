"""SQLite persistence for generation groundedness evaluations."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eu_taxonomy_rag.evaluation.generation_eval import GenerationEvaluationResult
from eu_taxonomy_rag.paths import DEFAULT_EVAL_DB

DEFAULT_DB_PATH = DEFAULT_EVAL_DB


@dataclass(frozen=True)
class StoredGenerationEvaluation:
    """One persisted chat interaction and its groundedness evaluation."""

    id: int
    created_at: str
    user_question: str
    generated_answer: str
    retrieval_method: str
    top_k: int
    candidate_k: int | None
    retrieved_chunk_ids: list[str]
    retrieved_chunk_texts: list[str]
    faithfulness_score: float
    contradiction_rate: float
    unsupported_rate: float
    num_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    best_claim_score: float
    avg_claim_score: float
    score_range: float
    claims: list[dict[str, Any]]
    warning: str | None = None
    evaluation_failed: bool = False
    abstention_response: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> StoredGenerationEvaluation:
        keys = row.keys()
        return cls(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            user_question=str(row["user_question"]),
            generated_answer=str(row["generated_answer"]),
            retrieval_method=str(row["retrieval_method"]),
            top_k=int(row["top_k"]),
            candidate_k=row["candidate_k"],
            retrieved_chunk_ids=json.loads(row["retrieved_chunk_ids"]),
            retrieved_chunk_texts=json.loads(row["retrieved_chunk_texts"]),
            faithfulness_score=float(row["faithfulness_score"]),
            contradiction_rate=float(row["contradiction_rate"]),
            unsupported_rate=float(row["unsupported_rate"]),
            num_claims=int(row["num_claims"]),
            supported_claims=int(row["supported_claims"]),
            contradicted_claims=int(row["contradicted_claims"]),
            unsupported_claims=int(row["unsupported_claims"]),
            best_claim_score=float(row["best_claim_score"]),
            avg_claim_score=float(row["avg_claim_score"]),
            score_range=float(row["score_range"]),
            claims=json.loads(row["claims_json"]),
            warning=row["warning"],
            evaluation_failed=bool(row["evaluation_failed"]),
            abstention_response=bool(row["abstention_response"]) if "abstention_response" in keys else False,
        )


@dataclass(frozen=True)
class GenerationMetricsSummary:
    """Aggregate metrics over stored generation evaluations."""

    num_evaluations: int
    average_faithfulness: float
    best_faithfulness: float
    worst_faithfulness: float
    score_range: float
    average_contradiction_rate: float
    average_unsupported_rate: float
    average_best_claim_score: float = 0.0
    average_avg_claim_score: float = 0.0
    average_claim_score_range: float = 0.0
    average_num_claims: float = 0.0
    average_supported_claims: float = 0.0
    average_contradicted_claims: float = 0.0
    average_unsupported_claims: float = 0.0
    average_top_k: float = 0.0
    average_candidate_k: float = 0.0


GENERATION_EVAL_KPI_COLUMNS: dict[str, str] = {
    "faithfulness_score": "Faithfulness",
    "contradiction_rate": "Contradiction rate",
    "unsupported_rate": "Unsupported rate",
    "num_claims": "Number of claims",
    "supported_claims": "Supported claims",
    "contradicted_claims": "Contradicted claims",
    "unsupported_claims": "Unsupported claims",
    "best_claim_score": "Best claim score",
    "avg_claim_score": "Average claim score",
    "score_range": "Claim score range",
    "top_k": "Top-k retrieval",
    "candidate_k": "Candidate-k (hybrid)",
}


def _evaluable_records(
    records: list[StoredGenerationEvaluation],
    *,
    include_abstention: bool = False,
    include_failed: bool = False,
) -> list[StoredGenerationEvaluation]:
    filtered: list[StoredGenerationEvaluation] = []
    for record in records:
        if record.evaluation_failed and not include_failed:
            continue
        if record.abstention_response and not include_abstention:
            continue
        filtered.append(record)
    return filtered


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_generation_metrics_summary_from_records(
    records: list[StoredGenerationEvaluation],
    *,
    include_abstention: bool = False,
    include_failed: bool = False,
) -> GenerationMetricsSummary:
    """Compute aggregate metrics over a list of stored evaluations."""
    evaluable = _evaluable_records(
        records,
        include_abstention=include_abstention,
        include_failed=include_failed,
    )
    if not evaluable:
        return GenerationMetricsSummary(
            num_evaluations=0,
            average_faithfulness=0.0,
            best_faithfulness=0.0,
            worst_faithfulness=0.0,
            score_range=0.0,
            average_contradiction_rate=0.0,
            average_unsupported_rate=0.0,
        )

    faithfulness_values = [record.faithfulness_score for record in evaluable]
    best = max(faithfulness_values)
    worst = min(faithfulness_values)
    candidate_values = [
        float(record.candidate_k) for record in evaluable if record.candidate_k is not None
    ]

    return GenerationMetricsSummary(
        num_evaluations=len(evaluable),
        average_faithfulness=_mean(faithfulness_values),
        best_faithfulness=best,
        worst_faithfulness=worst,
        score_range=best - worst,
        average_contradiction_rate=_mean([record.contradiction_rate for record in evaluable]),
        average_unsupported_rate=_mean([record.unsupported_rate for record in evaluable]),
        average_best_claim_score=_mean([record.best_claim_score for record in evaluable]),
        average_avg_claim_score=_mean([record.avg_claim_score for record in evaluable]),
        average_claim_score_range=_mean([record.score_range for record in evaluable]),
        average_num_claims=_mean([float(record.num_claims) for record in evaluable]),
        average_supported_claims=_mean([float(record.supported_claims) for record in evaluable]),
        average_contradicted_claims=_mean([float(record.contradicted_claims) for record in evaluable]),
        average_unsupported_claims=_mean([float(record.unsupported_claims) for record in evaluable]),
        average_top_k=_mean([float(record.top_k) for record in evaluable]),
        average_candidate_k=_mean(candidate_values),
    )


def kpi_value(record: StoredGenerationEvaluation, kpi_key: str) -> float | None:
    """Return a numeric KPI value for one stored evaluation."""
    if kpi_key not in GENERATION_EVAL_KPI_COLUMNS:
        raise KeyError(f"Unknown KPI key: {kpi_key}")
    if kpi_key == "candidate_k":
        return None if record.candidate_k is None else float(record.candidate_k)
    return float(getattr(record, kpi_key))


def records_to_kpi_dataframe(
    records: list[StoredGenerationEvaluation],
    kpi_keys: list[str],
):
    """Build a chronological dataframe for KPI trend charts."""
    import pandas as pd

    ordered = sorted(records, key=lambda record: (record.created_at, record.id))
    rows: list[dict[str, object]] = []
    for record in ordered:
        row: dict[str, object] = {
            "created_at": record.created_at,
            "evaluation_id": record.id,
        }
        for kpi_key in kpi_keys:
            row[kpi_key] = kpi_value(record, kpi_key)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["created_at", *kpi_keys])

    frame = pd.DataFrame(rows)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    indexed = frame.set_index("created_at").sort_index()
    return indexed[kpi_keys]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(generation_evaluations)")
    }
    if "abstention_response" not in columns:
        connection.execute(
            "ALTER TABLE generation_evaluations "
            "ADD COLUMN abstention_response INTEGER NOT NULL DEFAULT 0"
        )


def init_evaluation_db(db_path: Path | None = None) -> None:
    """Create evaluation tables when they do not exist."""
    path = db_path or DEFAULT_DB_PATH
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_question TEXT NOT NULL,
                generated_answer TEXT NOT NULL,
                retrieval_method TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                candidate_k INTEGER,
                retrieved_chunk_ids TEXT NOT NULL,
                retrieved_chunk_texts TEXT NOT NULL,
                faithfulness_score REAL NOT NULL,
                contradiction_rate REAL NOT NULL,
                unsupported_rate REAL NOT NULL,
                num_claims INTEGER NOT NULL,
                supported_claims INTEGER NOT NULL,
                contradicted_claims INTEGER NOT NULL,
                unsupported_claims INTEGER NOT NULL,
                best_claim_score REAL NOT NULL,
                avg_claim_score REAL NOT NULL,
                score_range REAL NOT NULL,
                claims_json TEXT NOT NULL,
                warning TEXT,
                evaluation_failed INTEGER NOT NULL DEFAULT 0,
                abstention_response INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        _ensure_schema(connection)
        connection.commit()


def save_generation_evaluation(
    *,
    user_question: str,
    generated_answer: str,
    retrieval_method: str,
    top_k: int,
    candidate_k: int | None,
    retrieved_chunk_ids: list[str],
    retrieved_chunk_texts: list[str],
    evaluation: GenerationEvaluationResult,
    db_path: Path | None = None,
) -> int:
    """Persist one chat interaction and return the new row id."""
    path = db_path or DEFAULT_DB_PATH
    init_evaluation_db(path)
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO generation_evaluations (
                created_at,
                user_question,
                generated_answer,
                retrieval_method,
                top_k,
                candidate_k,
                retrieved_chunk_ids,
                retrieved_chunk_texts,
                faithfulness_score,
                contradiction_rate,
                unsupported_rate,
                num_claims,
                supported_claims,
                contradicted_claims,
                unsupported_claims,
                best_claim_score,
                avg_claim_score,
                score_range,
                claims_json,
                warning,
                evaluation_failed,
                abstention_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                user_question,
                generated_answer,
                retrieval_method,
                top_k,
                candidate_k,
                json.dumps(retrieved_chunk_ids),
                json.dumps(retrieved_chunk_texts),
                evaluation.faithfulness_score,
                evaluation.contradiction_rate,
                evaluation.unsupported_rate,
                evaluation.num_claims,
                evaluation.supported_claims,
                evaluation.contradicted_claims,
                evaluation.unsupported_claims,
                evaluation.best_claim_score,
                evaluation.avg_claim_score,
                evaluation.score_range,
                json.dumps([claim.to_dict() for claim in evaluation.claims]),
                evaluation.warning,
                int(evaluation.evaluation_failed),
                int(evaluation.abstention_response),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def load_recent_evaluations(
    limit: int = 50,
    db_path: Path | None = None,
) -> list[StoredGenerationEvaluation]:
    """Return the most recent stored evaluations."""
    path = db_path or DEFAULT_DB_PATH
    init_evaluation_db(path)
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM generation_evaluations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [StoredGenerationEvaluation.from_row(row) for row in rows]


def load_evaluation_by_id(
    evaluation_id: int,
    db_path: Path | None = None,
) -> StoredGenerationEvaluation | None:
    """Load one stored evaluation by id."""
    path = db_path or DEFAULT_DB_PATH
    init_evaluation_db(path)
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM generation_evaluations WHERE id = ?",
            (evaluation_id,),
        ).fetchone()
    if row is None:
        return None
    return StoredGenerationEvaluation.from_row(row)


def compute_generation_metrics_summary(
    db_path: Path | None = None,
    *,
    limit: int | None = None,
    include_abstention: bool = False,
    include_failed: bool = False,
) -> GenerationMetricsSummary:
    """Compute aggregate metrics over stored evaluations."""
    path = db_path or DEFAULT_DB_PATH
    init_evaluation_db(path)
    if limit is None:
        with _connect(path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM generation_evaluations"
            ).fetchone()
        total = int(row["total"]) if row is not None else 0
        records = load_recent_evaluations(limit=total or 1, db_path=path)
    else:
        records = load_recent_evaluations(limit=limit, db_path=path)
    return compute_generation_metrics_summary_from_records(
        records,
        include_abstention=include_abstention,
        include_failed=include_failed,
    )
