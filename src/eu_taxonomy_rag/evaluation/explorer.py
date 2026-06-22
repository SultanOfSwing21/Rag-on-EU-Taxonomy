"""Helpers to browse FAQ chunks and evaluation datasets in the UI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.dashboard import AVAILABLE_DATASETS, EvalDatasetSpec
from eu_taxonomy_rag.evaluation.runner import load_eval_dataset


@dataclass(frozen=True)
class QuestionOption:
    """Selectable question for interactive retrieval tests."""

    key: str
    label: str
    question: str
    source: str
    expected_chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def truncate(text: str, max_len: int = 90) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def build_faq_question_options(chunks: list[Chunk]) -> list[QuestionOption]:
    options: list[QuestionOption] = []
    for chunk in chunks:
        section = chunk.metadata.get("section", "Unknown")
        options.append(
            QuestionOption(
                key=f"faq::{chunk.chunk_id}",
                label=f"{chunk.chunk_id} · [{section}] {truncate(chunk.question)}",
                question=chunk.question,
                source="faq_original",
                expected_chunk_ids=(chunk.chunk_id,),
                metadata={"section": section, **chunk.metadata},
            )
        )
    return options


def build_eval_question_options(spec: EvalDatasetSpec) -> list[QuestionOption]:
    if not spec.path.exists():
        return []

    items = load_eval_dataset(spec.path)
    options: list[QuestionOption] = []
    for index, item in enumerate(items):
        extra: dict[str, Any] = {"difficulty": item.difficulty}
        if hasattr(item, "persona") and item.persona:
            extra["persona"] = item.persona
        if hasattr(item, "query_type") and item.query_type:
            extra["query_type"] = item.query_type
        if hasattr(item, "similarity_score") and item.similarity_score is not None:
            extra["similarity_score"] = item.similarity_score

        expected = ", ".join(item.expected_chunk_ids)
        options.append(
            QuestionOption(
                key=f"eval::{spec.key}::{index}",
                label=f"[{item.difficulty}] {truncate(item.question)} → {expected}",
                question=item.question,
                source=spec.key,
                expected_chunk_ids=tuple(item.expected_chunk_ids),
                metadata=extra,
            )
        )
    return options


def load_all_eval_question_options() -> list[QuestionOption]:
    options: list[QuestionOption] = []
    for spec in AVAILABLE_DATASETS:
        if spec.exists:
            options.extend(build_eval_question_options(spec))
    return options


def chunks_to_dataframe(chunks: list[Chunk]) -> pd.DataFrame:
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "section": chunk.metadata.get("section", "Unknown"),
            "index": chunk.metadata.get("index", ""),
            "question": chunk.question,
            "answer_preview": truncate(chunk.answer, 120),
            "answer_length": len(chunk.answer),
        }
        for chunk in chunks
    ]
    return pd.DataFrame(rows)


def eval_items_to_dataframe(spec: EvalDatasetSpec) -> pd.DataFrame:
    if not spec.path.exists():
        return pd.DataFrame()

    items = load_eval_dataset(spec.path)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        row: dict[str, Any] = {
            "row": index,
            "question": item.question,
            "difficulty": item.difficulty,
            "expected_chunk_ids": ", ".join(item.expected_chunk_ids),
            "num_expected": len(item.expected_chunk_ids),
        }
        if hasattr(item, "persona"):
            row["persona"] = item.persona
        if hasattr(item, "query_type"):
            row["query_type"] = item.query_type
        if hasattr(item, "similarity_score"):
            row["similarity_score"] = item.similarity_score
        rows.append(row)
    return pd.DataFrame(rows)


def chunk_sections(chunks: list[Chunk]) -> list[str]:
    sections = sorted({chunk.metadata.get("section", "Unknown") for chunk in chunks})
    return sections


def filter_chunks(
    chunks: list[Chunk],
    *,
    sections: list[str] | None = None,
    search: str = "",
) -> list[Chunk]:
    filtered = chunks
    if sections:
        allowed = set(sections)
        filtered = [chunk for chunk in filtered if chunk.metadata.get("section", "Unknown") in allowed]
    if search.strip():
        needle = search.strip().lower()
        filtered = [
            chunk
            for chunk in filtered
            if needle in chunk.question.lower()
            or needle in chunk.answer.lower()
            or needle in chunk.chunk_id.lower()
        ]
    return filtered


def filter_eval_dataframe(df: pd.DataFrame, *, difficulty: str | None = None, search: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    result = df
    if difficulty and difficulty != "All":
        result = result[result["difficulty"] == difficulty]
    if search.strip():
        needle = search.strip().lower()
        mask = result["question"].str.lower().str.contains(needle, na=False)
        if "persona" in result.columns:
            mask = mask | result["persona"].fillna("").str.lower().str.contains(needle, na=False)
        if "expected_chunk_ids" in result.columns:
            mask = mask | result["expected_chunk_ids"].str.lower().str.contains(needle, na=False)
        result = result[mask]
    return result
