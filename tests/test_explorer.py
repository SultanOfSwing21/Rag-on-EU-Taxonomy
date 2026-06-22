import pandas as pd
import pytest

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.dashboard import EvalDatasetSpec
from eu_taxonomy_rag.evaluation.explorer import (
    build_faq_question_options,
    chunk_sections,
    chunks_to_dataframe,
    filter_chunks,
    filter_eval_dataframe,
    truncate,
)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="faq-0001",
            question="What is the EU Taxonomy?",
            answer="A classification system.",
            metadata={"section": "General", "index": 1},
        ),
        Chunk(
            chunk_id="faq-0002",
            question="Who must report?",
            answer="Large companies under NFRD.",
            metadata={"section": "Reporting", "index": 2},
        ),
    ]


def test_truncate() -> None:
    assert truncate("short") == "short"
    assert truncate("x" * 100, max_len=20).endswith("…")


def test_build_faq_question_options(sample_chunks: list[Chunk]) -> None:
    options = build_faq_question_options(sample_chunks)

    assert len(options) == 2
    assert options[0].expected_chunk_ids == ("faq-0001",)
    assert options[0].source == "faq_original"


def test_chunks_to_dataframe(sample_chunks: list[Chunk]) -> None:
    df = chunks_to_dataframe(sample_chunks)

    assert len(df) == 2
    assert set(df["chunk_id"]) == {"faq-0001", "faq-0002"}


def test_filter_chunks(sample_chunks: list[Chunk]) -> None:
    filtered = filter_chunks(sample_chunks, sections=["Reporting"], search="")

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "faq-0002"


def test_chunk_sections(sample_chunks: list[Chunk]) -> None:
    assert chunk_sections(sample_chunks) == ["General", "Reporting"]


def test_filter_eval_dataframe() -> None:
    df = pd.DataFrame(
        [
            {"row": 0, "question": "simple q", "difficulty": "simple", "expected_chunk_ids": "faq-0001"},
            {"row": 1, "question": "complex q", "difficulty": "complex", "expected_chunk_ids": "faq-0002"},
        ]
    )

    filtered = filter_eval_dataframe(df, difficulty="simple", search="")
    assert len(filtered) == 1
    assert filtered.iloc[0]["difficulty"] == "simple"


def test_eval_items_to_dataframe_missing_file(tmp_path) -> None:
    from eu_taxonomy_rag.evaluation.explorer import eval_items_to_dataframe

    spec = EvalDatasetSpec("x", "X", tmp_path / "missing.jsonl", "missing")
    assert eval_items_to_dataframe(spec).empty
