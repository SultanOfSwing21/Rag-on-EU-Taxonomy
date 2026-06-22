from pathlib import Path

import pytest

from eu_taxonomy_rag.cache.chunk_cache import (
    build_chunks_from_file,
    load_chunks,
    load_or_build_chunks,
    save_chunks,
)
from eu_taxonomy_rag.core.models import Chunk


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="faq-0001",
            question="What is the EU Taxonomy?",
            answer="An classification system for sustainable activities.",
            metadata={"section": "General", "index": 1, "source": "test.md"},
        ),
        Chunk(
            chunk_id="faq-0002",
            question="Who must report?",
            answer="Large companies under the NFRD.",
            metadata={"section": "Reporting", "index": 2, "source": "test.md"},
        ),
    ]


def test_save_and_load_chunks(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    cache_path = tmp_path / "chunks.jsonl"

    save_chunks(sample_chunks, cache_path)
    loaded = load_chunks(cache_path)

    assert len(loaded) == 2
    assert loaded[0].chunk_id == "faq-0001"
    assert loaded[0].question == sample_chunks[0].question
    assert loaded[0].answer == sample_chunks[0].answer
    assert loaded[0].metadata == sample_chunks[0].metadata
    assert loaded[1].chunk_id == "faq-0002"


def test_load_or_build_chunks_uses_cache(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    cache_path = tmp_path / "chunks.jsonl"
    save_chunks(sample_chunks, cache_path)

    chunks = load_or_build_chunks(
        faq_path=tmp_path / "missing.md",
        cache_path=cache_path,
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "faq-0001"


def test_load_or_build_chunks_rebuilds_when_forced(
    tmp_path: Path, sample_chunks: list[Chunk]
) -> None:
    cache_path = tmp_path / "chunks.jsonl"
    save_chunks(sample_chunks, cache_path)

    chunks = load_or_build_chunks(
        faq_path="data/taxonomy_faqs_cleaned.md",
        cache_path=cache_path,
        force_rebuild=True,
    )

    assert len(chunks) == 324
    assert chunks[0].chunk_id == "faq-0001"
    assert "section" in chunks[0].metadata


def test_build_chunks_from_file() -> None:
    chunks = build_chunks_from_file("data/taxonomy_faqs_cleaned.md")

    assert len(chunks) == 324
    assert len({chunk.chunk_id for chunk in chunks}) == 324
    assert all(chunk.question and chunk.answer for chunk in chunks)
    assert all(chunk.metadata.get("section") for chunk in chunks)
