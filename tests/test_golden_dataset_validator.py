import numpy as np
import pytest

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.golden_dataset import GoldenQuestion
from eu_taxonomy_rag.evaluation.golden_dataset_validator import (
    clean_golden_dataset,
    validate_complex_question,
)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="faq-0001",
            question='What does "Forest management" mean for DNSH?',
            answer="Rules for forest management.",
            metadata={"section": "Climate Delegated Act", "index": 1},
        ),
        Chunk(
            chunk_id="faq-0002",
            question='How is "Rehabilitation and restoration of forests" treated?',
            answer="Restoration rules.",
            metadata={"section": "Climate Delegated Act", "index": 2},
        ),
        Chunk(
            chunk_id="faq-0003",
            question="What is the EU Taxonomy?",
            answer="A classification system.",
            metadata={"section": "EU Taxonomy - General", "index": 3},
        ),
    ]


@pytest.fixture
def embeddings(chunks: list[Chunk]) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.92, 0.08, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def id_to_index(chunks: list[Chunk]) -> dict[str, int]:
    return {chunk.chunk_id: index for index, chunk in enumerate(chunks)}


def test_rejects_duplicate_topic(chunks, embeddings, id_to_index) -> None:
    question = "What should companies know about both about considerations on nuclear waste and about considerations on nuclear waste?"
    outcome = validate_complex_question(
        question,
        ["faq-0001", "faq-0002"],
        {c.chunk_id: c for c in chunks},
        embeddings,
        id_to_index,
    )
    assert not outcome.is_valid
    assert "duplicate_topic" in outcome.reasons or "grammatically_broken" in outcome.reasons


def test_rejects_grammatically_broken(chunks, embeddings, id_to_index) -> None:
    question = "How do should reporting undertakings allocate turnover and are turnover defined?"
    outcome = validate_complex_question(
        question,
        ["faq-0001", "faq-0002"],
        {c.chunk_id: c for c in chunks},
        embeddings,
        id_to_index,
    )
    assert not outcome.is_valid


def test_accepts_natural_question(chunks, embeddings, id_to_index) -> None:
    question = "How do Forest management and Rehabilitation and restoration of forests interact under EU Taxonomy rules?"
    outcome = validate_complex_question(
        question,
        ["faq-0001", "faq-0002"],
        {c.chunk_id: c for c in chunks},
        embeddings,
        id_to_index,
    )
    assert outcome.is_valid


def test_clean_dataset_keeps_simple_and_cleans_complex(chunks, embeddings, id_to_index, monkeypatch) -> None:
    monkeypatch.setattr(
        "eu_taxonomy_rag.evaluation.golden_dataset_validator.compute_chunk_embeddings",
        lambda chunks: embeddings,
    )

    dataset = [
        GoldenQuestion("Simple Q?", ["faq-0001"], "simple"),
        GoldenQuestion(
            "How do should reporting undertakings allocate turnover and are turnover defined?",
            ["faq-0001", "faq-0002"],
            "complex",
        ),
        GoldenQuestion(
            "How do Forest management and Rehabilitation and restoration of forests interact under EU Taxonomy rules?",
            ["faq-0001", "faq-0002"],
            "complex",
        ),
    ]

    cleaned, report = clean_golden_dataset(dataset, chunks, seed=42)

    assert len([q for q in cleaned if q.difficulty == "simple"]) == 1
    assert report.final_complex >= 1
    assert report.final_total >= 2
