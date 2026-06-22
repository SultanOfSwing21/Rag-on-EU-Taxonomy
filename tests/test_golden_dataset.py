import json

import numpy as np
import pytest

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.golden_dataset import (
    GenerationConfig,
    generate_golden_dataset,
    load_golden_dataset,
    save_golden_dataset,
)
from eu_taxonomy_rag.retrieval.embeddings import clear_embedding_cache


@pytest.fixture(autouse=True)
def reset_embedding_cache() -> None:
    clear_embedding_cache()
    yield
    clear_embedding_cache()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="faq-0001",
            question="What is the EU Taxonomy?",
            answer="A classification system.",
            metadata={"section": "EU Taxonomy - General", "index": 1},
        ),
        Chunk(
            chunk_id="faq-0002",
            question="Who must report Taxonomy alignment?",
            answer="Large companies under the NFRD.",
            metadata={"section": "Taxonomy-Alignment Reporting", "index": 2},
        ),
        Chunk(
            chunk_id="faq-0003",
            question="What are technical screening criteria?",
            answer="Conditions for Taxonomy alignment.",
            metadata={"section": "Climate Delegated Act", "index": 3},
        ),
        Chunk(
            chunk_id="faq-0004",
            question="How is DNSH assessed?",
            answer="Activities must not harm other objectives.",
            metadata={"section": "Climate Delegated Act", "index": 4},
        ),
    ]


@pytest.fixture
def mock_embeddings(monkeypatch: pytest.MonkeyPatch, sample_chunks: list[Chunk]) -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.2, 0.9, 0.0],
            [0.9, 0.1, 0.0],
            [0.85, 0.15, 0.0],
        ],
        dtype=np.float32,
    )

    class FakeModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
            return vectors[: len(texts)]

    monkeypatch.setattr(
        "eu_taxonomy_rag.retrieval.embeddings.get_embedding_model",
        lambda model_name=None: FakeModel(),
    )


def test_generate_golden_dataset_schema_and_validation(
    sample_chunks: list[Chunk], mock_embeddings: None
) -> None:
    config = GenerationConfig(seed=42, target_simple=8, target_complex=4, min_neighbor_similarity=0.3)
    dataset, stats = generate_golden_dataset(sample_chunks, config)

    assert len(dataset) == 12
    assert stats.simple_count == 8
    assert stats.complex_count == 4

    for item in dataset:
        assert item.question
        assert item.expected_chunk_ids
        assert item.difficulty in {"simple", "complex"}
        for chunk_id in item.expected_chunk_ids:
            assert chunk_id.startswith("faq-")


def test_complex_questions_reference_multiple_chunks(
    sample_chunks: list[Chunk], mock_embeddings: None
) -> None:
    config = GenerationConfig(seed=42, target_simple=4, target_complex=6, min_neighbor_similarity=0.3)
    dataset, _ = generate_golden_dataset(sample_chunks, config)
    complex_ = [q for q in dataset if q.difficulty == "complex"]

    assert complex_
    assert all(len(q.expected_chunk_ids) >= 2 for q in complex_)


def test_save_and_load_roundtrip(tmp_path, sample_chunks: list[Chunk], mock_embeddings: None) -> None:
    config = GenerationConfig(seed=42, target_simple=5, target_complex=3, min_neighbor_similarity=0.3)
    dataset, _ = generate_golden_dataset(sample_chunks, config)
    path = tmp_path / "golden.jsonl"

    save_golden_dataset(dataset, path)
    loaded = load_golden_dataset(path)

    assert len(loaded) == len(dataset)
    assert loaded[0].question == dataset[0].question
    assert loaded[0].expected_chunk_ids == dataset[0].expected_chunk_ids


def test_generation_is_reproducible(sample_chunks: list[Chunk], mock_embeddings: None) -> None:
    config = GenerationConfig(seed=99, target_simple=6, target_complex=3, min_neighbor_similarity=0.3)
    first, _ = generate_golden_dataset(sample_chunks, config)
    second, _ = generate_golden_dataset(sample_chunks, config)

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
