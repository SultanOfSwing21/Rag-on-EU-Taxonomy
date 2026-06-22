import json

import pytest

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.golden_dataset import GoldenQuestion
from eu_taxonomy_rag.evaluation.natural_dataset import (
    generate_natural_dataset,
    load_natural_dataset,
    save_natural_dataset,
    select_source_questions,
)


class FakeLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "multi-topic" in user_prompt.lower():
            return (
                "We are consolidating our Taxonomy reporting across several related activities. "
                "What requirements apply to each area and how should we present them together?"
            )
        return (
            "I'm preparing our first EU Taxonomy disclosure. "
            "Can you explain what this requirement means for our reporting team in practice?"
        )


@pytest.fixture
def source_dataset() -> list[GoldenQuestion]:
    simple = [
        GoldenQuestion(
            question=f"Simple source question {index}?",
            expected_chunk_ids=[f"faq-{index:04d}"],
            difficulty="simple",
        )
        for index in range(1, 6)
    ]
    complex_ = [
        GoldenQuestion(
            question=f"Complex source question {index}?",
            expected_chunk_ids=[f"faq-{index:04d}", f"faq-{index+1:04d}"],
            difficulty="complex",
        )
        for index in range(1, 4)
    ]
    return simple + complex_


@pytest.fixture
def chunks(source_dataset: list[GoldenQuestion]) -> list[Chunk]:
    chunk_ids = {cid for item in source_dataset for cid in item.expected_chunk_ids}
    return [
        Chunk(
            chunk_id=chunk_id,
            question=f"Official FAQ question for {chunk_id}",
            answer=f"Official answer for {chunk_id} with regulatory details.",
            metadata={"section": "EU Taxonomy - General", "index": int(chunk_id.split("-")[1])},
        )
        for chunk_id in sorted(chunk_ids)
    ]


def test_select_source_questions_is_diverse(source_dataset: list[GoldenQuestion]) -> None:
    selected = select_source_questions(source_dataset, n_simple=3, n_complex=2, seed=42)

    assert len(selected) == 5
    assert sum(1 for item in selected if item.difficulty == "simple") == 3
    assert sum(1 for item in selected if item.difficulty == "complex") == 2


def test_generate_natural_dataset_with_fake_llm(source_dataset: list[GoldenQuestion], chunks: list[Chunk]) -> None:
    sources = select_source_questions(source_dataset, n_simple=3, n_complex=2, seed=42)
    dataset, stats = generate_natural_dataset(sources, chunks, FakeLLM(), seed=42)

    assert len(dataset) == 5
    assert stats.generated_simple == 3
    assert stats.generated_complex == 2
    assert all(item.question for item in dataset)
    assert all(item.expected_chunk_ids for item in dataset)


def test_save_and_load_roundtrip(tmp_path, source_dataset: list[GoldenQuestion], chunks: list[Chunk]) -> None:
    sources = select_source_questions(source_dataset, n_simple=2, n_complex=1, seed=42)
    dataset, _ = generate_natural_dataset(sources, chunks, FakeLLM(), seed=42)
    path = tmp_path / "natural.jsonl"

    save_natural_dataset(dataset, path)
    loaded = load_natural_dataset(path)

    assert len(loaded) == 3
    assert loaded[0].question == dataset[0].question
    assert loaded[0].query_type == "natural_simple"
    assert loaded[0].persona is not None
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["difficulty"] in {"simple", "complex"}
    assert row["query_type"] in {"natural_simple", "natural_multihop"}
    assert row["persona"]


def test_load_natural_dataset_748_metadata() -> None:
    from eu_taxonomy_rag.evaluation.natural_dataset import NATURAL_DATASET_748_PATH, load_natural_dataset

    if not NATURAL_DATASET_748_PATH.exists():
        pytest.skip("natural_user_queries_748.jsonl not available")

    dataset = load_natural_dataset(NATURAL_DATASET_748_PATH)

    assert len(dataset) == 748
    assert all(item.query_type for item in dataset)
    assert all(item.persona for item in dataset)
    complex_items = [item for item in dataset if item.difficulty == "complex"]
    assert len(complex_items) == 100
    assert all(item.similarity_score is not None for item in complex_items)
    assert all(item.similarity_score is None for item in dataset if item.difficulty == "simple")
